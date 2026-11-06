"""
src/lifestage/train_nhts.py
Life-stage classifier trained on REAL NHTS 2022 survey data.

Predicts the likely vehicle class (sedan, SUV/crossover, pickup, minivan/van)
from the main driver's age group, household income band, household size,
and area type. Probabilities feed the recommender's life-stage score.

Key choices:
- Unit = one household vehicle; label = its body type (VEHTYPE).
- Age = the vehicle's MAIN driver (WHOMAIN joined to the person file).
- Survey weights (WTHHFIN) in training AND evaluation, so results represent
  US households, not just the people who answered the survey.
- Only vehicles 5 years old or newer: NHTS records every car a household
  owns, but the app recommends what to BUY now. Recent vehicles reflect
  current choices (SUVs 45%) better than the whole fleet (sedans 46%).
- Train/test split grouped by HOUSEID: one household's cars never appear in
  both sets (same idea as the VIN-grouped split for the price model).

Needs data/raw/vehv2pub.csv and data/raw/perv2pub.csv.

Run:
    python -m src.lifestage.train_nhts
"""
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import sklearn
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

RAW = Path("data/raw")
MODELS = Path("models")
SEED = 42
MAX_VEHICLE_AGE = 5   # years; "recently bought" proxy

# NHTS 2022 VEHTYPE: 1 car, 2 van, 3 SUV, 4 pickup; 5 other truck, 6 RV,
# 7 motorcycle, 97 other are dropped (not cars this app recommends)
VEHTYPE_TO_CLASS = {1: "sedan", 2: "minivan_van", 3: "suv_crossover", 4: "pickup"}

# HBHUR (5 levels). Mapping inferred from the data: code 1 is ~87% rural,
# 2 is low-density/small town, 4 is lower-density urban, 3 and 5 are dense
# urban. Verify against the NHTS 2022 codebook.
HBHUR_TO_AREA = {1: "rural", 2: "rural", 3: "urban", 4: "suburban", 5: "urban"}

# Same encodings the API already uses (api/predictor.py builds this vector)
AGE_MAP = {"18-34": 0, "35-54": 1, "55-64": 2, "65+": 3}
INCOME_MAP = {"low": 0, "middle": 1, "high": 2}
AREA_MAP = {"urban": 0, "suburban": 1, "rural": 2}


def income_band(code):
    # HHFAMINC: 1-4 = under $35k, 5-7 = $35k-$100k, 8-11 = $100k+; negatives = refused
    if 1 <= code <= 4:
        return "low"
    if 5 <= code <= 7:
        return "middle"
    if 8 <= code <= 11:
        return "high"
    return None


def age_group(age):
    if age < 35:
        return "18-34"
    if age < 55:
        return "35-54"
    if age < 65:
        return "55-64"
    return "65+"


def load_nhts() -> pd.DataFrame:
    veh = pd.read_csv(RAW / "vehv2pub.csv", usecols=[
        "HOUSEID", "VEHTYPE", "WHOMAIN", "HHFAMINC", "HHSIZE", "HBHUR",
        "WTHHFIN", "VEHAGE"])
    per = pd.read_csv(RAW / "perv2pub.csv", usecols=["HOUSEID", "PERSONID", "R_AGE"])
    log.info(f"Vehicles: {len(veh):,}  households: {veh.HOUSEID.nunique():,}")

    df = veh.merge(per, left_on=["HOUSEID", "WHOMAIN"],
                   right_on=["HOUSEID", "PERSONID"], how="inner")
    log.info(f"After joining main driver's age: {len(df):,}")

    df["vehicle_class"] = df["VEHTYPE"].map(VEHTYPE_TO_CLASS)
    df["income_band"] = df["HHFAMINC"].map(income_band)
    df["area_type"] = df["HBHUR"].map(HBHUR_TO_AREA)
    df = df[df["R_AGE"] >= 18].copy()
    df = df[(df["VEHAGE"] >= 0) & (df["VEHAGE"] <= MAX_VEHICLE_AGE)]
    df["age_group"] = df["R_AGE"].map(age_group)
    df = df.dropna(subset=["vehicle_class", "income_band", "area_type"])
    log.info(f"Usable rows (4 car classes, adult driver, known income, "
             f"vehicle <= {MAX_VEHICLE_AGE} yrs old): {len(df):,}")
    return df


def make_features(df: pd.DataFrame) -> np.ndarray:
    """Exactly the 6 features api/predictor.py sends at serving time."""
    age = df["age_group"].map(AGE_MAP).to_numpy()
    inc = df["income_band"].map(INCOME_MAP).to_numpy()
    hh = df["HHSIZE"].clip(upper=4).to_numpy()
    area = df["area_type"].map(AREA_MAP).to_numpy()
    return np.column_stack([age, inc, hh, area, age * inc, hh * area]).astype(float)


def evaluate(name, proba, y, w, n_classes):
    ll = log_loss(y, proba, sample_weight=w, labels=list(range(n_classes)))
    f1 = f1_score(y, proba.argmax(1), average="macro", sample_weight=w)
    log.info(f"  {name:<28} weighted log loss={ll:.4f}  macro F1={f1:.3f}")
    return ll, f1


def run():
    sklearn.set_config(enable_metadata_routing=True)
    df = load_nhts()
    le = LabelEncoder()
    y = le.fit_transform(df["vehicle_class"])
    X = make_features(df)
    w = df["WTHHFIN"].to_numpy()
    groups = df["HOUSEID"].to_numpy()
    k = len(le.classes_)

    dist = (df.groupby("vehicle_class")["WTHHFIN"].sum() / w.sum()).round(3)
    log.info(f"Weighted class mix: {dist.to_dict()}")

    # Household-grouped 80/20 split
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr, te = next(sgkf.split(X, y, groups))
    assert not set(groups[tr]) & set(groups[te]), "household leaked across split"
    log.info(f"Train {len(tr):,} / test {len(te):,} vehicles (no shared households)")

    log.info("\nHeld-out results:")
    # Baseline: always predict the weighted class mix
    prior = np.bincount(y[tr], weights=w[tr], minlength=k) / w[tr].sum()
    results = {"prior_baseline": evaluate("Prior (no features)",
                                          np.tile(prior, (len(te), 1)), y[te], w[te], k)}

    inner = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
                 .split(X[tr], y[tr], groups[tr]))
    candidates = {
        # One-hot the 4 base features so each age group / income band / area
        # gets its own effect (ordinal codes force a straight-line trend).
        # The API still sends the same 6 numbers; the pipeline picks columns 0-3.
        "logistic_regression_onehot": make_pipeline(
            ColumnTransformer([("onehot", OneHotEncoder(handle_unknown="ignore"),
                                [0, 1, 2, 3])]),
                        LogisticRegression(max_iter=3000).set_fit_request(sample_weight=True)),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED,
        ).set_fit_request(sample_weight=True),
    }
    for name, est in candidates.items():
        # Calibrate with household-grouped inner folds (no leakage there either)
        model = CalibratedClassifierCV(est, cv=inner, method="sigmoid")
        model.fit(X[tr], y[tr], sample_weight=w[tr])
        results[name] = evaluate(name, model.predict_proba(X[te]), y[te], w[te], k)

    best = min(candidates, key=lambda n: results[n][0])
    gain = (1 - results[best][0] / results["prior_baseline"][0]) * 100
    log.info(f"\nBest: {best}  ({gain:.1f}% lower log loss than the no-feature baseline)")

    # Refit the winner on ALL data, calibrated with household-grouped folds
    all_folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
                     .split(X, y, groups))
    final = CalibratedClassifierCV(candidates[best], cv=all_folds, method="sigmoid")
    final.fit(X, y, sample_weight=w)

    MODELS.mkdir(exist_ok=True)
    joblib.dump(final, MODELS / "lifestage_classifier.pkl")
    joblib.dump(le, MODELS / "lifestage_label_encoder.pkl")
    meta = {
        "data_source": "NHTS 2022 (FHWA), vehv2pub + perv2pub, survey-weighted, "
                       f"vehicles <= {MAX_VEHICLE_AGE} yrs old",
        "age_map": AGE_MAP, "income_map": INCOME_MAP, "area_map": AREA_MAP,
        "classes": list(le.classes_),
        "model": best,
        "heldout_weighted_log_loss": round(results[best][0], 4),
        "heldout_macro_f1": round(results[best][1], 3),
        "prior_baseline_log_loss": round(results["prior_baseline"][0], 4),
        "income_band_note": "low < $35k, middle $35k-$100k, high > $100k (NHTS bins)",
    }
    with open(MODELS / "lifestage_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Saved models/lifestage_classifier.pkl, label encoder, lifestage_meta.json")

    # Persona check (same 6-feature vector the API builds)
    log.info("\nPersonas:")
    personas = [
        ("21, student, low income, alone, urban",      "18-34", "low",    1, "urban"),
        ("38, family of 4, middle income, suburban",    "35-54", "middle", 4, "suburban"),
        ("68, retiree couple, middle income, suburban", "65+",   "middle", 2, "suburban"),
        ("45, couple, high income, rural",              "35-54", "high",   2, "rural"),
    ]
    for desc, a, i, h, ar in personas:
        av, iv, rv = AGE_MAP[a], INCOME_MAP[i], AREA_MAP[ar]
        p = final.predict_proba([[av, iv, h, rv, av * iv, h * rv]])[0]
        top = " ".join(f"{c}={v:.2f}" for c, v in
                       sorted(zip(le.classes_, p), key=lambda t: -t[1]))
        log.info(f"  {desc:<46} {top}")


if __name__ == "__main__":
    run()