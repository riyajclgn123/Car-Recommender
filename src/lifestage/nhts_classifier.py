"""
src/lifestage/nhts_classifier.py

Life-stage classifier: predicts likely vehicle class from
age group, income band, household size, and urban/rural.

Based on NHTS 2017 survey statistics. Outputs calibrated
probabilities that feed the recommender's scoring weights.

Run:
    python -m src.lifestage.nhts_classifier
"""

import numpy as np
import pandas as pd
import joblib
import logging
import mlflow
import matplotlib.pyplot as plt
import json
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, f1_score,
                              log_loss, confusion_matrix, ConfusionMatrixDisplay)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

MODELS = Path("models")
MODELS.mkdir(exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Vehicle classes ───────────────────────────────────────────────────────────
VEHICLE_CLASSES = ["sedan", "suv_crossover", "pickup", "minivan_van", "other"]


# ── Synthetic NHTS-calibrated dataset ────────────────────────────────────────
def generate_nhts_synthetic(n_samples: int = 50_000) -> pd.DataFrame:
    """
    Generate synthetic data calibrated to NHTS 2017 statistics.

    Key NHTS findings used:
    - SUV/crossover is most popular across all groups (~35%)
    - Young singles (18-34, 1 person) favor sedans
    - Families (35-54, 3-4 people) favor SUVs and minivans
    - Rural households favor pickups
    - High income skews toward SUVs and away from sedans
    - Retirees (65+) favor sedans and smaller SUVs
    """
    rng = np.random.default_rng(RANDOM_SEED)

    age_groups    = ["18-34", "35-54", "55-64", "65+"]
    income_bands  = ["low", "middle", "high"]
    hh_sizes      = [1, 2, 3, 4]
    area_types    = ["urban", "suburban", "rural"]

    # Probabilities calibrated to NHTS 2017
    # Format: P(vehicle_class | profile)
    # Classes: sedan, suv_crossover, pickup, minivan_van, other
    profiles = {
        # Young, urban, low income → mostly sedan
        ("18-34", "low",    1, "urban"):    [0.50, 0.25, 0.08, 0.02, 0.15],
        ("18-34", "low",    2, "urban"):    [0.40, 0.30, 0.10, 0.05, 0.15],
        ("18-34", "middle", 1, "urban"):    [0.40, 0.35, 0.10, 0.02, 0.13],
        ("18-34", "middle", 2, "suburban"): [0.30, 0.42, 0.12, 0.08, 0.08],
        ("18-34", "high",   1, "urban"):    [0.30, 0.45, 0.10, 0.02, 0.13],
        ("18-34", "low",    1, "rural"):    [0.30, 0.20, 0.35, 0.02, 0.13],

        # Families → SUV and minivan
        ("35-54", "low",    3, "suburban"): [0.25, 0.38, 0.15, 0.15, 0.07],
        ("35-54", "low",    4, "suburban"): [0.15, 0.38, 0.15, 0.25, 0.07],
        ("35-54", "middle", 3, "suburban"): [0.20, 0.45, 0.15, 0.15, 0.05],
        ("35-54", "middle", 4, "suburban"): [0.10, 0.42, 0.18, 0.25, 0.05],
        ("35-54", "high",   3, "suburban"): [0.15, 0.55, 0.15, 0.10, 0.05],
        ("35-54", "high",   4, "suburban"): [0.08, 0.52, 0.18, 0.18, 0.04],
        ("35-54", "middle", 2, "rural"):    [0.15, 0.30, 0.42, 0.08, 0.05],
        ("35-54", "high",   2, "rural"):    [0.10, 0.35, 0.45, 0.05, 0.05],

        # Retirees → sedan and smaller SUVs
        ("55-64", "low",    1, "urban"):    [0.50, 0.28, 0.10, 0.05, 0.07],
        ("55-64", "low",    2, "suburban"): [0.40, 0.35, 0.12, 0.08, 0.05],
        ("55-64", "middle", 2, "suburban"): [0.35, 0.42, 0.12, 0.06, 0.05],
        ("55-64", "high",   2, "suburban"): [0.28, 0.50, 0.12, 0.05, 0.05],
        ("55-64", "middle", 2, "rural"):    [0.25, 0.28, 0.40, 0.05, 0.02],

        # Older retirees → sedan dominant
        ("65+",   "low",    1, "urban"):    [0.58, 0.25, 0.08, 0.05, 0.04],
        ("65+",   "low",    2, "suburban"): [0.50, 0.32, 0.08, 0.06, 0.04],
        ("65+",   "middle", 2, "suburban"): [0.45, 0.38, 0.08, 0.05, 0.04],
        ("65+",   "high",   2, "suburban"): [0.38, 0.48, 0.08, 0.03, 0.03],
        ("65+",   "low",    2, "rural"):    [0.35, 0.22, 0.38, 0.03, 0.02],
    }

    # Default profile if key not found
    default_probs = [0.25, 0.38, 0.18, 0.12, 0.07]

    rows = []
    for _ in range(n_samples):
        age    = rng.choice(age_groups,   p=[0.22, 0.32, 0.22, 0.24])
        income = rng.choice(income_bands, p=[0.30, 0.42, 0.28])
        hh     = int(rng.choice(hh_sizes, p=[0.28, 0.35, 0.20, 0.17]))
        area   = rng.choice(area_types,   p=[0.30, 0.52, 0.18])

        # Find closest profile
        key = (age, income, min(hh, 4), area)
        probs = profiles.get(key, default_probs)

        # Add noise to prevent perfect memorization
        probs = np.array(probs) + rng.normal(0, 0.005, len(probs))
        probs = np.clip(probs, 0.01, 1.0)
        probs /= probs.sum()

        vehicle_class = rng.choice(VEHICLE_CLASSES, p=probs)

        rows.append({
            "age_group":      age,
            "income_band":    income,
            "household_size": hh,
            "area_type":      area,
            "vehicle_class":  vehicle_class,
        })

    return pd.DataFrame(rows)


# ── Feature engineering ───────────────────────────────────────────────────────
def encode_features(df: pd.DataFrame) -> tuple:
    """Encode categorical features as integers."""
    age_map    = {"18-34": 0, "35-54": 1, "55-64": 2, "65+": 3}
    income_map = {"low": 0, "middle": 1, "high": 2}
    area_map   = {"urban": 0, "suburban": 1, "rural": 2}

    X = pd.DataFrame({
        "age_group":      df["age_group"].map(age_map),
        "income_band":    df["income_band"].map(income_map),
        "household_size": df["household_size"],
        "area_type":      df["area_type"].map(area_map),
        # Interaction features
        "age_x_income":   df["age_group"].map(age_map) *
                          df["income_band"].map(income_map),
        "hh_x_area":      df["household_size"] *
                          df["area_type"].map(area_map),
    })

    le = LabelEncoder()
    y  = le.fit_transform(df["vehicle_class"])

    return X.values, y, le


# ── Calibration plot ──────────────────────────────────────────────────────────
def plot_calibration(model, X_val, y_val, class_names, le):
    """Plot calibration curves for each vehicle class."""
    y_prob = model.predict_proba(X_val)

    fig, axes = plt.subplots(1, len(class_names),
                              figsize=(4 * len(class_names), 4))
    for i, cls in enumerate(class_names):
        cls_idx = list(le.classes_).index(cls)
        y_bin   = (y_val == cls_idx).astype(int)

        if y_bin.sum() < 10:
            axes[i].set_title(f"{cls}\n(too few samples)")
            continue

        try:
            prob_true, prob_pred = calibration_curve(
                y_bin, y_prob[:, cls_idx], n_bins=8)
            axes[i].plot(prob_pred, prob_true, "s-",
                         color="#4C72B0", label="Model")
            axes[i].plot([0,1],[0,1], "--", color="gray", label="Perfect")
            axes[i].set_title(f"{cls}", fontweight="bold")
            axes[i].set_xlabel("Mean predicted prob")
            axes[i].set_ylabel("Fraction positive")
            axes[i].legend(fontsize=7)
            axes[i].grid(alpha=0.3)
        except Exception:
            axes[i].set_title(f"{cls}\n(calibration error)")

    plt.suptitle("Calibration curves by vehicle class",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("docs/plot_lifestage_calibration.png", bbox_inches="tight")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    log.info("=" * 55)
    log.info("DAY 10 — LIFE-STAGE CLASSIFIER")
    log.info("=" * 55)

    # Generate data
    log.info("Generating NHTS-calibrated synthetic dataset …")
    df = generate_nhts_synthetic(n_samples=50_000)
    log.info(f"Dataset: {df.shape}")
    log.info(f"\nClass distribution:")
    dist = df["vehicle_class"].value_counts(normalize=True)
    for cls, pct in dist.items():
        log.info(f"  {cls:<20} {pct:.1%}")

    # Encode
    X, y, le = encode_features(df)
    log.info(f"\nFeatures: {X.shape[1]}  Classes: {len(le.classes_)}")

    # Train/val split (stratified)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

    # ── Three models ──────────────────────────────────────────────────────────
    log.info("\n" + "="*55)
    log.info("TRAINING THREE CLASSIFIERS")
    log.info("="*55)

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED,
            C=1.0, class_weight="balanced"),
        "random_forest":       RandomForestClassifier(
            n_estimators=200, max_depth=8,
            random_state=RANDOM_SEED, n_jobs=-1,
            class_weight="balanced"),
        "gradient_boosting":   GradientBoostingClassifier(
            n_estimators=200, max_depth=4,
            learning_rate=0.05, random_state=RANDOM_SEED),
    }

    best_f1    = 0
    best_name  = None
    best_model = None

    cv = StratifiedKFold(n_splits=5, shuffle=True,
                         random_state=RANDOM_SEED)

    with mlflow.start_run(run_name="lifestage_classifier_comparison"):
        for name, clf in models.items():
            clf.fit(X_train, y_train)
            val_pred  = clf.predict(X_val)
            val_proba = clf.predict_proba(X_val)

            macro_f1 = f1_score(y_val, val_pred, average="macro")
            logloss  = log_loss(y_val, val_proba)
            cv_f1    = cross_val_score(
                clf, X_train, y_train,
                cv=cv, scoring="f1_macro", n_jobs=-1).mean()

            log.info(f"\n  {name}:")
            log.info(f"    Macro F1:    {macro_f1:.4f}")
            log.info(f"    Log loss:    {logloss:.4f}")
            log.info(f"    CV F1 mean:  {cv_f1:.4f}")

            mlflow.log_metrics({
                f"{name}_macro_f1": macro_f1,
                f"{name}_log_loss": logloss,
                f"{name}_cv_f1":    cv_f1,
            })

            if macro_f1 > best_f1:
                best_f1    = macro_f1
                best_name  = name
                best_model = clf

    log.info(f"\nBest model: {best_name}  Macro F1={best_f1:.4f}")

    # ── Calibrate the best model ──────────────────────────────────────────────
    log.info("\nCalibrating best model …")
    calibrated = CalibratedClassifierCV(best_model,
                                         cv=5, method="sigmoid")
    calibrated.fit(X_train, y_train)

    # ── Final evaluation ──────────────────────────────────────────────────────
    val_pred  = calibrated.predict(X_val)
    val_proba = calibrated.predict_proba(X_val)

    macro_f1 = f1_score(y_val, val_pred, average="macro")
    logloss  = log_loss(y_val, val_proba)

    log.info(f"\nCalibrated model:")
    log.info(f"  Macro F1: {macro_f1:.4f}")
    log.info(f"  Log loss: {logloss:.4f}")
    log.info(f"\nClassification report:")
    print(classification_report(y_val, val_pred,
                                target_names=le.classes_,
                                zero_division=0))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_val, val_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Life-stage classifier — confusion matrix",
                 fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("docs/plot_lifestage_confusion.png", bbox_inches="tight")
    plt.show()

    # Calibration plot
    plot_calibration(calibrated, X_val, y_val, VEHICLE_CLASSES, le)

    # ── Save model + metadata ─────────────────────────────────────────────────
    joblib.dump(calibrated, MODELS / "lifestage_classifier.pkl")
    joblib.dump(le,         MODELS / "lifestage_label_encoder.pkl")

    # Save encoding maps for use in recommender
    meta = {
        "age_map":    {"18-34": 0, "35-54": 1, "55-64": 2, "65+": 3},
        "income_map": {"low": 0, "middle": 1, "high": 2},
        "area_map":   {"urban": 0, "suburban": 1, "rural": 2},
        "classes":    list(le.classes_),
        "macro_f1":   macro_f1,
        "log_loss":   logloss,
    }
    with open(MODELS / "lifestage_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"\nSaved → models/lifestage_classifier.pkl")
    log.info(f"Saved → models/lifestage_label_encoder.pkl")
    log.info(f"Saved → models/lifestage_meta.json")

    # ── Quick prediction test ─────────────────────────────────────────────────
    log.info("\n" + "="*55)
    log.info("PREDICTION TEST — 3 PERSONAS")
    log.info("="*55)

    personas = [
        ("21-yr student, low income, 1 person, urban",
         [0, 0, 1, 0, 0*0, 1*0]),
        ("38-yr family, middle income, 4 people, suburban",
         [1, 1, 4, 1, 1*1, 4*1]),
        ("68-yr retiree, middle income, 2 people, suburban",
         [3, 1, 2, 1, 3*1, 2*1]),
    ]

    for desc, features in personas:
        proba = calibrated.predict_proba([features])[0]
        pred  = le.classes_[np.argmax(proba)]
        log.info(f"\n  {desc}")
        log.info(f"  Predicted: {pred}")
        for cls, p in zip(le.classes_, proba):
            bar = "█" * int(p * 30)
            log.info(f"    {cls:<20} {p:.2f}  {bar}")

    log.info("\nCalibrated probabilities ✅")
    log.info("CLASSIFIER COMPLETE")


if __name__ == "__main__":
    run()