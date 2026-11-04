"""
src/models/train_gbm.py

Day 6: Random Forest → LightGBM → CatBoost → Optuna tuning.
Every run logged to MLflow.

Run:
    python -m src.models.train_gbm
"""

import numpy as np
import polars as pl
import lightgbm as lgb
import catboost as cb
import optuna
import mlflow
import mlflow.lightgbm
import matplotlib.pyplot as plt
import logging
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROC = Path("data/processed")
MODELS = Path("models")
MODELS.mkdir(exist_ok=True)


# ── Metrics (always in dollars) ───────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    true_d = np.exp(y_true)
    pred_d = np.exp(y_pred)
    mae  = np.mean(np.abs(true_d - pred_d))
    mape = np.mean(np.abs((true_d - pred_d) / true_d)) * 100
    rmse = np.sqrt(np.mean((true_d - pred_d) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return {"mae": mae, "mape": mape, "rmse": rmse, "r2": r2}


def log_metrics(metrics: dict, label: str = "") -> None:
    if label:
        log.info(f"\n  {label}:")
    log.info(f"    MAE:  ${metrics['mae']:>10,.0f}")
    log.info(f"    MAPE: {metrics['mape']:>9.2f}%")
    log.info(f"    RMSE: ${metrics['rmse']:>10,.0f}")
    log.info(f"    R²:   {metrics['r2']:>10.4f}")


# ── Load data ─────────────────────────────────────────────────────────────────
def load_data():
    X_train = pl.read_parquet(PROC / "X_train.parquet").to_numpy()
    y_train = pl.read_parquet(PROC / "y_train.parquet")["log_price"].to_numpy()
    X_val   = pl.read_parquet(PROC / "X_val.parquet").to_numpy()
    y_val   = pl.read_parquet(PROC / "y_val.parquet")["log_price"].to_numpy()
    X_test  = pl.read_parquet(PROC / "X_test.parquet").to_numpy()
    y_test  = pl.read_parquet(PROC / "y_test.parquet")["log_price"].to_numpy()
    feat_names = pl.read_parquet(PROC / "X_train.parquet").columns
    log.info(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test, feat_names


# ── 1. Random Forest ──────────────────────────────────────────────────────────
def run_random_forest(X_train, y_train, X_val, y_val, feat_names):
    log.info("\n" + "="*55)
    log.info("1. RANDOM FOREST")
    log.info("="*55)

    # Overfitting curve: train error vs val error as n_estimators grows
    n_trees_list = [10, 25, 50, 100, 200]
    train_maes, val_maes = [], []

    for n in n_trees_list:
        rf = RandomForestRegressor(
            n_estimators=n,
            max_depth=15,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        )
        rf.fit(X_train, y_train)
        train_mae = compute_metrics(y_train, rf.predict(X_train))["mae"]
        val_mae   = compute_metrics(y_val,   rf.predict(X_val))["mae"]
        train_maes.append(train_mae)
        val_maes.append(val_mae)
        log.info(f"  n={n:3d}  train MAE=${train_mae:,.0f}  val MAE=${val_mae:,.0f}")

    # Plot overfitting curve
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(n_trees_list, train_maes, "o-", label="Train MAE", color="#4C72B0")
    ax.plot(n_trees_list, val_maes,   "o-", label="Val MAE",   color="#C44E52")
    ax.axhline(3518, color="gray", linestyle="--", label="Baseline $3,518")
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("MAE ($)")
    ax.set_title("Random Forest — overfitting curve", fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x/1000:.0f}k"))
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("docs/plot_rf_overfitting.png", bbox_inches="tight")
    plt.show()

    # Final RF with 200 trees
    rf_final = RandomForestRegressor(
        n_estimators=200, max_depth=15,
        min_samples_leaf=5, n_jobs=-1, random_state=42
    )
    rf_final.fit(X_train, y_train)
    metrics = compute_metrics(y_val, rf_final.predict(X_val))

    with mlflow.start_run(run_name="random_forest_200"):
        mlflow.set_tag("model_type", "random_forest")
        mlflow.log_params({"n_estimators": 200, "max_depth": 15})
        for k, v in metrics.items():
            mlflow.log_metric(f"val_{k}", v)

    log_metrics(metrics, "RANDOM FOREST (200 trees, val)")

    # Feature importance
    imp = rf_final.feature_importances_
    idx = np.argsort(imp)[-15:]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([feat_names[i] for i in idx], imp[idx], color="#4C72B0")
    ax.set_title("Random Forest — top 15 feature importances", fontweight="bold")
    plt.tight_layout()
    plt.savefig("docs/plot_rf_importance.png", bbox_inches="tight")
    plt.show()

    return rf_final, metrics


# ── 2. LightGBM ───────────────────────────────────────────────────────────────
def run_lightgbm(X_train, y_train, X_val, y_val, feat_names):
    log.info("\n" + "="*55)
    log.info("2. LIGHTGBM (default params)")
    log.info("="*55)

    params = {
        "objective":   "regression",
        "metric":      "mae",
        "learning_rate": 0.05,
        "num_leaves":  127,
        "min_child_samples": 20,
        "verbosity":   -1,
        "random_state": 42,
    }

    dtrain = lgb.Dataset(X_train, label=y_train,
                         feature_name=feat_names)
    dval   = lgb.Dataset(X_val,   label=y_val,
                         feature_name=feat_names, reference=dtrain)

    callbacks = [
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(100),
    ]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=callbacks,
    )

    val_pred = model.predict(X_val)
    metrics  = compute_metrics(y_val, val_pred)
    log_metrics(metrics, "LIGHTGBM (default, val)")

    with mlflow.start_run(run_name="lightgbm_default"):
        mlflow.set_tag("model_type", "lightgbm")
        mlflow.log_params(params)
        mlflow.log_param("best_iteration", model.best_iteration)
        for k, v in metrics.items():
            mlflow.log_metric(f"val_{k}", v)

    # Feature importance
    imp = model.feature_importance(importance_type="gain")
    idx = np.argsort(imp)[-15:]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([feat_names[i] for i in idx], imp[idx], color="#55A868")
    ax.set_title("LightGBM — top 15 features (gain)", fontweight="bold")
    plt.tight_layout()
    plt.savefig("docs/plot_lgbm_importance.png", bbox_inches="tight")
    plt.show()

    return model, metrics


# ── 3. Optuna tuning on LightGBM ─────────────────────────────────────────────
def run_optuna(X_train, y_train, X_val, y_val):
    log.info("\n" + "="*55)
    log.info("3. OPTUNA — LightGBM hyperparameter search (30 trials)")
    log.info("="*55)

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval   = lgb.Dataset(X_val,   label=y_val, reference=dtrain)

    def objective(trial):
        params = {
            "objective":     "regression",
            "metric":        "mae",
            "verbosity":     -1,
            "random_state":  42,
            "feature_pre_filter": False,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves":    trial.suggest_int("num_leaves", 31, 511),
            "max_depth":     trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq":      1,
            "reg_alpha":     trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":    trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        model = lgb.train(
            params, dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(30, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        val_pred = model.predict(X_val)
        # Optuna minimizes — return MAE in log space (fast proxy)
        return np.mean(np.abs(y_val - val_pred))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    log.info(f"\n  Best trial MAE (log space): {study.best_value:.6f}")
    log.info(f"  Best params: {study.best_params}")

    # Retrain with best params
    best_params = {
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,
        "random_state": 42,
        **study.best_params,
        "bagging_freq": 1,
    }

    final_model = lgb.train(
        best_params, dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(100)],
    )

    val_pred = final_model.predict(X_val)
    metrics  = compute_metrics(y_val, val_pred)
    log_metrics(metrics, "LIGHTGBM TUNED (val)")

    with mlflow.start_run(run_name="lightgbm_optuna_tuned"):
        mlflow.set_tag("model_type", "lightgbm_tuned")
        mlflow.log_params(best_params)
        mlflow.log_param("best_iteration", final_model.best_iteration)
        mlflow.log_param("n_optuna_trials", 30)
        for k, v in metrics.items():
            mlflow.log_metric(f"val_{k}", v)

    # Save the best model
    joblib.dump(final_model, MODELS / "lgbm_tuned.pkl")
    log.info(f"  Saved → models/lgbm_tuned.pkl")

    return final_model, metrics, study.best_params


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    log.info("=" * 55)
    log.info("DAY 6 — ENSEMBLES + BOOSTING")
    log.info("=" * 55)

    X_train, y_train, X_val, y_val, X_test, y_test, feat_names = load_data()

    rf_model,  rf_metrics  = run_random_forest(X_train, y_train,
                                                X_val, y_val, feat_names)
    lgb_model, lgb_metrics = run_lightgbm(X_train, y_train,
                                           X_val, y_val, feat_names)
    tuned_model, tuned_metrics, best_params = run_optuna(X_train, y_train,
                                                          X_val, y_val)

    # Final evaluation on TEST set (only for the best model)
    test_pred    = tuned_model.predict(X_test)
    test_metrics = compute_metrics(y_test, test_pred)
    log_metrics(test_metrics, "LIGHTGBM TUNED (TEST — final)")

    log.info("\n" + "="*55)
    log.info("DAY 6 SUMMARY")
    log.info("="*55)
    log.info(f"  Baseline:          MAE=${3518:>8,.0f}")
    log.info(f"  Random Forest:     MAE=${rf_metrics['mae']:>8,.0f}  "
             f"R²={rf_metrics['r2']:.4f}")
    log.info(f"  LightGBM default:  MAE=${lgb_metrics['mae']:>8,.0f}  "
             f"R²={lgb_metrics['r2']:.4f}")
    log.info(f"  LightGBM tuned:    MAE=${tuned_metrics['mae']:>8,.0f}  "
             f"R²={tuned_metrics['r2']:.4f}")
    log.info(f"  LightGBM test:     MAE=${test_metrics['mae']:>8,.0f}  "
             f"R²={test_metrics['r2']:.4f}")


if __name__ == "__main__":
    run()