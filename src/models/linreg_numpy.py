"""
src/models/linreg_numpy.py

Linear regression with gradient descent — written from scratch in NumPy.
No sklearn for the core math. The point is to watch learning happen.

Run:
    python -m src.models.linreg_numpy
"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import mlflow
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

PROC = Path("data/processed")


class LinearRegressionGD:
    """
    Linear regression trained with batch gradient descent.
    Target is log_price; metrics reported in dollars.
    """

    def __init__(self, learning_rate: float = 0.01, n_epochs: int = 100):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.w = None      # weights — shape (n_features,)
        self.b = None      # bias — scalar
        self.loss_history = []   # loss per epoch
        self.weight_history = [] # snapshot of weights every 10 epochs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        n, p = X.shape
        # Initialize weights to zero
        self.w = np.zeros(p)
        self.b = 0.0

        for epoch in range(self.n_epochs):
            # 1: compute predictions (ŷ = X @ w + b)
            y_pred = X @ self.w + self.b

            # 2: compute MSE loss  L = mean((ŷ - y)²)
            loss = np.mean((y_pred - y) ** 2)

            # 3: compute gradients
            error = y_pred - y
            dw = (2/n) * X.T @ error
            db = (2/n) * np.sum(error)

            # Update weights
            self.w -= self.lr * dw
            self.b -= self.lr * db

            self.loss_history.append(loss)
            if epoch % 10 == 0:
                self.weight_history.append(self.w.copy())
                log.info(f"  Epoch {epoch:3d}  loss={loss:.6f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Convert log_price predictions back to dollars for reporting."""
    true_dollars = np.exp(y_true)
    pred_dollars = np.exp(y_pred)
    mae  = np.mean(np.abs(true_dollars - pred_dollars))
    mape = np.mean(np.abs((true_dollars - pred_dollars) / true_dollars)) * 100
    rmse = np.sqrt(np.mean((true_dollars - pred_dollars) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return {"mae": mae, "mape": mape, "rmse": rmse, "r2": r2}


def plot_loss_curve(loss_history: list, lr: float, run_name: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(loss_history, color="#4C72B0", linewidth=2)
    ax.set_title(f"Loss curve — lr={lr}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (log_price)")
    ax.grid(alpha=0.3)
    path = f"docs/plot_loss_{run_name}.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    log.info(f"  Saved → {path}")


def plot_top_weights(w: np.ndarray, feature_names: list,
                     run_name: str, top_n: int = 15) -> None:
    """Plot the largest weights by absolute value."""
    idx = np.argsort(np.abs(w))[-top_n:]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#C44E52" if w[i] < 0 else "#4C72B0" for i in idx]
    ax.barh([feature_names[i] for i in idx], w[idx], color=colors)
    ax.set_title(f"Top {top_n} weights by magnitude — lr={run_name}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Weight value")
    ax.axvline(0, color="black", linewidth=0.8)
    path = f"docs/plot_weights_{run_name}.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    log.info(f"  Saved → {path}")


def run():
    log.info("=" * 55)
    log.info("LINEAR REGRESSION — GRADIENT DESCENT FROM SCRATCH")
    log.info("=" * 55)

    # Load features (already scaled in features.py)
    X_train = pl.read_parquet(PROC / "X_train.parquet").to_numpy()
    y_train = pl.read_parquet(PROC / "y_train.parquet")["log_price"].to_numpy()
    X_val   = pl.read_parquet(PROC / "X_val.parquet").to_numpy()
    y_val   = pl.read_parquet(PROC / "y_val.parquet")["log_price"].to_numpy()

    feature_names = pl.read_parquet(PROC / "X_train.parquet").columns

    log.info(f"X_train: {X_train.shape}  y_train: {y_train.shape}")

    # ── Experiment 1: learning rate too SMALL ─────────────────────────────────
    # ── Experiment 2: learning rate too LARGE ─────────────────────────────────
    # ── Experiment 3: learning rate just right ────────────────────────────────

    experiments = [
        ("too_small",  0.0001, 100),
        ("too_large",  0.5,    100),
        ("just_right", 0.05,   500),
    ]

    best_mae = float("inf")
    best_run = None

    for run_name, lr, epochs in experiments:
        log.info(f"\n{'='*40}")
        log.info(f"Experiment: {run_name}  lr={lr}  epochs={epochs}")
        log.info(f"{'='*40}")

        with mlflow.start_run(run_name=f"linreg_{run_name}"):
            mlflow.set_tag("model_type", "linear_regression_numpy")
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("n_epochs", epochs)

            model = LinearRegressionGD(learning_rate=lr, n_epochs=epochs)
            model.fit(X_train, y_train)

            # Evaluate
            val_pred = model.predict(X_val)
            metrics = compute_metrics(y_val, val_pred)

            log.info(f"\n  VAL METRICS ({run_name}):")
            log.info(f"    MAE:  ${metrics['mae']:>10,.0f}")
            log.info(f"    MAPE: {metrics['mape']:>9.2f}%")
            log.info(f"    R²:   {metrics['r2']:>10.4f}")

            for k, v in metrics.items():
                mlflow.log_metric(f"val_{k}", v)

            # Plots
            plot_loss_curve(model.loss_history, lr, run_name)
            plot_top_weights(model.w, feature_names, run_name)

            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                best_run = run_name

        # ── Ridge and Lasso (sklearn analytical solution) ─────────────────────────
    from sklearn.linear_model import Ridge, Lasso

    log.info("\n" + "="*55)
    log.info("RIDGE AND LASSO (sklearn)")
    log.info("="*55)

    for name, model_cls, alpha in [
        ("ridge", Ridge, 1.0),
        ("lasso", Lasso, 0.001),
    ]:
        with mlflow.start_run(run_name=f"linreg_{name}"):
            mlflow.set_tag("model_type", name)
            mlflow.log_param("alpha", alpha)

            m = model_cls(alpha=alpha)
            m.fit(X_train, y_train)

            val_pred = m.predict(X_val)
            metrics = compute_metrics(y_val, val_pred)

            log.info(f"\n  {name.upper()} (alpha={alpha}):")
            log.info(f"    MAE:  ${metrics['mae']:>10,.0f}")
            log.info(f"    MAPE: {metrics['mape']:>9.2f}%")
            log.info(f"    R²:   {metrics['r2']:>10.4f}")

            for k, v in metrics.items():
                mlflow.log_metric(f"val_{k}", v)

            feat_names = pl.read_parquet(PROC / "X_train.parquet").columns
            plot_top_weights(m.coef_, feat_names, name)

    log.info(f"\nBest run: {best_run}  MAE=${best_mae:,.0f}")
    log.info("All runs logged to MLflow — check http://localhost:5000")

if __name__ == "__main__":
    run()