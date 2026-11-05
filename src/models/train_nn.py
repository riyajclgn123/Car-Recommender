"""
src/models/train_nn.py

Day 8: PyTorch MLP with embeddings for categorical features.
Logs gradient norms and weight histograms to TensorBoard.

Run:
    python -m src.models.train_nn
"""

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import mlflow
import json
import logging
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

PROC   = Path("data/processed")
MODELS = Path("models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {DEVICE}")

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ── Dataset ───────────────────────────────────────────────────────────────────
class CarDataset(Dataset):
    """
    Wraps X (features) and y (log_price) as PyTorch tensors.
    Splits features into:
      - categorical columns (will go through embeddings)
      - continuous columns (already scaled)
    """
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 cat_indices: list, cont_indices: list):
        self.X_cat  = torch.tensor(X[:, cat_indices], dtype=torch.long)
        self.X_cont = torch.tensor(X[:, cont_indices], dtype=torch.float32)
        self.y      = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_cat[idx], self.X_cont[idx], self.y[idx]


# ── Model ─────────────────────────────────────────────────────────────────────
class CarPriceMLP(nn.Module):
    """
    MLP with:
    - One embedding per categorical feature
    - Batch norm + dropout for regularization
    - 3 hidden layers
    """

    def __init__(self, cat_vocab_sizes: list, emb_dim: int,
                 n_cont: int, hidden_dims: list, dropout: float):
        super().__init__()

        # Embedding for each categorical column
        # emb_dim is the size of the learned vector per category
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size + 1, emb_dim)   # +1 for unseen categories
            for vocab_size in cat_vocab_sizes
        ])

        # Total input size = embeddings + continuous features
        emb_total = len(cat_vocab_sizes) * emb_dim
        input_dim = emb_total + n_cont

        # Build hidden layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = hidden_dim

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 1)

    def forward(self, x_cat, x_cont):
        # Embed each categorical column and concatenate
        embedded = [emb(x_cat[:, i])
                    for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embedded + [x_cont], dim=1)
        x = self.hidden(x)
        return self.output(x).squeeze(1)


# ── Metrics ───────────────────────────────────────────────────────────────────
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


# ── Training loop ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x_cat, x_cont, y in loader:
        x_cat  = x_cat.to(device)
        x_cont = x_cont.to(device)
        y      = y.to(device)

        optimizer.zero_grad()
        pred = model(x_cat, x_cont)
        loss = criterion(pred, y)
        loss.backward()

        # Gradient clipping — prevents explosion
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item() * len(y)

    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds, targets = [], []
    with torch.no_grad():
        for x_cat, x_cont, y in loader:
            x_cat  = x_cat.to(device)
            x_cont = x_cont.to(device)
            y      = y.to(device)
            pred   = model(x_cat, x_cont)
            loss   = criterion(pred, y)
            total_loss += loss.item() * len(y)
            preds.extend(pred.cpu().numpy())
            targets.extend(y.cpu().numpy())
    return (total_loss / len(loader.dataset),
            np.array(preds), np.array(targets))


# ── Log gradients + weights to TensorBoard ───────────────────────────────────
def log_gradients(model, writer, step):
    total_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.norm().item()
            total_norm += norm ** 2
            writer.add_scalar(f"grad_norm/{name}", norm, step)
            writer.add_histogram(f"weights/{name}", param.data, step)
    writer.add_scalar("grad_norm/total",
                      total_norm ** 0.5, step)


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    log.info("=" * 55)
    log.info("DAY 8 — PYTORCH MLP WITH EMBEDDINGS")
    log.info("=" * 55)

    # Load feature metadata to identify categorical columns
    with open(PROC / "feature_meta.json") as f:
        meta = json.load(f)

    feat_names   = meta["features"]
    cat_cols     = meta["categorical_cols"]
    encoders     = meta["encoders"]

    # Identify indices for categorical vs continuous columns
    cat_indices  = [feat_names.index(c) for c in cat_cols
                    if c in feat_names]
    cont_indices = [i for i in range(len(feat_names))
                    if i not in cat_indices]

    log.info(f"Categorical features: {len(cat_indices)}")
    log.info(f"Continuous features:  {len(cont_indices)}")

    # Vocabulary sizes (number of unique categories per column)
    cat_vocab_sizes = [len(encoders[c]) for c in cat_cols
                       if c in feat_names]

    # Load data
    X_train = pl.read_parquet(PROC/"X_train.parquet").to_numpy()
    y_train = pl.read_parquet(PROC/"y_train.parquet")["log_price"].to_numpy()
    X_val   = pl.read_parquet(PROC/"X_val.parquet").to_numpy()
    y_val   = pl.read_parquet(PROC/"y_val.parquet")["log_price"].to_numpy()
    X_test  = pl.read_parquet(PROC/"X_test.parquet").to_numpy()
    y_test  = pl.read_parquet(PROC/"y_test.parquet")["log_price"].to_numpy()

    # Shift categorical values to be non-negative (LabelEncoder gives -1 for unseen)
    X_train[:, cat_indices] = np.clip(
        X_train[:, cat_indices], 0, None).astype(int)
    X_val[:, cat_indices]   = np.clip(
        X_val[:, cat_indices],   0, None).astype(int)
    X_test[:, cat_indices]  = np.clip(
        X_test[:, cat_indices],  0, None).astype(int)

    # Datasets + loaders
    train_ds = CarDataset(X_train, y_train, cat_indices, cont_indices)
    val_ds   = CarDataset(X_val,   y_val,   cat_indices, cont_indices)
    test_ds  = CarDataset(X_test,  y_test,  cat_indices, cont_indices)

    train_loader = DataLoader(train_ds, batch_size=2048,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=4096,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=4096,
                              shuffle=False, num_workers=0)

    # ── Model config ──────────────────────────────────────────────────────────
    config = {
        "emb_dim":     16,
        "hidden_dims": [256, 128, 64],
        "dropout":     0.3,
        "lr":          1e-3,
        "epochs":      30,
        "batch_size":  2048,
    }

    model = CarPriceMLP(
        cat_vocab_sizes = cat_vocab_sizes,
        emb_dim         = config["emb_dim"],
        n_cont          = len(cont_indices),
        hidden_dims     = config["hidden_dims"],
        dropout         = config["dropout"],
    ).to(DEVICE)

    log.info(f"\nModel architecture:")
    log.info(model)
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Total parameters: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5)
    criterion = nn.MSELoss()

    writer = SummaryWriter("runs/mlp_embeddings")

    # ── Training ──────────────────────────────────────────────────────────────
    best_val_mae = float("inf")
    best_epoch   = 0

    with mlflow.start_run(run_name="pytorch_mlp_embeddings"):
        mlflow.set_tag("model_type", "pytorch_mlp")
        mlflow.log_params(config)

        for epoch in range(config["epochs"]):
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, DEVICE)
            val_loss, val_preds, val_targets = eval_epoch(
                model, val_loader, criterion, DEVICE)

            metrics = compute_metrics(val_targets, val_preds)
            scheduler.step(val_loss)

            # Log to TensorBoard
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val",   val_loss,   epoch)
            writer.add_scalar("MAE/val",    metrics["mae"], epoch)
            writer.add_scalar("R2/val",     metrics["r2"],  epoch)

            # Log gradients every 5 epochs
            if epoch % 5 == 0:
                log_gradients(model, writer, epoch)

            log.info(f"Epoch {epoch+1:2d}/{config['epochs']}  "
                     f"train_loss={train_loss:.4f}  "
                     f"val_loss={val_loss:.4f}  "
                     f"val_MAE=${metrics['mae']:,.0f}  "
                     f"R²={metrics['r2']:.4f}")

            # Save best model
            if metrics["mae"] < best_val_mae:
                best_val_mae = metrics["mae"]
                best_epoch   = epoch + 1
                torch.save(model.state_dict(),
                           MODELS / "mlp_best.pt")

            mlflow.log_metrics({
                f"val_{k}": v for k, v in metrics.items()
            }, step=epoch)

        # ── Final evaluation on test set ──────────────────────────────────────
        log.info(f"\nBest val MAE: ${best_val_mae:,.0f} at epoch {best_epoch}")

        # Load best weights
        model.load_state_dict(torch.load(MODELS / "mlp_best.pt",
                                         weights_only=True))
        _, test_preds, test_targets = eval_epoch(
            model, test_loader, criterion, DEVICE)
        test_metrics = compute_metrics(test_targets, test_preds)

        log.info("\n" + "="*55)
        log.info("TEST SET RESULTS")
        log.info("="*55)
        log.info(f"  MAE:  ${test_metrics['mae']:>10,.0f}")
        log.info(f"  MAPE: {test_metrics['mape']:>9.2f}%")
        log.info(f"  RMSE: ${test_metrics['rmse']:>10,.0f}")
        log.info(f"  R²:   {test_metrics['r2']:>10.4f}")

        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        # ── Compare with LightGBM ─────────────────────────────────────────────
        log.info("\n" + "="*55)
        log.info("COMPARISON")
        log.info("="*55)
        log.info(f"  LightGBM tuned: MAE=$1,635  R²=0.974")
        log.info(f"  PyTorch MLP:    MAE=${test_metrics['mae']:,.0f}"
                 f"  R²={test_metrics['r2']:.3f}")

        writer.close()
        log.info("\nTensorBoard logs → runs/mlp_embeddings")
        log.info("Check http://localhost:6006")


if __name__ == "__main__":
    run()