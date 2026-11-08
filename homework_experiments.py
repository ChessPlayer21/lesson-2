"""
Задание 3: Эксперименты и анализ.

3.1 Гиперпараметры: learning_rate × batch_size × optimizer.
3.2 Feature Engineering: полиномы, взаимодействия, статистические признаки.
"""

import itertools
import logging
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


# ============================================================
# Данные
# ============================================================

def load_diabetes_split() -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                    np.ndarray, np.ndarray, np.ndarray]:
    """Стандартизованные train/val/test тензоры diabetes."""
    d = load_diabetes(as_frame=True)
    X = d.data.values.astype(np.float32)
    y = d.target.values.astype(np.float32)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)
    sc = StandardScaler().fit(X_tr)
    return (sc.transform(X_tr).astype(np.float32), y_tr,
            sc.transform(X_va).astype(np.float32), y_va,
            sc.transform(X_te).astype(np.float32), y_te)


class LinearRegression(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def train_one(model, trl, val, epochs, lr, opt_cls) -> float:
    """Возвращает лучший val MSE за обучение."""
    opt = opt_cls(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    best = float("inf")
    for _ in range(epochs):
        model.train()
        for xb, yb in trl:
            opt.zero_grad()
            loss = crit(model(xb), yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            total, n = 0.0, 0
            for xb, yb in val:
                total += crit(model(xb), yb).item() * xb.size(0)
                n += xb.size(0)
        best = min(best, total / n)
    return best


# ============================================================
# 3.1 Гиперпараметры
# ============================================================

def grid_search(
    X_tr, y_tr, X_va, y_va,
    learning_rates: List[float] = (0.001, 0.01, 0.05),
    batch_sizes: List[int] = (16, 32, 64),
    optimizer_names: List[str] = ("SGD", "Adam", "RMSprop"),
    epochs: int = 100,
) -> pd.DataFrame:
    """Сетка lr × batch × optimizer. Возвращает DataFrame результатов."""
    optimizers = {"SGD": torch.optim.SGD,
                  "Adam": torch.optim.Adam,
                  "RMSprop": torch.optim.RMSprop}
    rows = []
    for lr, bs, opt_name in itertools.product(
            learning_rates, batch_sizes, optimizer_names):
        trl = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
                         batch_size=bs, shuffle=True)
        val = DataLoader(TensorDataset(torch.tensor(X_va), torch.tensor(y_va)),
                         batch_size=bs)
        model = LinearRegression(X_tr.shape[1])
        best = train_one(model, trl, val, epochs, lr, optimizers[opt_name])
        logger.info("lr=%.3f bs=%3d opt=%-7s → val_MSE=%.2f",
                    lr, bs, opt_name, best)
        rows.append({"lr": lr, "batch_size": bs, "optimizer": opt_name,
                     "val_mse": best})
    return pd.DataFrame(rows)


def plot_hyperparam_results(df: pd.DataFrame, save_path: str):
    """Три subplot'а: val_mse vs lr для каждого optimizer'а."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, opt in zip(axes, df["optimizer"].unique()):
        sub = df[df["optimizer"] == opt]
        for bs in sub["batch_size"].unique():
            s2 = sub[sub["batch_size"] == bs].sort_values("lr")
            ax.plot(s2["lr"], s2["val_mse"], marker="o", label=f"bs={bs}")
        ax.set_xscale("log"); ax.set_title(opt)
        ax.set_xlabel("learning rate"); ax.grid(True); ax.legend()
    axes[0].set_ylabel("val MSE")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    logger.info("График гиперпараметров: %s", save_path)
    plt.close()


# ============================================================
# 3.2 Feature Engineering
# ============================================================

def build_features(X: np.ndarray, mode: str) -> np.ndarray:
    """
    mode:
        'base'         — исходные признаки;
        'poly'         — + полиномы степени 2;
        'interactions' — + попарные произведения;
        'stats'        — + mean, var по строке;
        'all'          — всё вместе.
    """
    feats = [X]
    if mode in ("poly", "all"):
        feats.append(PolynomialFeatures(2, include_bias=False)
                     .fit_transform(X))
    if mode in ("interactions", "all"):
        n = X.shape[1]
        inter = [X[:, i] * X[:, j] for i in range(n) for j in range(i + 1, n)]
        feats.append(np.stack(inter, axis=1))
    if mode in ("stats", "all"):
        feats.append(np.column_stack([X.mean(1), X.var(1)]))
    return np.concatenate(feats, axis=1).astype(np.float32)


def compare_feature_engineering(epochs: int = 200) -> pd.DataFrame:
    """Сравнивает базовую модель и варианты FE по val MSE / R²."""
    X_tr, y_tr, X_va, y_va, X_te, y_te = load_diabetes_split()
    rows = []
    for mode in ("base", "poly", "interactions", "stats", "all"):
        Xtr = build_features(X_tr, mode)
        Xva = build_features(X_va, mode)
        Xte = build_features(X_te, mode)

        trl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(y_tr)),
                         batch_size=32, shuffle=True)
        val = DataLoader(TensorDataset(torch.tensor(Xva), torch.tensor(y_va)),
                         batch_size=64)
        tst = DataLoader(TensorDataset(torch.tensor(Xte), torch.tensor(y_te)),
                         batch_size=64)

        model = LinearRegression(Xtr.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        crit = nn.MSELoss()

        for _ in range(epochs):
            model.train()
            for xb, yb in trl:
                opt.zero_grad()
                loss = crit(model(xb), yb); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            val_mse, n = 0.0, 0
            for xb, yb in val:
                val_mse += crit(model(xb), yb).item() * xb.size(0); n += xb.size(0)
            val_mse /= n

            preds, tgts = [], []
            for xb, yb in tst:
                preds.append(model(xb).numpy()); tgts.append(yb.numpy())
            preds = np.concatenate(preds); tgts = np.concatenate(tgts)
            test_mse = float(np.mean((preds - tgts) ** 2))
            r2 = 1 - np.sum((preds - tgts) ** 2) / np.sum((tgts - tgts.mean()) ** 2)

        logger.info("FE mode=%-12s | val_mse=%.2f | test_mse=%.2f | R²=%.4f",
                    mode, val_mse, test_mse, r2)
        rows.append({"mode": mode, "n_features": Xtr.shape[1],
                     "val_mse": val_mse, "test_mse": test_mse, "r2": r2})
    return pd.DataFrame(rows)


def plot_fe_comparison(df: pd.DataFrame, save_path: str):
    """Столбчатая диаграмма test_mse по режимам FE."""
    plt.figure(figsize=(8, 5))
    plt.bar(df["mode"], df["test_mse"])
    for i, v in enumerate(df["test_mse"]):
        plt.text(i, v, f"{v:.0f}", ha="center", va="bottom")
    plt.ylabel("Test MSE"); plt.title("Feature Engineering vs baseline")
    plt.grid(axis="y")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# ============================================================
# Точка входа
# ============================================================

def main():
    X_tr, y_tr, X_va, y_va, _, _ = load_diabetes_split()

    logger.info("=== 3.1 Grid search гиперпараметров ===")
    gs = grid_search(X_tr, y_tr, X_va, y_va, epochs=100)
    gs.to_csv(f"{PLOTS_DIR}/grid_search.csv", index=False)
    plot_hyperparam_results(gs, f"{PLOTS_DIR}/grid_search.png")

    logger.info("=== 3.2 Feature Engineering ===")
    fe = compare_feature_engineering(epochs=200)
    fe.to_csv(f"{PLOTS_DIR}/feature_engineering.csv", index=False)
    plot_fe_comparison(fe, f"{PLOTS_DIR}/feature_engineering.png")


if __name__ == "__main__":
    main()