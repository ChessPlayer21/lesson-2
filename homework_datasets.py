"""
Задание 2: Работа с датасетами.

2.1 Кастомный CSVDataset:
    - загрузка из CSV;
    - нормализация числовых признаков (StandardScaler);
    - one-hot-кодирование категориальных;
    - поддержка binary-колонок.

2.2 Эксперименты на реальных датасетах:
    - diabetes (регрессия);
    - breast_cancer (бинарная классификация).
"""

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from homework_model_modification import (
    LinearRegressionRegularized,
    LogisticRegressionMultiClass,
    evaluate_logistic_regression,
    plot_learning_curve,
    train_linear_regression,
    train_logistic_regression,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = "data"; PLOTS_DIR = "plots"; MODELS_DIR = "models"
for d in (DATA_DIR, PLOTS_DIR, MODELS_DIR):
    os.makedirs(d, exist_ok=True)


# ============================================================
# 2.1 Кастомный Dataset
# ============================================================

@dataclass
class ColumnConfig:
    """kind: 'numeric' | 'categorical' | 'binary' | 'target'."""
    name: str
    kind: str = "numeric"

    def __post_init__(self):
        if self.kind not in ("numeric", "categorical", "binary", "target"):
            raise ValueError(f"Неизвестный тип: {self.kind}")


class CSVDataset(Dataset):
    """
    Универсальный CSV-Dataset.

    Параметры
    ----------
    csv_path : str
    columns : List[ColumnConfig]
    target_column : str
    normalize : bool   — применять StandardScaler к numeric;
    task : str         — 'regression' | 'classification';
    fit_scaler : bool  — обучить scaler или использовать готовый;
    scaler : Optional[StandardScaler]
    df : Optional[pd.DataFrame]  — если передана, CSV не читается.
    """

    def __init__(self, csv_path: str, columns: List[ColumnConfig],
                 target_column: str, normalize: bool = True,
                 task: str = "regression", fit_scaler: bool = True,
                 scaler: Optional[StandardScaler] = None,
                 df: Optional[pd.DataFrame] = None):
        assert task in ("regression", "classification")
        self.task = task
        self.columns = columns
        self.target_column = target_column
        self.normalize = normalize

        self.df = pd.read_csv(csv_path) if df is None else df.copy()
        self._validate_columns()

        self.numeric_cols = [c.name for c in columns
                             if c.kind == "numeric" and c.name != target_column]
        self.binary_cols  = [c.name for c in columns
                             if c.kind == "binary"  and c.name != target_column]
        self.cat_cols     = [c.name for c in columns
                             if c.kind == "categorical" and c.name != target_column]

        self._preprocess()

        if self.normalize and self.numeric_cols:
            if fit_scaler:
                self.scaler = StandardScaler().fit(self.X[:, self._numeric_idx])
            else:
                if scaler is None:
                    raise ValueError("fit_scaler=False требует scaler")
                self.scaler = scaler
            self.X[:, self._numeric_idx] = self.scaler.transform(
                self.X[:, self._numeric_idx])
        else:
            self.scaler = None

        self.X = self.X.astype(np.float32)
        self.y = (self.y.astype(np.float32).reshape(-1) if task == "regression"
                  else self.y.astype(np.int64).reshape(-1))

    def _validate_columns(self):
        missing = [c.name for c in self.columns if c.name not in self.df.columns]
        if missing:
            raise ValueError(f"Нет колонок: {missing}")

    def _preprocess(self):
        frames: List[pd.DataFrame] = []
        self.feature_names: List[str] = []
        self._numeric_idx: List[int] = []

        for col in self.numeric_cols:
            self._numeric_idx.append(len(self.feature_names))
            self.feature_names.append(col)
            frames.append(self.df[[col]].astype(np.float32))

        for col in self.binary_cols:
            vals = self.df[col]
            uniq = set(vals.dropna().unique().tolist())
            if not uniq.issubset({0, 1}):
                mapping = {v: i for i, v in enumerate(sorted(uniq))}
                vals = vals.map(mapping)
            self.feature_names.append(col)
            frames.append(vals.astype(np.float32).to_frame())

        if self.cat_cols:
            dummies = pd.get_dummies(self.df[self.cat_cols],
                                     columns=self.cat_cols, dtype=np.float32)
            self.feature_names.extend(dummies.columns.tolist())
            frames.append(dummies)

        if not frames:
            raise ValueError("Нет признаков для обучения")

        self.X = pd.concat(frames, axis=1).values.astype(np.float32)
        self.y = self.df[self.target_column].values

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.tensor(self.X[idx], dtype=torch.float32),
                torch.tensor(self.y[idx]))

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def n_classes(self) -> int:
        return int(len(np.unique(self.y)))


# ============================================================
# Утилиты
# ============================================================

def _split(df: pd.DataFrame, target: str, task: str,
           test_size: float = 0.2, val_size: float = 0.2,
           seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strat = df[target] if task == "classification" else None
    tr, te = train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)
    strat = tr[target] if task == "classification" else None
    tr, va = train_test_split(tr, test_size=val_size, random_state=seed, stratify=strat)
    return tr, va, te


def save_builtin_datasets() -> Tuple[str, str]:
    """Сохраняет diabetes и breast_cancer в CSV."""
    dia = load_diabetes(as_frame=True).frame.copy()
    dia.columns = [c.replace(" ", "_").replace("(", "").replace(")", "")
                   for c in dia.columns]
    dia_path = os.path.join(DATA_DIR, "diabetes.csv")
    dia.to_csv(dia_path, index=False)

    bc = load_breast_cancer(as_frame=True).frame.copy()
    bc.columns = [c.replace(" ", "_").replace("(", "").replace(")", "")
                  for c in bc.columns]
    bc_path = os.path.join(DATA_DIR, "breast_cancer.csv")
    bc.to_csv(bc_path, index=False)

    logger.info("Датасеты сохранены: %s, %s", dia_path, bc_path)
    return dia_path, bc_path


# ============================================================
# 2.2 Эксперименты
# ============================================================

def experiment_regression(dia_path: str):
    """Линейная регрессия на diabetes."""
    df = pd.read_csv(dia_path)
    target = "target"
    columns = [ColumnConfig(c, "numeric") for c in df.columns if c != target] + \
              [ColumnConfig(target, "target")]

    tr, va, te = _split(df, target, "regression")
    ds_tr = CSVDataset(dia_path, columns, target, task="regression", df=tr)
    ds_va = CSVDataset(dia_path, columns, target, fit_scaler=False,
                       task="regression", df=va, scaler=ds_tr.scaler)
    ds_te = CSVDataset(dia_path, columns, target, fit_scaler=False,
                       task="regression", df=te, scaler=ds_tr.scaler)

    trl = DataLoader(ds_tr, batch_size=32, shuffle=True)
    val = DataLoader(ds_va, batch_size=64)
    tst = DataLoader(ds_te, batch_size=64)

    model = LinearRegressionRegularized(ds_tr.n_features, l1=1e-4, l2=1e-2)
    hist = train_linear_regression(model, trl, val, epochs=300, lr=0.05,
                                   optimizer_cls=torch.optim.Adam, patience=20)
    plot_learning_curve(hist, "Linear on Diabetes",
                        f"{PLOTS_DIR}/diabetes_curve.png")

    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for xb, yb in tst:
            preds.append(model(xb).squeeze(-1)); tgts.append(yb)
    preds = torch.cat(preds).numpy(); tgts = torch.cat(tgts).numpy()
    mse = float(np.mean((preds - tgts) ** 2))
    r2 = 1 - np.sum((preds - tgts) ** 2) / np.sum((tgts - tgts.mean()) ** 2)
    logger.info("Diabetes | test MSE=%.4f | R²=%.4f", mse, r2)
    torch.save(model.state_dict(), f"{MODELS_DIR}/linear_diabetes.pt")


def experiment_classification(bc_path: str):
    """Логистическая регрессия на breast_cancer."""
    df = pd.read_csv(bc_path)
    target = "target"
    columns = [ColumnConfig(c, "numeric") for c in df.columns if c != target] + \
              [ColumnConfig(target, "target")]

    tr, va, _ = _split(df, target, "classification")
    ds_tr = CSVDataset(bc_path, columns, target, task="classification", df=tr)
    ds_va = CSVDataset(bc_path, columns, target, fit_scaler=False,
                       task="classification", df=va, scaler=ds_tr.scaler)

    trl = DataLoader(ds_tr, batch_size=32, shuffle=True)
    val = DataLoader(ds_va, batch_size=64)

    model = LogisticRegressionMultiClass(ds_tr.n_features, n_classes=2)
    train_logistic_regression(model, trl, val, epochs=100, lr=0.1,
                              optimizer_cls=torch.optim.Adam, weight_decay=1e-4)
    metrics, _, _ = evaluate_logistic_regression(
        model, val, class_names=["malignant", "benign"],
        save_cm_path=f"{PLOTS_DIR}/breast_cancer_cm.png")
    logger.info("BreastCancer | val: %s", metrics)
    torch.save(model.state_dict(), f"{MODELS_DIR}/logit_breast_cancer.pt")


def _demo():
    dia, bc = save_builtin_datasets()
    experiment_regression(dia)
    experiment_classification(bc)


if __name__ == "__main__":
    _demo()