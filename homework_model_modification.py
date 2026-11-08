"""
Задание 1: Модификация существующих моделей.

1.1 Линейная регрессия:
    - L1 (Lasso) и L2 (Ridge) регуляризация;
    - ранняя остановка (EarlyStopping).

1.2 Логистическая регрессия:
    - поддержка многоклассовой классификации (Softmax);
    - метрики: accuracy, precision, recall, F1, ROC-AUC;
    - визуализация матрицы ошибок.
"""

import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


# ============================================================
# 1.1 Линейная регрессия с L1/L2 и ранней остановкой
# ============================================================

class LinearRegressionRegularized(nn.Module):
    """
    Линейная регрессия с L1/L2-регуляризацией.

    Параметры
    ----------
    n_features : int
    l1 : float  — коэффициент L1 (Lasso)
    l2 : float  — коэффициент L2 (Ridge)
    """

    def __init__(self, n_features: int, l1: float = 0.0, l2: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.l1 = l1
        self.l2 = l2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def regularization_loss(self) -> torch.Tensor:
        """L1 + L2 штрафы по весам (bias не регуляризуем)."""
        w = self.linear.weight
        loss = torch.tensor(0.0, device=w.device)
        if self.l1 > 0:
            loss = loss + self.l1 * w.abs().sum()
        if self.l2 > 0:
            loss = loss + self.l2 * (w ** 2).sum()
        return loss


class EarlyStopping:
    """
    Ранняя остановка по валидационной метрике.

    patience : int      — сколько эпох ждать улучшения;
    min_delta : float   — минимальный прирост, считающийся улучшением;
    mode : str          — 'min' (loss) или 'max' (accuracy / R²).
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4,
                 mode: str = "min"):
        assert mode in ("min", "max")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        improved = (
            score < self.best_score - self.min_delta
            if self.mode == "min"
            else score > self.best_score + self.min_delta
        )
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_linear_regression(
    model: LinearRegressionRegularized,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 200,
    lr: float = 0.01,
    optimizer_cls: Callable = torch.optim.SGD,
    patience: int = 15,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Обучает линейную регрессию с L1/L2 и ранней остановкой."""
    optimizer = optimizer_cls(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    stopper = EarlyStopping(patience=patience, mode="min")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = criterion(pred, yb) + model.regularization_loss()
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss = total / n

        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb).squeeze(-1)
                total += (criterion(pred, yb) + model.regularization_loss()).item() * xb.size(0)
                n += xb.size(0)
        val_loss = total / n

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose and (epoch == 1 or epoch % 10 == 0):
            logger.info("Linear | epoch %3d | train=%.4f | val=%.4f",
                        epoch, train_loss, val_loss)

        if stopper.step(val_loss):
            logger.info("EarlyStopping сработал на эпохе %d", epoch)
            break

    return history


def plot_learning_curve(history: Dict[str, List[float]], title: str,
                        save_path: Optional[str] = None) -> None:
    """График train/val loss."""
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(title); plt.legend(); plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        logger.info("График сохранён: %s", save_path)
    plt.close()


# ============================================================
# 1.2 Логистическая регрессия: multiclass + метрики + CM
# ============================================================

class LogisticRegressionMultiClass(nn.Module):
    """
    Логистическая регрессия с поддержкой n_classes >= 2.

    - n_classes == 2 → 1 логит + BCEWithLogitsLoss;
    - n_classes  > 2 → n_classes логитов + CrossEntropyLoss.
    """

    def __init__(self, n_features: int, n_classes: int = 2):
        super().__init__()
        self.n_classes = n_classes
        out_dim = 1 if n_classes == 2 else n_classes
        self.linear = nn.Linear(n_features, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    @property
    def criterion(self) -> nn.Module:
        return (nn.BCEWithLogitsLoss() if self.n_classes == 2
                else nn.CrossEntropyLoss())


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None, n_classes: int = 2,
) -> Dict[str, float]:
    """accuracy, precision, recall, F1, ROC-AUC."""
    avg = "binary" if n_classes == 2 else "macro"
    metrics = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
    }
    try:
        if y_proba is not None:
            if n_classes == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            else:
                metrics["roc_auc"] = float(roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="macro"))
        else:
            metrics["roc_auc"] = float("nan")
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names=None,
                          save_path=None, title="Матрица ошибок"):
    """
    Визуализация матрицы ошибок.
    Гарантирует размер n_classes × n_classes, даже если какой-то класс
    не встретился в y_pred (иначе FixedLocator падает).
    """
    n_classes = len(class_names) if class_names is not None else max(
        int(y_true.max()), int(y_pred.max())) + 1
    labels = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(n_classes)
    if class_names:
        plt.xticks(ticks, class_names, rotation=45, ha="right")
        plt.yticks(ticks, class_names)
    else:
        plt.xticks(ticks)
        plt.yticks(ticks)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        logger.info("Матрица ошибок сохранена: %s", save_path)
    plt.close()


def evaluate_logistic_regression(
    model: LogisticRegressionMultiClass, loader: DataLoader,
    class_names: Optional[List[str]] = None,
    save_cm_path: Optional[str] = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Возвращает (metrics, y_true, y_pred)."""
    model.eval()
    logits_all, y_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits_all.append(model(xb))
            y_all.append(yb)
    logits = torch.cat(logits_all)
    y_true = torch.cat(y_all).cpu().numpy()

    if model.n_classes == 2:
        proba = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
        y_pred = (proba >= 0.5).astype(int)
        metrics = compute_classification_metrics(y_true, y_pred, proba, 2)
    else:
        proba = torch.softmax(logits, dim=-1).cpu().numpy()
        y_pred = proba.argmax(axis=1)
        metrics = compute_classification_metrics(
            y_true, y_pred, proba, model.n_classes)

    plot_confusion_matrix(y_true, y_pred, class_names=class_names,
                          save_path=save_cm_path)
    logger.info("Метрики: %s", {k: round(v, 4) for k, v in metrics.items()})
    return metrics, y_true, y_pred


def train_logistic_regression(
    model: LogisticRegressionMultiClass,
    train_loader: DataLoader, val_loader: DataLoader,
    epochs: int = 100, lr: float = 0.01,
    optimizer_cls: Callable = torch.optim.SGD,
    weight_decay: float = 0.0, verbose: bool = True,
) -> Dict[str, List[float]]:
    """Обучает бинарную или многоклассовую логистическую регрессию."""
    optimizer = optimizer_cls(model.parameters(), lr=lr,
                              weight_decay=weight_decay)
    criterion = model.criterion
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = (criterion(logits.squeeze(-1), yb.float())
                    if model.n_classes == 2 else criterion(logits, yb))
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss = total / n

        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                loss = (criterion(logits.squeeze(-1), yb.float())
                        if model.n_classes == 2 else criterion(logits, yb))
                total += loss.item() * xb.size(0)
                n += xb.size(0)
        val_loss = total / n

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose and (epoch == 1 or epoch % 10 == 0):
            logger.info("Logistic | epoch %3d | train=%.4f | val=%.4f",
                        epoch, train_loss, val_loss)
    return history


# ============================================================
# Демо
# ============================================================

def _demo() -> None:
    torch.manual_seed(0)

    # --- Линейная регрессия ---
    X = torch.randn(500, 5)
    w = torch.tensor([1.5, -2.0, 0.5, 3.0, -1.0])
    y = X @ w + 0.1 * torch.randn(500)
    tr = DataLoader(TensorDataset(X[:400], y[:400]), batch_size=32, shuffle=True)
    va = DataLoader(TensorDataset(X[400:], y[400:]), batch_size=64)
    lin = LinearRegressionRegularized(5, l1=1e-3, l2=1e-2)
    hist = train_linear_regression(lin, tr, va, epochs=200, lr=0.05)
    plot_learning_curve(hist, "Linear + L1/L2", f"{PLOTS_DIR}/linear_curve.png")

    # --- Бинарная логистическая ---
    Xc = torch.randn(600, 4)
    wc = torch.tensor([1.0, -1.5, 2.0, 0.5])
    p = torch.sigmoid(Xc @ wc)
    yc = (torch.rand(600) < p).long()
    tr = DataLoader(TensorDataset(Xc[:480], yc[:480]), batch_size=32, shuffle=True)
    va = DataLoader(TensorDataset(Xc[480:], yc[480:]), batch_size=64)
    log = LogisticRegressionMultiClass(4, n_classes=2)
    train_logistic_regression(log, tr, va, epochs=50, lr=0.1)
    evaluate_logistic_regression(log, va, class_names=["neg", "pos"],
                                 save_cm_path=f"{PLOTS_DIR}/cm_binary.png")

    # --- Многоклассовая (3 класса) ---
    centers = torch.tensor([[0., 0.], [3., 0.], [0., 3.]])
    Xm = torch.cat([c + 0.5 * torch.randn(200, 2) for c in centers])
    ym = torch.cat([torch.full((200,), i, dtype=torch.long) for i in range(3)])
    tr = DataLoader(TensorDataset(Xm[:450], ym[:450]), batch_size=32, shuffle=True)
    va = DataLoader(TensorDataset(Xm[450:], ym[450:]), batch_size=64)
    mc = LogisticRegressionMultiClass(2, n_classes=3)
    # Adam + больше эпох, чтобы все 3 класса гарантированно выучились
    train_logistic_regression(mc, tr, va, epochs=300, lr=0.1,
                              optimizer_cls=torch.optim.Adam)
    evaluate_logistic_regression(mc, va, class_names=["A", "B", "C"],
                                 save_cm_path=f"{PLOTS_DIR}/cm_multiclass.png")


if __name__ == "__main__":
    _demo()