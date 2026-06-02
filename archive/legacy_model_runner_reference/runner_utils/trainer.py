"""Minimal PyTorch trainer loop for binary classification."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from runner_utils.checkpoint import save_checkpoint
from runner_utils.metrics import compute_classification_metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    grad_clip: float | None = None,
) -> dict:
    """Train for one epoch.

    Batch x: torch.Tensor of shape (batch, ...).
    Batch y: torch.Tensor of shape (batch,).
    """
    model.to(device)
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        batch_size = int(y.shape[0])

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        predictions = torch.argmax(logits, dim=1)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((predictions == y).sum().item())
        total_count += batch_size

    return {
        "loss": total_loss / total_count,
        "accuracy": total_correct / total_count,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Evaluate a model without updating parameters.

    Batch x: torch.Tensor of shape (batch, ...).
    Batch y: torch.Tensor of shape (batch,).
    """
    model.to(device)
    model.eval()
    total_loss = 0.0
    total_count = 0
    y_true_batches = []
    y_pred_batches = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            batch_size = int(y.shape[0])

            logits = model(x)
            loss = criterion(logits, y)
            predictions = torch.argmax(logits, dim=1)

            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
            y_true_batches.append(y.detach().cpu().numpy())
            y_pred_batches.append(predictions.detach().cpu().numpy())

    y_true = np.concatenate(y_true_batches)
    y_pred = np.concatenate(y_pred_batches)
    metrics = compute_classification_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / total_count
    return metrics, y_true, y_pred


class Trainer:
    """Coordinate training, validation, checkpoints, and early stopping."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        device: str,
        checkpoint_dir: str,
        monitor_metric: str = "val_macro_f1",
        monitor_mode: str = "max",
        early_stop_patience: int = 10,
        grad_clip: float | None = None,
        verbose: bool = False,
    ):
        if monitor_mode not in {"max", "min"}:
            raise ValueError(f"monitor_mode must be 'max' or 'min', got {monitor_mode!r}")
        if early_stop_patience < 0:
            raise ValueError(
                f"early_stop_patience must be >= 0, got {early_stop_patience}"
            )

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        self.early_stop_patience = early_stop_patience
        self.grad_clip = grad_clip
        self.verbose = verbose

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
    ) -> dict:
        """Fit the model.

        train_loader x: torch.Tensor of shape (batch, ...).
        val_loader x: torch.Tensor of shape (batch, ...).
        """
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_macro_f1": [],
            "val_balanced_accuracy": [],
            "best_metric": None,
            "best_epoch": None,
        }
        best_metric = None
        best_epoch = None
        epochs_without_improvement = 0

        for epoch in range(1, num_epochs + 1):
            train_metrics = train_one_epoch(
                model=self.model,
                loader=train_loader,
                optimizer=self.optimizer,
                criterion=self.criterion,
                device=self.device,
                grad_clip=self.grad_clip,
            )
            val_metrics, _, _ = evaluate(
                model=self.model,
                loader=val_loader,
                criterion=self.criterion,
                device=self.device,
            )

            monitor_value = self._monitor_value(val_metrics)
            improved = self._is_improved(monitor_value, best_metric)
            if improved:
                best_metric = monitor_value
                best_epoch = epoch
                epochs_without_improvement = 0
                self._save_checkpoint("best.pt", epoch, best_metric, history)
            else:
                epochs_without_improvement += 1

            self._step_scheduler(monitor_value)
            self._record_history(history, train_metrics, val_metrics, best_metric, best_epoch)
            self._save_checkpoint("last.pt", epoch, best_metric, history)
            self._log_epoch(epoch, train_metrics, val_metrics, best_metric)

            if epochs_without_improvement > self.early_stop_patience:
                break

        return history

    def _monitor_value(self, val_metrics: dict) -> float:
        metric_name = self.monitor_metric
        if metric_name.startswith("val_"):
            metric_name = metric_name.removeprefix("val_")
        return float(val_metrics[metric_name])

    def _is_improved(self, current: float, best: float | None) -> bool:
        if best is None:
            return True
        if self.monitor_mode == "max":
            return current > best
        return current < best

    def _step_scheduler(self, monitor_value: float) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(monitor_value)
        else:
            self.scheduler.step()

    def _record_history(
        self,
        history: dict,
        train_metrics: dict,
        val_metrics: dict,
        best_metric: float | None,
        best_epoch: int | None,
    ) -> None:
        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["val_balanced_accuracy"].append(val_metrics["balanced_accuracy"])
        history["best_metric"] = best_metric
        history["best_epoch"] = best_epoch

    def _save_checkpoint(
        self,
        filename: str,
        epoch: int,
        best_metric: float | None,
        history: dict,
    ) -> None:
        save_checkpoint(
            path=str(self.checkpoint_dir / filename),
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            best_metric=float(best_metric),
            extra={
                "monitor_metric": self.monitor_metric,
                "monitor_mode": self.monitor_mode,
                "history": history.copy(),
            },
        )

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: dict,
        val_metrics: dict,
        best_metric: float | None,
    ) -> None:
        if self.verbose:
            print(
                f"epoch {epoch} | train_loss={train_metrics['loss']:.6f} | "
                f"val_loss={val_metrics['loss']:.6f} | "
                f"val_macro_f1={val_metrics['macro_f1']:.6f} | "
                f"best={best_metric:.6f}"
            )
