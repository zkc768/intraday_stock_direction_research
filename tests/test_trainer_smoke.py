import copy
import math

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _get_trainer_api():
    from ml_utils.trainer import Trainer, evaluate, train_one_epoch

    return train_one_epoch, evaluate, Trainer


def _tiny_classification_loader(batch_size=4, constant_labels=False):
    x = torch.tensor(
        [
            [-2.0, -1.0],
            [-1.5, -2.0],
            [-1.0, -0.5],
            [-0.5, -1.0],
            [0.5, 1.0],
            [1.0, 0.5],
            [1.5, 2.0],
            [2.0, 1.5],
        ],
        dtype=torch.float32,
    )
    if constant_labels:
        y = torch.zeros(x.shape[0], dtype=torch.long)
    else:
        y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def _tiny_model():
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    )


def _criterion():
    return nn.CrossEntropyLoss()


def _assert_loss_accuracy(result):
    assert {"loss", "accuracy"}.issubset(result)
    assert isinstance(result["loss"], float)
    assert math.isfinite(result["loss"])
    assert isinstance(result["accuracy"], float)
    assert 0.0 <= result["accuracy"] <= 1.0


def _assert_history_keys(history):
    assert {"train_loss", "val_loss", "val_macro_f1", "best_metric", "best_epoch"}.issubset(
        history
    )


def test_train_one_epoch_returns_loss_and_accuracy():
    train_one_epoch, _, _ = _get_trainer_api()
    model = _tiny_model()
    initial_state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    result = train_one_epoch(
        model=model,
        loader=_tiny_classification_loader(),
        optimizer=optimizer,
        criterion=_criterion(),
        device="cpu",
    )

    _assert_loss_accuracy(result)
    assert any(
        not torch.equal(initial_state[name], parameter)
        for name, parameter in model.state_dict().items()
    )


def test_train_one_epoch_supports_gradient_clipping():
    train_one_epoch, _, _ = _get_trainer_api()
    model = _tiny_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    result = train_one_epoch(
        model=model,
        loader=_tiny_classification_loader(),
        optimizer=optimizer,
        criterion=_criterion(),
        device="cpu",
        grad_clip=0.1,
    )

    _assert_loss_accuracy(result)


def test_evaluate_returns_metrics_and_predictions_without_training_side_effects():
    _, evaluate, _ = _get_trainer_api()
    model = _tiny_model()
    model.train()
    initial_state = copy.deepcopy(model.state_dict())

    metrics_dict, y_true, y_pred = evaluate(
        model=model,
        loader=_tiny_classification_loader(),
        criterion=_criterion(),
        device="cpu",
    )

    assert {"loss", "accuracy", "macro_f1", "balanced_accuracy", "confusion_matrix"}.issubset(
        metrics_dict
    )
    assert isinstance(y_true, np.ndarray)
    assert isinstance(y_pred, np.ndarray)
    assert len(y_true) == len(y_pred) == len(_tiny_classification_loader().dataset)
    assert all(
        torch.equal(initial_state[name], parameter)
        for name, parameter in model.state_dict().items()
    )


def test_trainer_fit_creates_best_and_last_checkpoints(tmp_path):
    _, _, Trainer = _get_trainer_api()
    model = _tiny_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=_criterion(),
        scheduler=None,
        device="cpu",
        checkpoint_dir=str(tmp_path),
        monitor_metric="val_macro_f1",
        monitor_mode="max",
        early_stop_patience=5,
    )

    history = trainer.fit(
        train_loader=_tiny_classification_loader(),
        val_loader=_tiny_classification_loader(),
        num_epochs=3,
    )

    assert isinstance(history, dict)
    _assert_history_keys(history)
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "last.pt").exists()


def test_trainer_fit_returns_history_with_expected_keys(tmp_path):
    _, _, Trainer = _get_trainer_api()
    model = _tiny_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=_criterion(),
        scheduler=None,
        device="cpu",
        checkpoint_dir=str(tmp_path),
        monitor_metric="val_macro_f1",
        monitor_mode="max",
        early_stop_patience=5,
    )

    history = trainer.fit(
        train_loader=_tiny_classification_loader(),
        val_loader=_tiny_classification_loader(),
        num_epochs=2,
    )

    _assert_history_keys(history)
    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2
    assert len(history["val_macro_f1"]) == 2


def test_trainer_early_stopping_stops_before_max_epochs(tmp_path):
    _, _, Trainer = _get_trainer_api()
    model = _tiny_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=_criterion(),
        scheduler=None,
        device="cpu",
        checkpoint_dir=str(tmp_path),
        monitor_metric="val_macro_f1",
        monitor_mode="max",
        early_stop_patience=1,
    )

    history = trainer.fit(
        train_loader=_tiny_classification_loader(constant_labels=True),
        val_loader=_tiny_classification_loader(constant_labels=True),
        num_epochs=10,
    )

    _assert_history_keys(history)
    assert len(history["train_loss"]) < 10


def test_trainer_rejects_invalid_monitor_mode(tmp_path):
    _, _, Trainer = _get_trainer_api()
    model = _tiny_model()

    with pytest.raises(ValueError, match="monitor_mode"):
        Trainer(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            criterion=_criterion(),
            scheduler=None,
            device="cpu",
            checkpoint_dir=str(tmp_path),
            monitor_metric="val_macro_f1",
            monitor_mode="sideways",
            early_stop_patience=5,
        )


def test_trainer_steps_non_plateau_scheduler_per_epoch(tmp_path):
    _, _, Trainer = _get_trainer_api()
    model = _tiny_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=_criterion(),
        scheduler=scheduler,
        device="cpu",
        checkpoint_dir=str(tmp_path),
        monitor_metric="val_macro_f1",
        monitor_mode="max",
        early_stop_patience=5,
    )

    trainer.fit(
        train_loader=_tiny_classification_loader(),
        val_loader=_tiny_classification_loader(),
        num_epochs=2,
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.025)
