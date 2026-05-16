import random
from pathlib import Path

import numpy as np
import pytest
import torch


def _checkpoint_functions():
    from ml_utils.checkpoint import load_checkpoint, save_checkpoint

    return save_checkpoint, load_checkpoint


def _build_model() -> torch.nn.Module:
    torch.manual_seed(123)
    return torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 2),
    )


def _input_tensor() -> torch.Tensor:
    return torch.arange(6, dtype=torch.float32).reshape(2, 3) / 10.0


def _take_optimizer_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad()
    loss = model(_input_tensor()).sum()
    loss.backward()
    optimizer.step()


def test_save_load_roundtrip_restores_model_weights(tmp_path: Path):
    save_checkpoint, load_checkpoint = _checkpoint_functions()
    model = _build_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint_path = tmp_path / "roundtrip.pt"
    expected = model(_input_tensor()).detach().clone()

    save_checkpoint(
        path=str(checkpoint_path),
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=3,
        best_metric=0.75,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)

    load_checkpoint(path=str(checkpoint_path), model=model, device="cpu")
    actual = model(_input_tensor()).detach()

    torch.testing.assert_close(actual, expected)


def test_optimizer_state_is_restored_when_optimizer_is_provided(tmp_path: Path):
    save_checkpoint, load_checkpoint = _checkpoint_functions()
    model = _build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    _take_optimizer_step(model, optimizer)
    checkpoint_path = tmp_path / "optimizer.pt"

    save_checkpoint(
        path=str(checkpoint_path),
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=4,
        best_metric=0.8,
    )
    fresh_model = _build_model()
    fresh_optimizer = torch.optim.Adam(fresh_model.parameters(), lr=0.01)

    checkpoint = load_checkpoint(
        path=str(checkpoint_path),
        model=fresh_model,
        optimizer=fresh_optimizer,
        device="cpu",
    )

    assert "optimizer_state_dict" in checkpoint
    assert fresh_optimizer.state_dict()["param_groups"] == checkpoint["optimizer_state_dict"]["param_groups"]
    assert len(fresh_optimizer.state_dict()["state"]) == len(checkpoint["optimizer_state_dict"]["state"])
    assert len(fresh_optimizer.state_dict()["state"]) > 0


def test_scheduler_state_is_restored_when_scheduler_is_provided(tmp_path: Path):
    save_checkpoint, load_checkpoint = _checkpoint_functions()
    model = _build_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    scheduler.step()
    checkpoint_path = tmp_path / "scheduler.pt"

    save_checkpoint(
        path=str(checkpoint_path),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=5,
        best_metric=0.81,
    )
    fresh_model = _build_model()
    fresh_optimizer = torch.optim.SGD(fresh_model.parameters(), lr=0.1)
    fresh_scheduler = torch.optim.lr_scheduler.StepLR(fresh_optimizer, step_size=2, gamma=0.5)

    checkpoint = load_checkpoint(
        path=str(checkpoint_path),
        model=fresh_model,
        optimizer=fresh_optimizer,
        scheduler=fresh_scheduler,
        device="cpu",
    )

    assert "scheduler_state_dict" in checkpoint
    assert fresh_scheduler.state_dict()["last_epoch"] == scheduler.state_dict()["last_epoch"]


def test_metadata_fields_roundtrip(tmp_path: Path):
    save_checkpoint, load_checkpoint = _checkpoint_functions()
    model = _build_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint_path = tmp_path / "metadata.pt"
    extra = {"feature_cols": ["open", "close"], "label_schema": {"non_up": 0, "up": 1}}

    save_checkpoint(
        path=str(checkpoint_path),
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=6,
        best_metric=0.82,
        extra=extra,
    )
    checkpoint = load_checkpoint(path=str(checkpoint_path), model=model, device="cpu")

    assert checkpoint["epoch"] == 6
    assert checkpoint["best_metric"] == 0.82
    assert checkpoint["extra"] == extra


def test_extra_none_roundtrips_as_none(tmp_path: Path):
    save_checkpoint, load_checkpoint = _checkpoint_functions()
    model = _build_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint_path = tmp_path / "extra_none.pt"

    save_checkpoint(
        path=str(checkpoint_path),
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=7,
        best_metric=0.83,
        extra=None,
    )
    checkpoint = load_checkpoint(path=str(checkpoint_path), model=model, device="cpu")

    assert checkpoint["extra"] is None


def test_missing_checkpoint_path_raises_clear_exception(tmp_path: Path):
    _, load_checkpoint = _checkpoint_functions()
    model = _build_model()

    with pytest.raises((FileNotFoundError, ValueError)):
        load_checkpoint(path=str(tmp_path / "missing.pt"), model=model, device="cpu")


def test_weights_only_true_restores_model_weights_and_returns_metadata(tmp_path: Path):
    save_checkpoint, load_checkpoint = _checkpoint_functions()
    model = _build_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint_path = tmp_path / "weights_only.pt"
    expected = model(_input_tensor()).detach().clone()

    save_checkpoint(
        path=str(checkpoint_path),
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=8,
        best_metric=0.84,
        extra={"run_id": "weights-only"},
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(0.0)

    checkpoint = load_checkpoint(
        path=str(checkpoint_path),
        model=model,
        device="cpu",
        weights_only=True,
    )

    torch.testing.assert_close(model(_input_tensor()).detach(), expected)
    assert checkpoint["epoch"] == 8
    assert checkpoint["best_metric"] == 0.84
    assert checkpoint["extra"] == {"run_id": "weights-only"}


def test_checkpoint_file_does_not_store_full_model_object(tmp_path: Path):
    save_checkpoint, _ = _checkpoint_functions()
    model = _build_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint_path = tmp_path / "raw.pt"

    save_checkpoint(
        path=str(checkpoint_path),
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=9,
        best_metric=0.85,
    )
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert "model" not in raw
    assert "model_state_dict" in raw


def test_rng_state_is_present(tmp_path: Path):
    save_checkpoint, _ = _checkpoint_functions()
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    model = _build_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint_path = tmp_path / "rng.pt"

    save_checkpoint(
        path=str(checkpoint_path),
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=10,
        best_metric=0.86,
    )
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert "rng_state" in raw
    assert {"python", "numpy", "torch"}.issubset(raw["rng_state"])
    assert "cuda" in raw["rng_state"]
