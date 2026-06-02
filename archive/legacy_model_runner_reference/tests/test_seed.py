import os
import random

import numpy as np
import torch


def test_same_seed_reproducibility():
    from ml_utils.seed import seed_everything

    seed_everything(123)
    first_random = random.random()
    first_numpy = np.random.rand(5)
    first_torch = torch.randn(5)

    seed_everything(123)
    second_random = random.random()
    second_numpy = np.random.rand(5)
    second_torch = torch.randn(5)

    assert first_random == second_random
    np.testing.assert_array_equal(first_numpy, second_numpy)
    assert torch.equal(first_torch, second_torch)


def test_different_seed_changes_torch_sequence():
    from ml_utils.seed import seed_everything

    seed_everything(123)
    first = torch.randn(5)

    seed_everything(456)
    second = torch.randn(5)

    assert not torch.equal(first, second)


def test_pythonhashseed_is_set():
    from ml_utils.seed import seed_everything

    seed_everything(789)

    assert os.environ["PYTHONHASHSEED"] == "789"


def test_deterministic_true_sets_deterministic_flags():
    original_cudnn_deterministic = torch.backends.cudnn.deterministic
    original_cudnn_benchmark = torch.backends.cudnn.benchmark
    original_deterministic_algorithms = (
        torch.are_deterministic_algorithms_enabled()
    )

    try:
        from ml_utils.seed import seed_everything

        seed_everything(123, deterministic=True)

        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        assert torch.are_deterministic_algorithms_enabled() is True
    finally:
        torch.backends.cudnn.deterministic = original_cudnn_deterministic
        torch.backends.cudnn.benchmark = original_cudnn_benchmark
        torch.use_deterministic_algorithms(
            original_deterministic_algorithms,
            warn_only=True,
        )


def test_seed_everything_returns_none():
    from ml_utils.seed import seed_everything

    assert seed_everything(123) is None
