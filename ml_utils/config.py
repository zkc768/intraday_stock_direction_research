from dataclasses import dataclass, field
from typing import Any


@dataclass
class DataConfig:
    tickers: list[str]
    data_dir: str
    timestamp_col: str = "timestamp"
    price_col: str = "close"
    feature_cols: list[str] = field(default_factory=list)
    bars_per_day: int = 78
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    timezone_policy: str = "naive"

    def __post_init__(self) -> None:
        if not self.tickers:
            raise ValueError(f"DataConfig.tickers must be non-empty, got {self.tickers!r}")
        if not self.feature_cols:
            raise ValueError(f"DataConfig.feature_cols must be non-empty, got {self.feature_cols!r}")
        if not (0 < self.train_ratio < 1):
            raise ValueError(f"DataConfig.train_ratio must be in (0, 1), got {self.train_ratio!r}")
        if not (0 < self.val_ratio < 1):
            raise ValueError(f"DataConfig.val_ratio must be in (0, 1), got {self.val_ratio!r}")
        if self.train_ratio + self.val_ratio >= 1:
            raise ValueError(
                "DataConfig.train_ratio + DataConfig.val_ratio must be < 1, "
                f"got {self.train_ratio!r} + {self.val_ratio!r}"
            )
        if self.bars_per_day <= 0:
            raise ValueError(f"DataConfig.bars_per_day must be > 0, got {self.bars_per_day!r}")
        if self.timezone_policy not in {"naive", "utc"}:
            raise ValueError(f"DataConfig.timezone_policy must be 'naive' or 'utc', got {self.timezone_policy!r}")


@dataclass
class WindowConfig:
    window_size: int
    label_horizon_k: int
    stride: int = 1
    drop_cross_boundary: bool = True

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError(f"WindowConfig.window_size must be > 0, got {self.window_size!r}")
        if self.label_horizon_k <= 0:
            raise ValueError(f"WindowConfig.label_horizon_k must be > 0, got {self.label_horizon_k!r}")
        if self.stride <= 0:
            raise ValueError(f"WindowConfig.stride must be > 0, got {self.stride!r}")


@dataclass
class TrainConfig:
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float = 0.0
    grad_clip: float | None = None
    early_stop_patience: int = 10
    monitor_metric: str = "val_macro_f1"
    monitor_mode: str = "max"
    device: str = "cpu"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"TrainConfig.batch_size must be > 0, got {self.batch_size!r}")
        if self.num_epochs <= 0:
            raise ValueError(f"TrainConfig.num_epochs must be > 0, got {self.num_epochs!r}")
        if self.learning_rate <= 0:
            raise ValueError(f"TrainConfig.learning_rate must be > 0, got {self.learning_rate!r}")
        if self.monitor_mode not in {"max", "min"}:
            raise ValueError(
                f"TrainConfig.monitor_mode must be 'max' or 'min', got {self.monitor_mode!r}"
            )
        if self.device not in {"cpu", "cuda"}:
            raise ValueError(f"TrainConfig.device must be 'cpu' or 'cuda', got {self.device!r}")
        if self.early_stop_patience <= 0:
            raise ValueError(f"TrainConfig.early_stop_patience must be > 0, got {self.early_stop_patience!r}")
        if self.weight_decay < 0:
            raise ValueError(f"TrainConfig.weight_decay must be >= 0, got {self.weight_decay!r}")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValueError(f"TrainConfig.grad_clip must be > 0, got {self.grad_clip!r}")


@dataclass
class ModelConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"ModelConfig.name must be a non-empty string, got {self.name!r}")
        if not isinstance(self.params, dict):
            raise ValueError(f"ModelConfig.params must be a dict, got {self.params!r}")
