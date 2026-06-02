import pytest


def _valid_data_config_kwargs():
    return {
        "tickers": ["AAA", "BBB"],
        "data_dir": "data",
        "timestamp_col": "timestamp",
        "price_col": "close",
        "feature_cols": ["open", "high", "low", "close", "volume"],
        "bars_per_day": 78,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "timezone_policy": "naive",
    }


def _valid_train_config_kwargs():
    return {
        "batch_size": 32,
        "num_epochs": 5,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "grad_clip": None,
        "early_stop_patience": 2,
        "monitor_metric": "val_macro_f1",
        "monitor_mode": "max",
        "device": "cpu",
        "seed": 42,
    }


def test_normal_data_config_accepts_valid_values():
    from ml_utils.config import DataConfig

    config = DataConfig(**_valid_data_config_kwargs())

    assert config.tickers == ["AAA", "BBB"]
    assert config.data_dir == "data"
    assert config.timestamp_col == "timestamp"
    assert config.price_col == "close"
    assert config.feature_cols == ["open", "high", "low", "close", "volume"]
    assert config.bars_per_day == 78
    assert config.train_ratio == pytest.approx(0.7)
    assert config.val_ratio == pytest.approx(0.15)
    assert config.timezone_policy == "naive"


def test_data_config_defaults_to_legacy_binary_label_mode():
    from ml_utils.config import DataConfig

    config = DataConfig(**_valid_data_config_kwargs())

    assert config.label_mode == "legacy_binary"


def test_data_config_accepts_no_trade_band_label_mode():
    from ml_utils.config import DataConfig

    kwargs = _valid_data_config_kwargs()
    kwargs["label_mode"] = "no_trade_band"

    config = DataConfig(**kwargs)

    assert config.label_mode == "no_trade_band"


def test_data_config_rejects_unknown_label_mode():
    from ml_utils.config import DataConfig

    kwargs = _valid_data_config_kwargs()
    kwargs["label_mode"] = "three_class"

    with pytest.raises(ValueError, match="label_mode"):
        DataConfig(**kwargs)


def test_data_config_defaults_threshold_bps_to_zero():
    from ml_utils.config import DataConfig

    config = DataConfig(**_valid_data_config_kwargs())

    assert config.threshold_bps == pytest.approx(0.0)


@pytest.mark.parametrize("threshold_bps", [5, 12.5])
def test_data_config_accepts_positive_threshold_bps(threshold_bps):
    from ml_utils.config import DataConfig

    kwargs = _valid_data_config_kwargs()
    kwargs["threshold_bps"] = threshold_bps

    config = DataConfig(**kwargs)

    assert config.threshold_bps == pytest.approx(threshold_bps)


def test_data_config_rejects_negative_threshold_bps():
    from ml_utils.config import DataConfig

    kwargs = _valid_data_config_kwargs()
    kwargs["threshold_bps"] = -1.0

    with pytest.raises(ValueError, match="threshold_bps"):
        DataConfig(**kwargs)


@pytest.mark.parametrize("label_mode", ["volatility_scaled", "three_class", "threshold_sweep"])
def test_phase1b_config_does_not_add_deferred_label_modes(label_mode):
    from ml_utils.config import DataConfig

    kwargs = _valid_data_config_kwargs()
    kwargs["label_mode"] = label_mode

    with pytest.raises(ValueError, match="label_mode"):
        DataConfig(**kwargs)


def test_normal_window_config_accepts_valid_values():
    from ml_utils.config import WindowConfig

    config = WindowConfig(window_size=12, label_horizon_k=3, stride=1, drop_cross_boundary=True)

    assert config.window_size == 12
    assert config.label_horizon_k == 3
    assert config.stride == 1
    assert config.drop_cross_boundary is True


def test_normal_train_config_accepts_valid_values():
    from ml_utils.config import TrainConfig

    config = TrainConfig(**_valid_train_config_kwargs())

    assert config.batch_size == 32
    assert config.num_epochs == 5
    assert config.learning_rate == pytest.approx(0.001)
    assert config.weight_decay == pytest.approx(0.0)
    assert config.grad_clip is None
    assert config.early_stop_patience == 2
    assert config.monitor_metric == "val_macro_f1"
    assert config.monitor_mode == "max"
    assert config.device == "cpu"
    assert config.seed == 42


def test_normal_model_config_accepts_valid_values():
    from ml_utils.config import ModelConfig

    config = ModelConfig(name="lstm", params={"hidden_size": 32, "dropout": 0.1})

    assert config.name == "lstm"
    assert config.params == {"hidden_size": 32, "dropout": 0.1}


def test_boundary_data_config_accepts_minimal_valid_lists():
    from ml_utils.config import DataConfig

    config = DataConfig(
        tickers=["AAA"],
        data_dir="data",
        timestamp_col="timestamp",
        price_col="close",
        feature_cols=["close"],
        bars_per_day=1,
        train_ratio=0.01,
        val_ratio=0.01,
        timezone_policy="utc",
    )

    assert config.tickers == ["AAA"]
    assert config.feature_cols == ["close"]
    assert config.bars_per_day == 1
    assert config.train_ratio == pytest.approx(0.01)
    assert config.val_ratio == pytest.approx(0.01)
    assert config.timezone_policy == "utc"


def test_boundary_window_config_accepts_minimal_positive_values():
    from ml_utils.config import WindowConfig

    config = WindowConfig(window_size=1, label_horizon_k=1, stride=1, drop_cross_boundary=True)

    assert config.window_size == 1
    assert config.label_horizon_k == 1
    assert config.stride == 1
    assert config.drop_cross_boundary is True


def test_boundary_train_config_accepts_minimal_positive_values():
    from ml_utils.config import TrainConfig

    config = TrainConfig(
        batch_size=1,
        num_epochs=1,
        learning_rate=1e-12,
        weight_decay=0.0,
        grad_clip=None,
        early_stop_patience=1,
        monitor_metric="val_macro_f1",
        monitor_mode="min",
        device="cuda",
        seed=0,
    )

    assert config.batch_size == 1
    assert config.num_epochs == 1
    assert config.learning_rate == pytest.approx(1e-12)
    assert config.weight_decay == pytest.approx(0.0)
    assert config.grad_clip is None
    assert config.early_stop_patience == 1
    assert config.monitor_metric == "val_macro_f1"
    assert config.monitor_mode == "min"
    assert config.device == "cuda"
    assert config.seed == 0


def test_error_data_config_rejects_empty_tickers():
    from ml_utils.config import DataConfig

    kwargs = _valid_data_config_kwargs()
    kwargs["tickers"] = []

    with pytest.raises(ValueError, match="tickers"):
        DataConfig(**kwargs)


def test_error_data_config_rejects_empty_feature_cols():
    from ml_utils.config import DataConfig

    kwargs = _valid_data_config_kwargs()
    kwargs["feature_cols"] = []

    with pytest.raises(ValueError, match="feature_cols"):
        DataConfig(**kwargs)


def test_error_data_config_rejects_train_ratio_zero():
    from ml_utils.config import DataConfig

    kwargs = _valid_data_config_kwargs()
    kwargs["train_ratio"] = 0.0

    with pytest.raises(ValueError, match="train_ratio"):
        DataConfig(**kwargs)


def test_error_data_config_rejects_train_ratio_one():
    from ml_utils.config import DataConfig

    kwargs = _valid_data_config_kwargs()
    kwargs["train_ratio"] = 1.0

    with pytest.raises(ValueError, match="train_ratio"):
        DataConfig(**kwargs)


def test_error_data_config_rejects_val_ratio_zero():
    from ml_utils.config import DataConfig

    kwargs = _valid_data_config_kwargs()
    kwargs["val_ratio"] = 0.0

    with pytest.raises(ValueError, match="val_ratio"):
        DataConfig(**kwargs)


def test_error_data_config_rejects_val_ratio_one():
    from ml_utils.config import DataConfig

    kwargs = _valid_data_config_kwargs()
    kwargs["val_ratio"] = 1.0

    with pytest.raises(ValueError, match="val_ratio"):
        DataConfig(**kwargs)


def test_error_data_config_rejects_ratio_sum_not_less_than_one():
    from ml_utils.config import DataConfig

    kwargs = _valid_data_config_kwargs()
    kwargs["train_ratio"] = 0.8
    kwargs["val_ratio"] = 0.2

    with pytest.raises(ValueError, match="train_ratio.*val_ratio"):
        DataConfig(**kwargs)


def test_error_data_config_rejects_non_positive_bars_per_day():
    from ml_utils.config import DataConfig

    kwargs = _valid_data_config_kwargs()
    kwargs["bars_per_day"] = 0

    with pytest.raises(ValueError, match="bars_per_day"):
        DataConfig(**kwargs)


def test_error_data_config_rejects_invalid_timezone_policy():
    from ml_utils.config import DataConfig

    kwargs = _valid_data_config_kwargs()
    kwargs["timezone_policy"] = "local"

    with pytest.raises(ValueError, match="timezone_policy"):
        DataConfig(**kwargs)


def test_error_window_config_rejects_non_positive_window_size():
    from ml_utils.config import WindowConfig

    with pytest.raises(ValueError, match="window_size"):
        WindowConfig(window_size=0, label_horizon_k=3, stride=1, drop_cross_boundary=True)


def test_error_window_config_rejects_non_positive_label_horizon_k():
    from ml_utils.config import WindowConfig

    with pytest.raises(ValueError, match="label_horizon_k"):
        WindowConfig(window_size=12, label_horizon_k=0, stride=1, drop_cross_boundary=True)


def test_error_window_config_rejects_non_positive_stride():
    from ml_utils.config import WindowConfig

    with pytest.raises(ValueError, match="stride"):
        WindowConfig(window_size=12, label_horizon_k=3, stride=0, drop_cross_boundary=True)


def test_error_train_config_rejects_non_positive_batch_size():
    from ml_utils.config import TrainConfig

    kwargs = _valid_train_config_kwargs()
    kwargs["batch_size"] = 0

    with pytest.raises(ValueError, match="batch_size"):
        TrainConfig(**kwargs)


def test_error_train_config_rejects_non_positive_num_epochs():
    from ml_utils.config import TrainConfig

    kwargs = _valid_train_config_kwargs()
    kwargs["num_epochs"] = 0

    with pytest.raises(ValueError, match="num_epochs"):
        TrainConfig(**kwargs)


def test_error_train_config_rejects_non_positive_learning_rate():
    from ml_utils.config import TrainConfig

    kwargs = _valid_train_config_kwargs()
    kwargs["learning_rate"] = 0.0

    with pytest.raises(ValueError, match="learning_rate"):
        TrainConfig(**kwargs)


def test_error_train_config_rejects_invalid_monitor_mode():
    from ml_utils.config import TrainConfig

    kwargs = _valid_train_config_kwargs()
    kwargs["monitor_mode"] = "middle"

    with pytest.raises(ValueError, match="monitor_mode"):
        TrainConfig(**kwargs)


def test_error_train_config_rejects_invalid_device():
    from ml_utils.config import TrainConfig

    kwargs = _valid_train_config_kwargs()
    kwargs["device"] = "tpu"

    with pytest.raises(ValueError, match="device"):
        TrainConfig(**kwargs)


def test_deterministic_data_config_keeps_ratio_values_exactly():
    from ml_utils.config import DataConfig

    config = DataConfig(**_valid_data_config_kwargs())

    assert config.train_ratio == pytest.approx(0.7)
    assert config.val_ratio == pytest.approx(0.15)
    assert 1 - config.train_ratio - config.val_ratio == pytest.approx(0.15)


def test_deterministic_model_config_params_are_preserved():
    from ml_utils.config import ModelConfig

    params = {"hidden_size": 32, "dropout": 0.1}
    config = ModelConfig(name="lstm", params=params)

    assert config.params == params
    assert config.params["hidden_size"] == 32
    assert config.params["dropout"] == pytest.approx(0.1)
