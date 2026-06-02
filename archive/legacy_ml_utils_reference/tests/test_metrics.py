import numpy as np
import pytest


METRIC_KEYS = {
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "confusion_matrix",
    "classification_report",
}


def _metrics_module():
    import ml_utils.metrics as metrics

    return metrics


def _assert_metric_keys(result):
    assert isinstance(result, dict)
    assert METRIC_KEYS.issubset(result)


def test_compute_classification_metrics_matches_hand_calculated_values():
    metrics = _metrics_module()
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    result = metrics.compute_classification_metrics(y_true, y_pred)

    _assert_metric_keys(result)
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert result["balanced_accuracy"] == pytest.approx(0.75)
    assert result["precision_macro"] == pytest.approx((1.0 + 2 / 3) / 2)
    assert result["recall_macro"] == pytest.approx(0.75)
    np.testing.assert_array_equal(
        result["confusion_matrix"],
        np.array([[1, 1], [0, 2]]),
    )
    assert isinstance(result["classification_report"], dict)


def test_compute_metrics_confusion_matrix_stays_2x2_when_class_missing():
    metrics = _metrics_module()
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 1, 1])

    result = metrics.compute_classification_metrics(y_true, y_pred)

    assert result["confusion_matrix"].shape == (2, 2)
    np.testing.assert_array_equal(
        result["confusion_matrix"],
        np.array([[0, 3], [0, 0]]),
    )


def test_classification_report_uses_binary_class_keys_and_zero_division_safe():
    metrics = _metrics_module()
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 1, 1])

    result = metrics.compute_classification_metrics(y_true, y_pred)

    report = result["classification_report"]
    assert "0" in report
    assert "1" in report
    assert "accuracy" in report
    assert "macro avg" in report
    assert "weighted avg" in report


def test_always_predict_baseline_metrics_for_always_up():
    metrics = _metrics_module()
    y_test = np.array([0, 0, 0, 1])

    result = metrics.always_predict_baseline_metrics(y_test, constant_label=1)

    _assert_metric_keys(result)
    np.testing.assert_array_equal(
        result["confusion_matrix"],
        np.array([[0, 3], [0, 1]]),
    )
    assert result["accuracy"] == pytest.approx(0.25)
    assert result["balanced_accuracy"] == pytest.approx(0.5)
    assert result["macro_f1"] == pytest.approx(0.2)


def test_always_predict_baseline_metrics_for_always_down():
    metrics = _metrics_module()
    y_test = np.array([0, 1, 1, 1])

    result = metrics.always_predict_baseline_metrics(y_test, constant_label=0)

    _assert_metric_keys(result)
    np.testing.assert_array_equal(
        result["confusion_matrix"],
        np.array([[1, 0], [3, 0]]),
    )
    assert result["accuracy"] == pytest.approx(0.25)
    assert result["balanced_accuracy"] == pytest.approx(0.5)
    assert result["macro_f1"] == pytest.approx(0.2)


def test_dummy_baseline_metrics_supports_required_strategies():
    metrics = _metrics_module()
    y_train = np.array([0, 0, 0, 1, 1])
    y_test = np.array([0, 1, 1, 0])

    for strategy in ("stratified", "most_frequent", "prior", "uniform"):
        result = metrics.dummy_baseline_metrics(
            y_train,
            y_test,
            strategy=strategy,
            random_state=0,
        )

        _assert_metric_keys(result)
        assert result["confusion_matrix"].shape == (2, 2)
        assert isinstance(result["accuracy"], float)
        assert isinstance(result["macro_f1"], float)
        assert isinstance(result["balanced_accuracy"], float)


def test_dummy_baseline_metrics_prior_matches_hand_calculated_majority_prediction():
    metrics = _metrics_module()
    y_train = np.array([0, 0, 0, 1])
    y_test = np.array([0, 1, 1, 1])

    result = metrics.dummy_baseline_metrics(
        y_train,
        y_test,
        strategy="prior",
        random_state=0,
    )

    _assert_metric_keys(result)
    np.testing.assert_array_equal(
        result["confusion_matrix"],
        np.array([[1, 0], [3, 0]]),
    )
    assert result["accuracy"] == pytest.approx(0.25)
    assert result["balanced_accuracy"] == pytest.approx(0.5)
    assert result["macro_f1"] == pytest.approx(0.2)


def test_dummy_baseline_metrics_stratified_is_reproducible_with_random_state():
    metrics = _metrics_module()
    y_train = np.array([0, 0, 0, 1, 1])
    y_test = np.array([0, 1, 1, 0])

    first = metrics.dummy_baseline_metrics(
        y_train,
        y_test,
        strategy="stratified",
        random_state=42,
    )
    second = metrics.dummy_baseline_metrics(
        y_train,
        y_test,
        strategy="stratified",
        random_state=42,
    )

    np.testing.assert_array_equal(
        first["confusion_matrix"],
        second["confusion_matrix"],
    )
    assert first["accuracy"] == second["accuracy"]
    assert first["macro_f1"] == second["macro_f1"]
    assert first["balanced_accuracy"] == second["balanced_accuracy"]


def test_format_metrics_table_returns_text_with_model_names_and_metrics(capsys):
    metrics = _metrics_module()
    metrics_dict = {
        "model_a": {
            "accuracy": 0.75,
            "macro_f1": 0.7333333333,
            "balanced_accuracy": 0.75,
            "precision_macro": 0.8333333333,
            "recall_macro": 0.75,
        },
        "dummy_stratified": {
            "accuracy": 0.5,
            "macro_f1": 0.5,
            "balanced_accuracy": 0.5,
            "precision_macro": 0.5,
            "recall_macro": 0.5,
        },
    }

    table = metrics.format_metrics_table(metrics_dict)

    assert isinstance(table, str)
    assert "model_a" in table
    assert "dummy_stratified" in table
    assert "macro_f1" in table
    assert "balanced_accuracy" in table
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_metrics_functions_do_not_print(capsys):
    metrics = _metrics_module()
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_train = np.array([0, 0, 0, 1, 1])
    y_test = np.array([0, 1, 1, 0])

    metrics.compute_classification_metrics(y_true, y_pred)
    metrics.always_predict_baseline_metrics(y_test, constant_label=1)
    metrics.dummy_baseline_metrics(
        y_train,
        y_test,
        strategy="stratified",
        random_state=0,
    )
    metrics.format_metrics_table(
        {
            "model_a": {
                "accuracy": 0.75,
                "macro_f1": 0.7333333333,
                "balanced_accuracy": 0.75,
                "precision_macro": 0.8333333333,
                "recall_macro": 0.75,
            }
        }
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
