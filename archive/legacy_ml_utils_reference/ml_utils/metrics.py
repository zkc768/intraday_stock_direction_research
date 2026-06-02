"""Binary classification metrics and baselines."""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


_LABELS = [0, 1]
_SCALAR_METRICS = (
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict:
    """Compute binary classification metrics for class 0 non-up and class 1 up."""
    del y_proba
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            recall_score(
                y_true,
                y_pred,
                labels=_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "precision_macro": float(
            precision_score(
                y_true,
                y_pred,
                labels=_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true,
                y_pred,
                labels=_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=_LABELS),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=_LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }


def dummy_baseline_metrics(
    y_train: np.ndarray,
    y_test: np.ndarray,
    strategy: str = "stratified",
    random_state: int = 0,
) -> dict:
    """Fit a sklearn dummy baseline on train labels and evaluate on test labels."""
    x_train = np.zeros((len(y_train), 1))
    x_test = np.zeros((len(y_test), 1))
    classifier = DummyClassifier(strategy=strategy, random_state=random_state)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    return compute_classification_metrics(y_test, y_pred)


def always_predict_baseline_metrics(
    y_test: np.ndarray,
    constant_label: int,
) -> dict:
    """Evaluate an always-constant binary-label baseline."""
    y_pred = np.full(y_test.shape, constant_label)
    return compute_classification_metrics(y_test, y_pred)


def format_metrics_table(metrics_dict: dict[str, dict]) -> str:
    """Format scalar metrics as a readable plain-text table."""
    headers = ("model", *_SCALAR_METRICS)
    rows = []
    for model_name, values in metrics_dict.items():
        rows.append(
            (
                model_name,
                *(_format_metric_value(values[metric]) for metric in _SCALAR_METRICS),
            )
        )

    widths = [
        max(len(str(item)) for item in (header, *(row[index] for row in rows)))
        for index, header in enumerate(headers)
    ]
    lines = [_format_table_row(headers, widths)]
    lines.append(_format_table_row(tuple("-" * width for width in widths), widths))
    lines.extend(_format_table_row(row, widths) for row in rows)
    return "\n".join(lines)


def _format_metric_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _format_table_row(row: tuple, widths: list[int]) -> str:
    return " | ".join(str(value).ljust(width) for value, width in zip(row, widths))
