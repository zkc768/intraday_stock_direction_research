import pytest
import torch


def _get_lstm_classifier():
    from ml_utils.models.lstm_classifier import LSTMClassifier

    return LSTMClassifier


def test_lstm_classifier_forward_output_shape():
    torch.manual_seed(0)
    LSTMClassifier = _get_lstm_classifier()
    model = LSTMClassifier(
        input_size=7,
        hidden_size=16,
        num_layers=1,
        num_classes=2,
        dropout=0.0,
        bidirectional=False,
    )
    x = torch.randn(32, 60, 7)

    logits = model(x)

    assert logits.shape == (32, 2)


def test_lstm_classifier_backward_pass_produces_gradients():
    torch.manual_seed(0)
    LSTMClassifier = _get_lstm_classifier()
    model = LSTMClassifier(
        input_size=7,
        hidden_size=16,
        num_layers=1,
        num_classes=2,
        dropout=0.0,
        bidirectional=False,
    )
    x = torch.randn(8, 12, 7)
    y = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.long)
    criterion = torch.nn.CrossEntropyLoss()

    loss = criterion(model(x), y)
    loss.backward()

    assert any(
        parameter.requires_grad and parameter.grad is not None
        for parameter in model.parameters()
    )


def test_lstm_classifier_bidirectional_output_shape():
    torch.manual_seed(0)
    LSTMClassifier = _get_lstm_classifier()
    model = LSTMClassifier(
        input_size=7,
        hidden_size=16,
        num_layers=1,
        num_classes=2,
        dropout=0.0,
        bidirectional=True,
    )
    x = torch.randn(10, 20, 7)

    logits = model(x)

    assert logits.shape == (10, 2)


def test_lstm_classifier_single_layer_dropout_uses_zero_lstm_dropout():
    torch.manual_seed(0)
    LSTMClassifier = _get_lstm_classifier()
    model = LSTMClassifier(
        input_size=7,
        hidden_size=16,
        num_layers=1,
        num_classes=2,
        dropout=0.5,
        bidirectional=False,
    )
    x = torch.randn(4, 6, 7)

    logits = model(x)

    assert logits.shape == (4, 2)
    assert model.lstm.dropout == 0


def test_lstm_classifier_returns_logits_not_probabilities():
    torch.manual_seed(0)
    LSTMClassifier = _get_lstm_classifier()
    model = LSTMClassifier(
        input_size=7,
        hidden_size=16,
        num_layers=1,
        num_classes=2,
        dropout=0.0,
        bidirectional=False,
    )
    x = torch.randn(32, 60, 7)

    logits = model(x)

    assert not torch.all((logits >= 0.0) & (logits <= 1.0))
    assert not torch.allclose(
        logits.sum(dim=1),
        torch.ones(logits.shape[0], dtype=logits.dtype),
        atol=1e-4,
        rtol=1e-4,
    )


def test_lstm_classifier_rejects_missing_batch_dimension():
    torch.manual_seed(0)
    LSTMClassifier = _get_lstm_classifier()
    model = LSTMClassifier(
        input_size=7,
        hidden_size=16,
        num_layers=1,
        num_classes=2,
        dropout=0.0,
        bidirectional=False,
    )
    x = torch.randn(60, 7)

    with pytest.raises((AssertionError, ValueError)):
        model(x)
