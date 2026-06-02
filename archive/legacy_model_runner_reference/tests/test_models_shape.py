import pytest
import torch


def _get_lstm_classifier():
    from runner_utils.models.lstm_classifier import LSTMClassifier

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


def _get_dlinear_classifier():
    from runner_utils.models.dlinear_classifier import DLinearClassifier

    return DLinearClassifier


def test_dlinear_classifier_default_forward_shape():
    torch.manual_seed(0)
    DLinearClassifier = _get_dlinear_classifier()
    model = DLinearClassifier(seq_len=60, input_size=7)
    x = torch.randn(32, 60, 7)

    logits = model(x)

    assert logits.shape == (32, 2)
    assert logits.dtype == x.dtype
    assert logits.requires_grad is True


def test_dlinear_classifier_custom_num_classes_shape():
    torch.manual_seed(0)
    DLinearClassifier = _get_dlinear_classifier()
    model = DLinearClassifier(
        seq_len=24,
        input_size=5,
        num_classes=3,
        moving_avg_kernel=5,
    )
    x = torch.randn(4, 24, 5)

    logits = model(x)

    assert logits.shape == (4, 3)


def test_dlinear_classifier_individual_false_and_true_forward_shape():
    torch.manual_seed(0)
    DLinearClassifier = _get_dlinear_classifier()

    for individual in [False, True]:
        model = DLinearClassifier(
            seq_len=60,
            input_size=7,
            moving_avg_kernel=5,
            individual=individual,
        )
        x = torch.randn(8, 60, 7)

        logits = model(x)

        assert logits.shape == (8, 2)


def test_dlinear_classifier_default_config_values():
    DLinearClassifier = _get_dlinear_classifier()
    model = DLinearClassifier(seq_len=60, input_size=7)

    assert model.seq_len == 60
    assert model.input_size == 7
    assert model.num_classes == 2
    assert model.moving_avg_kernel == 5
    assert model.individual is False


def test_dlinear_classifier_rejects_even_moving_average_kernel():
    DLinearClassifier = _get_dlinear_classifier()

    with pytest.raises(ValueError, match="odd|moving_avg_kernel|kernel"):
        DLinearClassifier(seq_len=60, input_size=7, moving_avg_kernel=6)


def test_dlinear_classifier_rejects_non_positive_moving_average_kernel():
    DLinearClassifier = _get_dlinear_classifier()

    for moving_avg_kernel in [0, -3]:
        with pytest.raises(ValueError, match="moving_avg_kernel|kernel"):
            DLinearClassifier(
                seq_len=60,
                input_size=7,
                moving_avg_kernel=moving_avg_kernel,
            )


def test_dlinear_classifier_rejects_non_3d_input():
    DLinearClassifier = _get_dlinear_classifier()
    model = DLinearClassifier(seq_len=60, input_size=7)
    x = torch.randn(60, 7)

    with pytest.raises((AssertionError, ValueError), match="shape|dim"):
        model(x)


def test_dlinear_classifier_rejects_wrong_sequence_length():
    DLinearClassifier = _get_dlinear_classifier()
    model = DLinearClassifier(seq_len=60, input_size=7)
    x = torch.randn(4, 59, 7)

    with pytest.raises((AssertionError, ValueError), match="seq_len|shape"):
        model(x)


def test_dlinear_classifier_rejects_wrong_input_size():
    DLinearClassifier = _get_dlinear_classifier()
    model = DLinearClassifier(seq_len=60, input_size=7)
    x = torch.randn(4, 60, 6)

    with pytest.raises((AssertionError, ValueError), match="input_size|feature|shape"):
        model(x)


def test_dlinear_classifier_has_flatten_classification_head():
    DLinearClassifier = _get_dlinear_classifier()
    model = DLinearClassifier(seq_len=60, input_size=7, num_classes=2)

    assert model.classifier.in_features == 60 * 7
    assert model.classifier.out_features == 2
