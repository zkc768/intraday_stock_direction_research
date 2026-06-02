import pytest
import torch


def _get_tcn_classifier():
    from runner_utils.models.tcn_classifier import TCNClassifier

    return TCNClassifier


def _make_tcn_classifier():
    TCNClassifier = _get_tcn_classifier()
    return TCNClassifier(
        input_size=5,
        num_channels=[8, 8],
        kernel_size=3,
        dropout=0.0,
        num_classes=2,
        causal=True,
    )


def test_tcn_classifier_forward_shape():
    torch.manual_seed(0)
    model = _make_tcn_classifier()
    x = torch.randn(4, 12, 5)

    logits = model(x)

    assert logits.shape == (4, 2)


def test_tcn_classifier_outputs_raw_logits_not_probabilities():
    torch.manual_seed(0)
    model = _make_tcn_classifier()
    x = torch.randn(4, 12, 5)

    logits = model(x)

    assert logits.shape == (4, 2)
    assert not torch.allclose(
        logits.sum(dim=1),
        torch.ones(logits.shape[0], dtype=logits.dtype),
        atol=1e-4,
        rtol=1e-4,
    )


def test_tcn_classifier_accepts_different_sequence_lengths():
    torch.manual_seed(0)
    model = _make_tcn_classifier()
    short_x = torch.randn(2, 12, 5)
    long_x = torch.randn(2, 24, 5)

    short_logits = model(short_x)
    long_logits = model(long_x)

    assert short_logits.shape == (2, 2)
    assert long_logits.shape == (2, 2)


def test_tcn_classifier_rejects_non_3d_input():
    torch.manual_seed(0)
    model = _make_tcn_classifier()
    x = torch.randn(4, 5)

    with pytest.raises((AssertionError, ValueError)):
        model(x)


def test_tcn_classifier_respects_eval_shape_determinism():
    torch.manual_seed(0)
    model = _make_tcn_classifier()
    model.eval()
    x = torch.randn(3, 12, 5)

    first_logits = model(x)
    second_logits = model(x)

    assert first_logits.shape == (3, 2)
    assert second_logits.shape == (3, 2)


def test_tcn_classifier_has_no_softmax_or_sigmoid_module():
    model = _make_tcn_classifier()

    forbidden_modules = (torch.nn.Softmax, torch.nn.Sigmoid)

    assert not any(
        isinstance(module, forbidden_modules)
        for module in model.modules()
    )


def test_tcn_classifier_keeps_output_on_input_device_cpu():
    torch.manual_seed(0)
    model = _make_tcn_classifier()
    x = torch.randn(4, 12, 5)

    logits = model(x)

    assert logits.device == x.device
