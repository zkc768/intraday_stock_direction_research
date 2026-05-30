import inspect

import pytest
import torch


def _get_classifier():
    from ml_utils.models.ms_dlinear_tcn_classifier import MultiScaleDLinearTCNClassifier

    return MultiScaleDLinearTCNClassifier


def _make_model(**overrides):
    MultiScaleDLinearTCNClassifier = _get_classifier()
    kwargs = {
        "seq_len": 12,
        "input_size": 5,
        "num_classes": 2,
        "moving_avg_kernels": (3, 5, 9),
        "tcn_channels": (8, 8),
        "tcn_kernel_size": 3,
        "dropout": 0.0,
    }
    kwargs.update(overrides)
    return MultiScaleDLinearTCNClassifier(**kwargs)


def test_ms_dlinear_tcn_forward_default_shape_dtype_and_grad():
    torch.manual_seed(0)
    model = _make_model()
    x = torch.randn(4, 12, 5)

    logits = model(x)
    loss = logits.sum()
    loss.backward()

    assert logits.shape == (4, 2)
    assert logits.dtype == x.dtype
    assert logits.device == x.device
    assert logits.requires_grad is True
    assert any(
        parameter.requires_grad and parameter.grad is not None
        for parameter in model.parameters()
    )


def test_ms_dlinear_tcn_custom_num_classes_shape():
    torch.manual_seed(0)
    model = _make_model(seq_len=10, input_size=3, num_classes=3)
    x = torch.randn(2, 10, 3)

    logits = model(x)

    assert logits.shape == (2, 3)


def test_ms_dlinear_tcn_outputs_raw_logits_not_probabilities():
    torch.manual_seed(0)
    model = _make_model()
    with torch.no_grad():
        model.fusion_classifier.weight.zero_()
        model.fusion_classifier.bias.copy_(torch.tensor([-1.0, 3.0]))
    x = torch.randn(6, 12, 5)

    logits = model(x)

    assert logits.shape == (6, 2)
    assert not torch.all((logits >= 0.0) & (logits <= 1.0))
    assert not torch.allclose(
        logits.sum(dim=1),
        torch.ones(logits.shape[0], dtype=logits.dtype),
        atol=1e-4,
        rtol=1e-4,
    )


def test_ms_dlinear_tcn_has_no_softmax_or_sigmoid_module():
    model = _make_model()
    forbidden_modules = (torch.nn.Softmax, torch.nn.Sigmoid)

    assert not any(
        isinstance(module, forbidden_modules)
        for module in model.modules()
    )


def test_ms_dlinear_tcn_rejects_non_3d_input():
    model = _make_model()
    x = torch.randn(12, 5)

    with pytest.raises((AssertionError, ValueError), match="shape|dim|3D"):
        model(x)


def test_ms_dlinear_tcn_rejects_wrong_sequence_length():
    model = _make_model(seq_len=12, input_size=5)
    x = torch.randn(4, 11, 5)

    with pytest.raises((AssertionError, ValueError), match="seq_len|shape"):
        model(x)


def test_ms_dlinear_tcn_rejects_wrong_input_size():
    model = _make_model(seq_len=12, input_size=5)
    x = torch.randn(4, 12, 4)

    with pytest.raises((AssertionError, ValueError), match="input_size|feature|shape"):
        model(x)


def test_ms_dlinear_tcn_rejects_ncl_transposed_input_from_caller():
    model = _make_model(seq_len=12, input_size=5)
    x = torch.randn(4, 5, 12)

    with pytest.raises((AssertionError, ValueError), match="seq_len|input_size|shape"):
        model(x)


def test_ms_dlinear_tcn_rejects_empty_moving_average_kernels():
    MultiScaleDLinearTCNClassifier = _get_classifier()

    with pytest.raises(ValueError, match="moving_avg_kernels|non-empty"):
        MultiScaleDLinearTCNClassifier(
            seq_len=12,
            input_size=5,
            moving_avg_kernels=(),
            tcn_channels=(8, 8),
        )


@pytest.mark.parametrize("field_name", ["seq_len", "input_size", "num_classes"])
def test_ms_dlinear_tcn_rejects_non_positive_core_dimensions(field_name):
    kwargs = {
        "seq_len": 12,
        "input_size": 5,
        "num_classes": 2,
        "moving_avg_kernels": (3, 5, 9),
        "tcn_channels": (8, 8),
    }
    kwargs[field_name] = 0
    MultiScaleDLinearTCNClassifier = _get_classifier()

    with pytest.raises(ValueError, match=field_name):
        MultiScaleDLinearTCNClassifier(**kwargs)


def test_ms_dlinear_tcn_rejects_non_positive_tcn_kernel_size():
    MultiScaleDLinearTCNClassifier = _get_classifier()

    with pytest.raises(ValueError, match="tcn_kernel_size"):
        MultiScaleDLinearTCNClassifier(
            seq_len=12,
            input_size=5,
            moving_avg_kernels=(3, 5, 9),
            tcn_channels=(8, 8),
            tcn_kernel_size=0,
        )


def test_ms_dlinear_tcn_rejects_even_moving_average_kernel():
    MultiScaleDLinearTCNClassifier = _get_classifier()

    with pytest.raises(ValueError, match="moving_avg_kernels|odd|kernel"):
        MultiScaleDLinearTCNClassifier(
            seq_len=12,
            input_size=5,
            moving_avg_kernels=(3, 6, 9),
            tcn_channels=(8, 8),
        )


def test_ms_dlinear_tcn_rejects_non_positive_moving_average_kernel():
    MultiScaleDLinearTCNClassifier = _get_classifier()

    with pytest.raises(ValueError, match="moving_avg_kernels|positive|kernel"):
        MultiScaleDLinearTCNClassifier(
            seq_len=12,
            input_size=5,
            moving_avg_kernels=(3, 0, 9),
            tcn_channels=(8, 8),
        )


def test_ms_dlinear_tcn_rejects_empty_tcn_channels():
    MultiScaleDLinearTCNClassifier = _get_classifier()

    with pytest.raises(ValueError, match="tcn_channels|non-empty"):
        MultiScaleDLinearTCNClassifier(
            seq_len=12,
            input_size=5,
            moving_avg_kernels=(3, 5, 9),
            tcn_channels=(),
        )


def test_ms_dlinear_tcn_rejects_invalid_tcn_channels():
    MultiScaleDLinearTCNClassifier = _get_classifier()

    with pytest.raises(ValueError, match="tcn_channels|positive"):
        MultiScaleDLinearTCNClassifier(
            seq_len=12,
            input_size=5,
            moving_avg_kernels=(3, 5, 9),
            tcn_channels=(8, 0),
        )


@pytest.mark.parametrize("dropout", [-0.1, 1.0])
def test_ms_dlinear_tcn_rejects_invalid_dropout(dropout):
    MultiScaleDLinearTCNClassifier = _get_classifier()

    with pytest.raises(ValueError, match="dropout"):
        MultiScaleDLinearTCNClassifier(
            seq_len=12,
            input_size=5,
            moving_avg_kernels=(3, 5, 9),
            tcn_channels=(8, 8),
            dropout=dropout,
        )


def test_ms_dlinear_tcn_forward_signature_accepts_only_x():
    MultiScaleDLinearTCNClassifier = _get_classifier()

    signature = inspect.signature(MultiScaleDLinearTCNClassifier.forward)
    parameters = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.name != "self"
    ]

    assert [parameter.name for parameter in parameters] == ["x"]


def test_ms_dlinear_tcn_constructor_has_no_stock_embedding_args():
    MultiScaleDLinearTCNClassifier = _get_classifier()

    signature = inspect.signature(MultiScaleDLinearTCNClassifier.__init__)
    forbidden_fragments = ("ticker", "stock", "embedding")

    assert not any(
        fragment in parameter_name
        for parameter_name in signature.parameters
        for fragment in forbidden_fragments
    )


def test_ms_dlinear_tcn_backward_pass_reaches_branch_and_fusion_parameters():
    torch.manual_seed(0)
    model = _make_model()
    x = torch.randn(3, 12, 5)
    y = torch.tensor([0, 1, 0], dtype=torch.long)
    criterion = torch.nn.CrossEntropyLoss()

    loss = criterion(model(x), y)
    loss.backward()

    grad_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }

    assert any("dlinear" in name or "scale" in name for name in grad_names)
    assert any("tcn" in name for name in grad_names)
    assert any("fusion" in name or "classifier" in name for name in grad_names)
