"""Tests for N08 #5D-5 fusion variants — slice 1 (late-average + logit-sum)
plus the shared ``_predict_logits`` base helper they rely on."""

from __future__ import annotations

import numpy as np
import pytest

from intraday_research.models.deep_sequence.base import SequenceClassifier
from intraday_research.models.deep_sequence.dlinear import DLinearClassifier
from intraday_research.models.deep_sequence.fusion import (
    DLinearLogitsPlusTCNLogitsFusion,
    DLinearTrendPlusTCNResidualFusion,
    LateAverageProbabilitiesFusion,
    SmallFusionMLP,
)
from intraday_research.models.deep_sequence.tcn import TCNClassifier

_IMPLEMENTED = [LateAverageProbabilitiesFusion, DLinearLogitsPlusTCNLogitsFusion]

# Small/fast sub-model configs so composing two torch models stays snappy.
_DL_FAST = {"max_epochs": 3, "batch_size": 16}
_TC_FAST = {"max_epochs": 3, "batch_size": 16, "num_blocks": 2, "channels": (16, 16)}


def _make_xy(n: int = 32, length: int = 20, c: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, length, c)).astype(np.float64)
    y = (rng.random(n) < 0.5).astype(np.int8)
    y[0] = 0
    y[1] = 1
    return X, y


def _mk(cls, *, random_state: int = 0, **overrides):
    kw = dict(
        dlinear_config=dict(_DL_FAST),
        tcn_config=dict(_TC_FAST),
        random_state=random_state,
    )
    kw.update(overrides)
    return cls(**kw)


# ---- 1. Protocol + proba contract -------------------------------------

@pytest.mark.parametrize("cls", _IMPLEMENTED)
def test_is_sequence_classifier(cls):
    assert isinstance(cls(), SequenceClassifier)


@pytest.mark.parametrize("cls", _IMPLEMENTED)
def test_fit_returns_self_and_proba_contract(cls):
    X, y = _make_xy(seed=2)
    clf = _mk(cls)
    assert clf.fit(X, y) is clf
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    assert proba.dtype == np.float64
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert (proba >= 0.0).all() and (proba <= 1.0).all()


# ---- 2. Determinism ----------------------------------------------------

@pytest.mark.parametrize("cls", _IMPLEMENTED)
def test_predict_proba_empty_batch(cls):
    # n==0 must return (0, 2) like the single models (Codex P2: stable softmax
    # must not raise on a zero-size reduction).
    X, y = _make_xy(seed=2)
    clf = _mk(cls, random_state=0).fit(X, y)
    empty = np.zeros((0, 20, 2), dtype=np.float64)
    proba = clf.predict_proba(empty)
    assert proba.shape == (0, 2)
    assert proba.dtype == np.float64


@pytest.mark.parametrize("cls", _IMPLEMENTED)
def test_determinism_same_seed_bit_exact(cls):
    X, y = _make_xy(seed=3)
    p1 = _mk(cls, random_state=7).fit(X, y).predict_proba(X)
    p2 = _mk(cls, random_state=7).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(p1, p2)


# ---- 3. Fusion actually combines --------------------------------------

@pytest.mark.parametrize("cls", _IMPLEMENTED)
def test_fusion_differs_from_each_component(cls):
    X, y = _make_xy(seed=4)
    fused = _mk(cls, random_state=0).fit(X, y).predict_proba(X)
    dlin = DLinearClassifier(random_state=0, **_DL_FAST).fit(X, y).predict_proba(X)
    tcn = TCNClassifier(random_state=0, **_TC_FAST).fit(X, y).predict_proba(X)
    assert not np.allclose(fused, dlin)
    assert not np.allclose(fused, tcn)


def test_late_average_differs_from_logit_sum():
    X, y = _make_xy(seed=5)
    avg = _mk(LateAverageProbabilitiesFusion, random_state=0).fit(X, y).predict_proba(X)
    logit = _mk(DLinearLogitsPlusTCNLogitsFusion, random_state=0).fit(X, y).predict_proba(X)
    assert not np.allclose(avg, logit)


# ---- 4. _predict_logits base helper -----------------------------------

def test_predict_logits_softmaxes_to_predict_proba():
    X, y = _make_xy(seed=6)
    clf = DLinearClassifier(random_state=0, **_DL_FAST).fit(X, y)
    logits = clf._predict_logits(X)
    assert logits.shape == (X.shape[0], 2)
    assert logits.dtype == np.float64
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    sm = e / e.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(sm, clf.predict_proba(X), atol=1e-10)


# ---- 5. Guards ---------------------------------------------------------

@pytest.mark.parametrize("cls", _IMPLEMENTED)
def test_reject_random_state_none_at_fit(cls):
    X, y = _make_xy(seed=2)
    assert cls().random_state is None
    with pytest.raises(ValueError, match="random_state"):
        cls().fit(X, y)


@pytest.mark.parametrize("cls", _IMPLEMENTED)
def test_reject_random_state_bool_at_fit(cls):
    X, y = _make_xy(seed=2)
    with pytest.raises(ValueError, match="random_state"):
        _mk(cls, random_state=True).fit(X, y)


@pytest.mark.parametrize("cls", _IMPLEMENTED)
def test_reject_nested_random_state_in_dlinear_config(cls):
    with pytest.raises(ValueError, match="random_state"):
        cls(dlinear_config={"random_state": 5})


@pytest.mark.parametrize("cls", _IMPLEMENTED)
def test_reject_nested_random_state_in_tcn_config(cls):
    with pytest.raises(ValueError, match="random_state"):
        cls(tcn_config={"random_state": 5})


@pytest.mark.parametrize("cls", _IMPLEMENTED)
def test_reject_bad_dlinear_config_type(cls):
    with pytest.raises(ValueError, match="dlinear_config"):
        cls(dlinear_config=[1, 2])


@pytest.mark.parametrize("cls", _IMPLEMENTED)
def test_reject_bad_tcn_config_type(cls):
    with pytest.raises(ValueError, match="tcn_config"):
        cls(tcn_config="nope")


@pytest.mark.parametrize("cls", _IMPLEMENTED)
def test_reject_predict_before_fit(cls):
    X, _ = _make_xy(seed=2)
    with pytest.raises(RuntimeError, match="before fit"):
        cls(random_state=0).predict_proba(X)


@pytest.mark.parametrize("cls", _IMPLEMENTED)
def test_reject_predict_shape_drift(cls):
    X, y = _make_xy(n=24, c=2, seed=2)
    clf = _mk(cls, random_state=0).fit(X, y)
    X_bad, _ = _make_xy(n=5, c=3, seed=2)
    with pytest.raises(ValueError, match="differs from the fitted"):
        clf.predict_proba(X_bad)


# ---- 6. Config forwarding (delegation to sub-models) -------------------

@pytest.mark.parametrize("cls", _IMPLEMENTED)
def test_config_forwarding_valid_axis(cls):
    X, y = _make_xy(seed=2)
    clf = cls(
        dlinear_config={**_DL_FAST, "moving_avg_kernel": 7},
        tcn_config={**_TC_FAST, "kernel_size": 5},
        random_state=0,
    ).fit(X, y)
    assert clf.predict_proba(X).shape == (X.shape[0], 2)


@pytest.mark.parametrize("cls", _IMPLEMENTED)
def test_config_forwarding_bad_axis_fails_loud(cls):
    X, y = _make_xy(seed=2)
    with pytest.raises(ValueError, match="moving_avg_kernel"):
        cls(dlinear_config={**_DL_FAST, "moving_avg_kernel": 4}, random_state=0).fit(X, y)


# ---- Slice 2: SmallFusionMLP (chronological-OOF stacking) --------------

def _mk_mlp(**overrides):
    kw = dict(
        dlinear_config=dict(_DL_FAST),
        tcn_config=dict(_TC_FAST),
        random_state=0,
    )
    kw.update(overrides)
    return SmallFusionMLP(**kw)


def test_mlp_is_sequence_classifier():
    assert isinstance(SmallFusionMLP(), SequenceClassifier)


def test_mlp_fit_returns_self_and_proba_contract():
    X, y = _make_xy(n=40, seed=2)
    clf = _mk_mlp()
    assert clf.fit(X, y) is clf
    proba = clf.predict_proba(X)
    assert proba.shape == (40, 2)
    assert proba.dtype == np.float64
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert (proba >= 0.0).all() and (proba <= 1.0).all()


def test_mlp_determinism_same_seed_bit_exact():
    X, y = _make_xy(n=40, seed=3)
    p1 = _mk_mlp(random_state=7).fit(X, y).predict_proba(X)
    p2 = _mk_mlp(random_state=7).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(p1, p2)


def test_mlp_differs_from_late_average():
    X, y = _make_xy(n=40, seed=4)
    mlp = _mk_mlp(random_state=0).fit(X, y).predict_proba(X)
    avg = _mk(LateAverageProbabilitiesFusion, random_state=0).fit(X, y).predict_proba(X)
    assert not np.allclose(mlp, avg)


@pytest.mark.parametrize("hidden", [8, 16, 32])
def test_mlp_axis_hidden_size(hidden):
    X, y = _make_xy(n=40, seed=2)
    assert _mk_mlp(mlp_hidden_size=hidden).fit(X, y).predict_proba(X).shape == (40, 2)


@pytest.mark.parametrize("dropout", [0.0, 0.05, 0.10])
def test_mlp_axis_dropout(dropout):
    X, y = _make_xy(n=40, seed=2)
    assert _mk_mlp(mlp_dropout=dropout).fit(X, y).predict_proba(X).shape == (40, 2)


def test_mlp_reject_hidden_off_grid():
    with pytest.raises(ValueError, match="mlp_hidden_size"):
        SmallFusionMLP(mlp_hidden_size=64)


def test_mlp_reject_hidden_bool_aliasing():
    with pytest.raises(ValueError, match="mlp_hidden_size"):
        SmallFusionMLP(mlp_hidden_size=True)


def test_mlp_reject_dropout_off_grid():
    with pytest.raises(ValueError, match="mlp_dropout"):
        SmallFusionMLP(mlp_dropout=0.5)


def test_mlp_reject_dropout_bool_aliasing_zero():
    with pytest.raises(ValueError, match="mlp_dropout"):
        SmallFusionMLP(mlp_dropout=False)


def test_mlp_reject_oof_tail_single_class():
    X, _ = _make_xy(n=20, seed=2)
    y = np.array([0, 1] * 7 + [0] * 6, dtype=np.int8)  # trailing OOF tail all 0
    with pytest.raises(ValueError, match="tail lacks both classes"):
        _mk_mlp(random_state=0).fit(X, y)


def test_mlp_reject_oof_prefix_single_class():
    X, _ = _make_xy(n=20, seed=2)
    y = np.array([0] * 14 + [0, 1, 0, 1, 0, 1], dtype=np.int8)  # prefix all 0
    with pytest.raises(ValueError, match="prefix lacks both classes"):
        _mk_mlp(random_state=0).fit(X, y)


def test_mlp_reject_too_few_rows():
    X = np.random.default_rng(0).standard_normal((2, 20, 2)).astype(np.float64)
    y = np.array([0, 1], dtype=np.int8)
    with pytest.raises(ValueError, match="too few rows"):
        _mk_mlp(random_state=0).fit(X, y)


def test_mlp_reject_random_state_none_at_fit():
    X, y = _make_xy(n=40, seed=2)
    assert SmallFusionMLP().random_state is None
    with pytest.raises(ValueError, match="random_state"):
        SmallFusionMLP().fit(X, y)


def test_mlp_reject_nested_random_state_in_config():
    with pytest.raises(ValueError, match="random_state"):
        SmallFusionMLP(dlinear_config={"random_state": 5})


def test_mlp_reject_predict_before_fit():
    X, _ = _make_xy(n=10, seed=2)
    with pytest.raises(RuntimeError, match="before fit"):
        SmallFusionMLP(random_state=0).predict_proba(X)


def test_mlp_predict_proba_empty_batch():
    X, y = _make_xy(n=40, seed=2)
    clf = _mk_mlp(random_state=0).fit(X, y)
    empty = np.zeros((0, 20, 2), dtype=np.float64)
    proba = clf.predict_proba(empty)
    assert proba.shape == (0, 2)
    assert proba.dtype == np.float64


def test_mlp_reject_predict_shape_drift():
    X, y = _make_xy(n=40, c=2, seed=2)
    clf = _mk_mlp(random_state=0).fit(X, y)
    X_bad, _ = _make_xy(n=5, c=3, seed=2)
    with pytest.raises(ValueError, match="differs from the fitted"):
        clf.predict_proba(X_bad)


def test_mlp_trains_on_oof_only(monkeypatch):
    """Lock the OOF contract (Codex slice-2 P3): base models fit ONLY the
    chronological prefix; the MLP's training features are sub-model logits over
    ONLY the trailing tail. A refactor that trains components on all rows or
    feeds in-sample logits to the MLP would break this."""
    import intraday_research.models.deep_sequence.dlinear as dl_mod
    import intraday_research.models.deep_sequence.tcn as tcn_mod

    rec = {"dl_fit": [], "dl_logit": [], "tcn_fit": [], "tcn_logit": []}

    class _SpyDL(dl_mod.DLinearClassifier):
        def fit(self, X, y):
            rec["dl_fit"].append(X.shape[0])
            return super().fit(X, y)

        def _predict_logits(self, X):
            rec["dl_logit"].append(X.shape[0])
            return super()._predict_logits(X)

    class _SpyTCN(tcn_mod.TCNClassifier):
        def fit(self, X, y):
            rec["tcn_fit"].append(X.shape[0])
            return super().fit(X, y)

        def _predict_logits(self, X):
            rec["tcn_logit"].append(X.shape[0])
            return super()._predict_logits(X)

    monkeypatch.setattr(dl_mod, "DLinearClassifier", _SpyDL)
    monkeypatch.setattr(tcn_mod, "TCNClassifier", _SpyTCN)

    n = 40
    n_tail = int(round(n * 0.3))  # mirrors _OOF_TAIL_FRACTION
    split_at = n - n_tail
    X, y = _make_xy(n=n, seed=2)
    _mk_mlp(random_state=0).fit(X, y)

    # Base models fit ONLY on the prefix.
    assert rec["dl_fit"] == [split_at]
    assert rec["tcn_fit"] == [split_at]
    # The MLP's OOF features are sub-model logits over ONLY the trailing tail
    # (the first _predict_logits call during fit).
    assert rec["dl_logit"][0] == n_tail
    assert rec["tcn_logit"][0] == n_tail


# ---- Slice 3: DLinearTrendPlusTCNResidualFusion (joint causal module) --

_TC_ARCH_FAST = {"num_blocks": 2, "channels": (16, 16)}


def _mk_tr(**overrides):
    kw = dict(
        dlinear_config={"moving_avg_kernel": 5},
        tcn_config=dict(_TC_ARCH_FAST),
        random_state=0,
        max_epochs=3,
        batch_size=16,
    )
    kw.update(overrides)
    return DLinearTrendPlusTCNResidualFusion(**kw)


def test_tr_is_sequence_classifier():
    assert isinstance(DLinearTrendPlusTCNResidualFusion(), SequenceClassifier)


def test_tr_fit_returns_self_and_proba_contract():
    X, y = _make_xy(n=32, seed=2)
    clf = _mk_tr()
    assert clf.fit(X, y) is clf
    proba = clf.predict_proba(X)
    assert proba.shape == (32, 2)
    assert proba.dtype == np.float64
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert (proba >= 0.0).all() and (proba <= 1.0).all()


def test_tr_determinism_same_seed_bit_exact():
    X, y = _make_xy(n=32, seed=3)
    p1 = _mk_tr(random_state=7).fit(X, y).predict_proba(X)
    p2 = _mk_tr(random_state=7).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(p1, p2)


def test_tr_causal_no_future_leak():
    # THE key slice-3 gate: the causal trailing MA must keep the residual TCN
    # branch causal end-to-end (a centered MA would leak future into residual[<=t]).
    X, y = _make_xy(n=12, c=2, seed=3)
    clf = _mk_tr(random_state=0).fit(X, y)
    rng = np.random.default_rng(99)
    t = 10
    xa = rng.standard_normal((1, 20, 2)).astype(np.float64)
    xb = xa.copy()
    xb[:, t + 1:, :] = rng.standard_normal((1, 20 - t - 1, 2))  # perturb the FUTURE
    fa = clf._forward_features(xa)  # (1, c_last, L)
    fb = clf._forward_features(xb)
    np.testing.assert_array_equal(fa[:, :, : t + 1], fb[:, :, : t + 1])
    assert not np.array_equal(fa[:, :, t + 1:], fb[:, :, t + 1:])  # future does differ


def test_tr_fit_differs_from_late_average():
    X, y = _make_xy(n=32, seed=4)
    tr = _mk_tr(random_state=0).fit(X, y).predict_proba(X)
    avg = _mk(LateAverageProbabilitiesFusion, random_state=0).fit(X, y).predict_proba(X)
    assert not np.allclose(tr, avg)


@pytest.mark.parametrize("kernel", [3, 5, 7, 11])
def test_tr_axis_moving_avg_kernel(kernel):
    X, y = _make_xy(n=32, seed=2)
    clf = _mk_tr(dlinear_config={"moving_avg_kernel": kernel}).fit(X, y)
    assert clf.predict_proba(X).shape == (32, 2)


def test_tr_reject_bad_moving_avg_kernel():
    with pytest.raises(ValueError, match="moving_avg_kernel"):
        DLinearTrendPlusTCNResidualFusion(dlinear_config={"moving_avg_kernel": 4})


def test_tr_reject_unsupported_dlinear_key():
    with pytest.raises(ValueError, match="moving_avg_kernel"):
        DLinearTrendPlusTCNResidualFusion(dlinear_config={"individual_channels": True})


def test_tr_reject_bad_tcn_axis():
    with pytest.raises(ValueError, match="kernel_size"):
        DLinearTrendPlusTCNResidualFusion(tcn_config={"kernel_size": 4})


def test_tr_tcn_config_forwarding():
    X, y = _make_xy(n=32, seed=2)
    clf = _mk_tr(
        tcn_config={"num_blocks": 3, "channels": (32, 32, 32), "gating": True}
    ).fit(X, y)
    assert clf.predict_proba(X).shape == (32, 2)


def test_tr_reject_random_state_none_at_fit():
    X, y = _make_xy(n=32, seed=2)
    assert DLinearTrendPlusTCNResidualFusion().random_state is None
    with pytest.raises(ValueError, match="random_state"):
        DLinearTrendPlusTCNResidualFusion().fit(X, y)


def test_tr_reject_nested_random_state():
    with pytest.raises(ValueError, match="random_state"):
        DLinearTrendPlusTCNResidualFusion(tcn_config={"random_state": 5})


def test_tr_reject_predict_before_fit():
    X, _ = _make_xy(n=10, seed=2)
    with pytest.raises(RuntimeError, match="before fit"):
        _mk_tr(random_state=0).predict_proba(X)


def test_tr_reject_predict_shape_drift():
    X, y = _make_xy(n=32, c=2, seed=2)
    clf = _mk_tr(random_state=0).fit(X, y)
    X_bad, _ = _make_xy(n=5, c=3, seed=2)
    with pytest.raises(ValueError, match="differs from the fitted"):
        clf.predict_proba(X_bad)


def test_tr_reject_tcn_training_kwargs():
    # tcn_config is residual-branch architecture only; training kwargs belong on
    # the fusion class itself (Codex slice-3 P2 — reject, don't silently ignore).
    with pytest.raises(ValueError, match="architecture axes only"):
        DLinearTrendPlusTCNResidualFusion(tcn_config={"max_epochs": 3})


def test_tr_causal_moving_average_formula():
    # Lock the trailing/left-pad causal MA formula incl. the first pad timesteps
    # (Codex slice-3 P3): kernel 5, left-replicate of x0.
    import torch

    from intraday_research.models.deep_sequence.fusion import _TrendResidualModule

    tcn_module = TCNClassifier(num_blocks=2, channels=(16, 16))._build_module(
        window_size=5, n_features=1
    )
    mod = _TrendResidualModule(
        window_size=5, n_features=1, ma_kernel=5, tcn_module=tcn_module
    )
    x = torch.tensor([[[0.0], [1.0], [2.0], [3.0], [4.0]]])  # (1, 5, 1)
    trend = mod._causal_moving_average(x).squeeze().tolist()
    assert trend == pytest.approx([0.0, 0.2, 0.6, 1.2, 2.0])
