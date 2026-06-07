"""Smoke test for the src-layout package skeleton (Phase 2B)."""

import intraday_research


def test_package_imports():
    assert hasattr(intraday_research, "__version__")
    assert isinstance(intraday_research.__version__, str)


def test_research_scope_is_validation_only():
    assert intraday_research.__research_scope__ == "validation_only"


def test_holdout_test_authorization_is_false():
    assert intraday_research.__holdout_test_authorized__ is False


def test_package_resolves_under_src_layout():
    """Ensure the package import resolves under src/, not a stray flat layout."""
    import pathlib

    p = pathlib.Path(intraday_research.__file__).resolve().as_posix()
    assert "/src/intraday_research/" in p, p
