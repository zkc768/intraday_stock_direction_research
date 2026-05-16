import numpy as np
import pandas as pd
import pytest


def test_label_formula_matches_hand_computed_future_average_return():
    from ml_utils.dataset import make_binary_labels_from_future_avg_return

    df = pd.DataFrame({"close": [100.0, 110.0, 132.0, 118.8, 106.92, 112.266]})

    result = make_binary_labels_from_future_avg_return(df, price_col="close", k=2)

    # For k=2, row t uses the returns t->t+1 and t+1->t+2.
    expected_future_avg = pd.Series(
        [
            np.mean([0.10, 0.20]),
            np.mean([0.20, -0.10]),
            np.mean([-0.10, -0.10]),
            np.mean([-0.10, 0.05]),
            np.nan,
            np.nan,
        ]
    )
    expected_labels = pd.Series([1.0, 1.0, 0.0, 0.0, np.nan, np.nan], name="label")

    np.testing.assert_allclose(
        result["future_avg_r"].to_numpy(),
        expected_future_avg.to_numpy(),
        rtol=1e-9,
        atol=1e-12,
        equal_nan=True,
    )
    pd.testing.assert_series_equal(result["label"], expected_labels)
    assert (expected_future_avg.iloc[:2] > 0).all()
    assert (expected_future_avg.iloc[2:4] < 0).all()
    assert len(result) == len(df)
    assert result["label"].tail(2).isna().all()
    assert not result["label"].iloc[:-2].isna().any()


def test_zero_future_average_return_maps_to_non_up_class_zero():
    from ml_utils.dataset import make_binary_labels_from_future_avg_return

    df = pd.DataFrame({"close": [100.0, 125.0, 93.75, 100.0]})

    result = make_binary_labels_from_future_avg_return(df, price_col="close", k=2)
    returns = df["close"].pct_change().shift(-1)
    expected_future_avg = np.mean([returns.iloc[0], returns.iloc[1]])

    assert returns.iloc[0] == 0.25
    assert returns.iloc[1] == -0.25
    assert expected_future_avg == 0.0
    assert result.loc[0, "future_avg_r"] == 0.0
    assert result.loc[0, "label"] == 0


def test_tail_k_rows_are_nan_and_row_count_is_preserved(raw_price_df):
    from ml_utils.dataset import make_binary_labels_from_future_avg_return

    k = 2

    result = make_binary_labels_from_future_avg_return(raw_price_df, price_col="close", k=k)

    assert len(result) == len(raw_price_df)
    assert result["label"].tail(k).isna().all()
    assert result["future_avg_r"].tail(k).isna().all()
    assert not result["label"].iloc[:-k].isna().any()


def test_label_generation_does_not_modify_input_dataframe(raw_price_df):
    from ml_utils.dataset import make_binary_labels_from_future_avg_return

    original = raw_price_df.copy(deep=True)

    make_binary_labels_from_future_avg_return(raw_price_df, price_col="close", k=2)

    pd.testing.assert_frame_equal(raw_price_df, original)
    assert "label" not in raw_price_df.columns
    assert "future_avg_r" not in raw_price_df.columns


@pytest.mark.parametrize("bad_k", [0, -1])
def test_invalid_k_raises_value_error(raw_price_df, bad_k):
    from ml_utils.dataset import make_binary_labels_from_future_avg_return

    with pytest.raises(ValueError):
        make_binary_labels_from_future_avg_return(raw_price_df, price_col="close", k=bad_k)


def test_missing_price_column_raises_clear_exception(raw_price_df):
    from ml_utils.dataset import make_binary_labels_from_future_avg_return

    with pytest.raises((ValueError, KeyError)):
        make_binary_labels_from_future_avg_return(raw_price_df, price_col="missing_close", k=2)
