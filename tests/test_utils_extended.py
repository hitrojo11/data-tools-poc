# tests/test_utils_extra.py
import pandas as pd
from utils import (
    sample_series,
    is_numeric_ish,
    detect_outliers_iqr,
    detect_types_fast,
    # load_csv,
    # load_excel_sheets,
)


def test_sample_series_empty():
    s = pd.Series([None, None])
    sampled = sample_series(s, n=10)
    assert len(sampled) == 0


def test_sample_series_small():
    s = pd.Series(list(range(5)))
    sampled = sample_series(s, n=10)
    assert len(sampled) == 5


def test_is_numeric_ish_true():
    s = pd.Series(["1", "2", "3", "4", "5"])
    assert is_numeric_ish(s, sample_size=5) is True


def test_is_numeric_ish_mixed():
    s = pd.Series(["1", "2", "x", "4", "5"])
    assert is_numeric_ish(s, sample_size=5) is False


def test_detect_outliers_iqr_basic():
    df = pd.DataFrame({"v": [1, 2, 3, 1000]})
    mask = detect_outliers_iqr(df, "v")
    assert int(mask.sum()) >= 1


def test_detect_types_fast_basic():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    meta = detect_types_fast(df)
    assert "a" in meta["column"].values
    assert "numeric" in meta["semantic_type"].values
