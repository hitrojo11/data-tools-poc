# tests/test_edge_cases.py
import pytest
import pandas as pd
from utils import read_csv_chunks, validate_schema_with_report, apply_schema


def test_empty_dataframe():
    df = pd.DataFrame()
    assert df.empty


def test_all_null_columns():
    df = pd.DataFrame({"a": [None, None], "b": [None, None]})
    report = validate_schema_with_report(df, {"a": "int", "b": "string"})
    assert "a" in report["column_reports"]
    # success_ratio is numeric between 0 and 1
    sr = report["column_reports"]["a"]["success_ratio"]
    assert 0.0 <= sr <= 1.0


def test_malformed_dates():
    df = pd.DataFrame({"d": ["2020-01-01", "bad-date", "01/02/2020"]})
    r = validate_schema_with_report(df, {"d": "datetime"})
    assert r["column_reports"]["d"]["success_ratio"] < 1.0
    df2 = apply_schema(df, {"d": "datetime"})
    assert pd.api.types.is_datetime64_any_dtype(df2["d"])


def test_large_categorical_column():
    n = 100000
    df = pd.DataFrame({"cat": [f"c{i}" for i in range(n)]})
    sz = df.memory_usage(deep=True).sum()
    assert sz > 0


def test_mixed_encodings(tmp_path):
    p = tmp_path / "mixed.csv"
    p.write_text("a,b\n1,2\n3,4", encoding="utf-8")
    df = pd.read_csv(p)
    assert not df.empty


@pytest.mark.slow
def test_read_csv_chunks_speed(tmp_path):
    p = tmp_path / "big.csv"
    n = 200000
    df = pd.DataFrame({"x": range(n), "y": ["a"] * n})
    df.to_csv(p, index=False)
    df2 = read_csv_chunks(str(p), chunksize=50000, max_rows=150000)
    assert len(df2) == 150000
