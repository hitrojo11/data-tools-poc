import pandas as pd
from utils import validate_schema, apply_schema


def test_validate_schema_and_apply():
    df = pd.DataFrame({"a": ["1", "2", "3"], "b": ["2020-01-01", "2020-02-01", None]})
    schema = {"a": "int", "b": "datetime"}
    report = validate_schema(df, schema)
    assert report["ok"] is True
    df2 = apply_schema(df, schema)
    assert pd.api.types.is_integer_dtype(df2["a"])
    assert pd.api.types.is_datetime64_any_dtype(df2["b"])
