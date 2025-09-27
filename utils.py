# utils.py
import pandas as pd

# import numpy as np


def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


def load_excel_sheets(file):
    xl = pd.ExcelFile(file)
    return {name: xl.parse(name) for name in xl.sheet_names}


def sample_series(series: pd.Series, n=1000):
    non_null = series.dropna()
    ln = len(non_null)
    if ln == 0:
        return non_null
    if ln <= n:
        return non_null.iloc[:n]
    return non_null.sample(n=n, random_state=0)


def is_numeric_ish(series: pd.Series, sample_size=1000) -> bool:
    s = sample_series(series, sample_size)
    if len(s) == 0:
        return False
    coerced = pd.to_numeric(s, errors="coerce")
    return bool(coerced.notna().mean() >= 0.9)


def detect_types_fast(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        ser = df[col]
        pandas_dtype = str(ser.dtypes)
        n_unique = ser.nunique(dropna=True)
        pct_missing = round(ser.isna().mean() * 100, 2)
        if pd.api.types.is_numeric_dtype(ser):
            semantic = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(ser):
            semantic = "datetime"
        elif pd.api.types.is_bool_dtype(ser):
            semantic = "boolean"
        else:
            semantic = "numeric-ish" if is_numeric_ish(ser) else "categorical/text"
        rows.append(
            {
                "column": col,
                "pandas_dtype": pandas_dtype,
                "n_unique": n_unique,
                "pct_missing": pct_missing,
                "semantic_type": semantic,
            }
        )
    return pd.DataFrame(rows)


def detect_outliers_iqr(df: pd.DataFrame, col: str) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(df[col]):
        return pd.Series([False] * len(df), index=df.index)
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (df[col] < lower) | (df[col] > upper)


# in utils.py (append)
def validate_schema(df: pd.DataFrame, schema: dict) -> dict:
    errors = []
    for col, expected in schema.items():
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
            continue
        s = df[col].dropna()
        try:
            if expected in ("float", "int"):
                pd.to_numeric(s, errors="raise")
            elif expected == "datetime":
                pd.to_datetime(s, errors="raise")
        except Exception as e:
            errors.append(f"{col} cannot coerce to {expected}: {e}")
    return {"ok": len(errors) == 0, "errors": errors}


def apply_schema(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    df2 = df.copy()
    for col, expected in schema.items():
        if col not in df2.columns:
            # continue
            try:
                if expected == "float":
                    df2[col] = pd.to_numeric(df2[col], errors="coerce").astype(float)
                elif expected == "int":
                    df2[col] = pd.to_numeric(df2[col], errors="coerce").astype("Int64")
                elif expected == "datetime":
                    df2[col] = pd.to_datetime(df2[col], errors="coerce")
                elif expected == "string":
                    df2[col] = df2[col].astype(str)
                elif expected == "category":
                    df2[col] = df2[col].astype("category")
            except Exception:
                pass
    return df2


def make_changelog_entry(action_type: str, details: dict) -> dict:
    from datetime import datetime

    return {
        "time": datetime.utcnow().isoformat() + "Z",
        "action": action_type,
        "details": details,
    }


def read_csv_chunks(file, chunksize=200_000, max_rows=1_000_000):
    it = pd.read_csv(file, chunksize=chunksize)
    parts = []
    rows = 0
    for chunk in it:
        parts.append(chunk)
        rows += len(chunk)
        if rows >= max_rows:
            break
    return pd.concat(parts, ignore_index=True)
