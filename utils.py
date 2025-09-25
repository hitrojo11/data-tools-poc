# utils.py
import pandas as pd
import numpy as np

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
    coerced = pd.to_numeric(s, errors='coerce')
    return coerced.notna().mean() >= 0.9

def detect_types_fast(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        ser = df[col]
        pandas_dtype = str(ser.dtypes)
        n_unique = ser.nunique(dropna=True)
        pct_missing = round(ser.isna().mean() * 100, 2)
        if pd.api.types.is_numeric_dtype(ser):
            semantic = 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(ser):
            semantic = 'datetime'
        elif pd.api.types.is_bool_dtype(ser):
            semantic = 'boolean'
        else:
            semantic = 'numeric-ish' if is_numeric_ish(ser) else 'categorical/text'
        rows.append({'column': col, 'pandas_dtype': pandas_dtype, 'n_unique': n_unique, 'pct_missing': pct_missing, 'semantic_type': semantic})
    return pd.DataFrame(rows)

def detect_outliers_iqr(df: pd.DataFrame, col: str) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(df[col]):
        return pd.Series([False] * len(df), index=df.index)
    q1 = df[col].quantile(0.25); q3 = df[col].quantile(0.75)
    iqr = q3 - q1; lower = q1 - 1.5 * iqr; upper = q3 + 1.5 * iqr
    return (df[col] < lower) | (df[col] > upper)
