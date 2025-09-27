# utils.py (corrected)
import pandas as pd
import os
import pickle
import gzip
import tempfile
from typing import Dict, Any
from datetime import datetime, timezone


def estimate_df_size_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum()) / (1024 * 1024)


def save_snapshot_to_tempfile(
    df: pd.DataFrame, compress: bool = True
) -> Dict[str, Any]:
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".pkl.gz" if compress else ".pkl"
    )
    tmp_path = tmp.name
    tmp.close()
    if compress:
        with gzip.open(tmp_path, "wb") as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(tmp_path, "wb") as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "path": tmp_path,
        "size_mb": os.path.getsize(tmp_path) / (1024 * 1024),
        "nrows": len(df),
        "ncols": len(df.columns),
    }


def load_snapshot_from_tempfile(meta: Dict[str, Any]) -> pd.DataFrame:
    path = meta.get("path")
    if path is None or not os.path.exists(path):
        return pd.DataFrame()
    if path.endswith(".gz") or path.endswith(".pkl.gz"):
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    else:
        with open(path, "rb") as f:
            return pickle.load(f)


def remove_snapshot_file(meta: Dict[str, Any]) -> None:
    path = meta.get("path")
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


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
            continue
        try:
            if expected == "float":
                coerced = pd.to_numeric(df2[col], errors="coerce")
                df2[col] = coerced.astype("Float64")
            elif expected == "int":
                coerced = pd.to_numeric(df2[col], errors="coerce")
                if coerced.isna().any():
                    df2[col] = coerced.astype("Int64")
                else:
                    df2[col] = coerced.astype("int64")
            elif expected == "datetime":
                df2[col] = pd.to_datetime(df2[col], errors="coerce")
            elif expected == "string":
                df2[col] = df2[col].astype("string")
            elif expected == "category":
                df2[col] = df2[col].astype("category")
        except Exception:
            # leave unchanged on failure
            pass
    return df2


def make_changelog_entry(action_type: str, details: dict) -> dict:
    return {
        "time": datetime.now(timezone.utc).isoformat(),
        "action": action_type,
        "details": details,
    }


def read_csv_chunks(file, chunksize=200_000, max_rows=1_000_000):
    parts = []
    rows = 0
    # pd.read_csv supports file-like objects and paths
    for chunk in pd.read_csv(file, chunksize=chunksize):
        parts.append(chunk)
        rows += len(chunk)
        if rows >= max_rows:
            break
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def push_history(
    session_state: dict,
    df: pd.DataFrame,
    mem_threshold_mb: float = 20.0,
    max_total_mb: float = 200.0,
):
    if "history" not in session_state:
        session_state["history"] = []
    size_mb = estimate_df_size_mb(df)
    if size_mb <= mem_threshold_mb:
        snap = {"type": "mem", "df": df.copy(), "size_mb": size_mb}
    else:
        meta = save_snapshot_to_tempfile(df)
        snap = {"type": "disk", "meta": meta, "size_mb": meta["size_mb"]}
    session_state["history"].append(snap)
    total = sum([s.get("size_mb", 0) for s in session_state["history"]])
    while total > max_total_mb and session_state["history"]:
        old = session_state["history"].pop(0)
        if old["type"] == "disk":
            remove_snapshot_file(old["meta"])
        total = sum([s.get("size_mb", 0) for s in session_state["history"]])


def pop_history(session_state: dict):
    if "history" not in session_state or not session_state["history"]:
        return None
    last = session_state["history"].pop()
    if last["type"] == "mem":
        return last["df"]
    else:
        df = load_snapshot_from_tempfile(last["meta"])
        remove_snapshot_file(last["meta"])
        return df


def validate_schema_with_report(df: pd.DataFrame, schema: dict, sample_n: int = 200):
    reports = {}
    errors = []
    for col, expected in schema.items():
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
            continue

        series = df[col].dropna()
        # choose a sample (or the whole thing if small)
        sample = (
            series.iloc[:sample_n]
            if len(series) <= sample_n
            else series.sample(n=sample_n, random_state=0)
        )

        # If no non-null values in sample, treat as "no evidence of failure" -> success_ratio 1.0
        if len(sample) == 0:
            success_ratio = 1.0
            failing = []
        else:
            if expected in ("int", "float"):
                coerced = pd.to_numeric(sample, errors="coerce")
                success_ratio = coerced.notna().mean()
                failing = sample[coerced.isna()].head(5).tolist()
            elif expected == "datetime":
                coerced = pd.to_datetime(sample, errors="coerce")
                success_ratio = coerced.notna().mean()
                failing = sample[coerced.isna()].head(5).tolist()
            else:
                success_ratio = 1.0
                failing = []

        reports[col] = {
            "expected": expected,
            "success_ratio": float(success_ratio),
            "examples_failing": failing,
        }
    ok = (
        all(r.get("success_ratio", 1.0) >= 0.9 for r in reports.values()) and not errors
    )
    return {"ok": ok, "column_reports": reports, "errors": errors}
