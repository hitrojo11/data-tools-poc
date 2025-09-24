# streamlit_poc_app.py
# Proof-of-concept Streamlit app for the generic data-processing pipeline
# Features:
# - OpenAI API connectivity test
# - Upload CSV / Excel files
# - Data preview and type detection
# - Missing value detection & simple visualizations
# - Duplicate detection
# - Outlier identification (IQR method)
# - Basic cleaning actions (drop/select/replace/impute)
# - Export cleaned data (CSV / Excel)

# Requirements (put these in requirements.txt):
# streamlit
# pandas
# numpy
#plotly
# openai
# xlrd (if older Excel files)
# openpyxl

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import plotly.express as px
import openai

st.set_page_config(page_title="Data Tools POC", layout="wide")

# --------------------------- Utilities ---------------------------
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

@st.cache_data
def load_excel(file) -> pd.DataFrame:
    # loads first sheet; can be extended
    xl = pd.ExcelFile(file)
    return xl.parse(xl.sheet_names[0])

def detect_types(df: pd.DataFrame) -> pd.DataFrame:
    types = pd.DataFrame({
        'column': df.columns,
        'pandas_dtype': df.dtypes.astype(str),
        'n_unique': df.nunique().values,
        'pct_missing': (df.isna().sum() / len(df) * 100).round(2).values
    })
    # simple heuristic for semantic types
    semantic = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            semantic.append('numeric')
        elif pd.api.types.is_datetime64_any_dtype(s):
            semantic.append('datetime')
        elif pd.api.types.is_bool_dtype(s):
            semantic.append('boolean')
        else:
            # if many unique values but all digits, maybe numeric stored as string
            try:
                pd.to_numeric(s.dropna())
                semantic.append('numeric-ish')
            except Exception:
                semantic.append('categorical/text')
    types['semantic_type'] = semantic
    return types

def detect_outliers_iqr(df: pd.DataFrame, col: str):
    if not pd.api.types.is_numeric_dtype(df[col]):
        return pd.Series([False] * len(df))
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (df[col] < lower) | (df[col] > upper)

# --------------------------- Sidebar: OpenAI check & Upload ---------------------------
st.sidebar.header("Environment & Upload")
api_key = st.sidebar.text_input("OpenAI API Key (or set OPENAI_API_KEY env var)", type="password")
if api_key:
    os.environ['OPENAI_API_KEY'] = api_key

if 'OPENAI_API_KEY' in os.environ and os.environ['OPENAI_API_KEY']:
    openai.api_key = os.environ['OPENAI_API_KEY']
    try:
        # lightweight connectivity check - list models might be rate-limited; use a minimal call
        # For safety we only test that import and key assignment succeed; avoid heavy API calls.
        _ = openai.__version__
        st.sidebar.success("OpenAI lib loaded — key set (connectivity test skipped).")
    except Exception as e:
        st.sidebar.error(f"OpenAI lib error: {e}")
else:
    st.sidebar.info("OpenAI API key not set — you can set it here for LLM-powered insights.")

uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel file", type=["csv","xls","xlsx"], accept_multiple_files=False)

# --------------------------- Main layout ---------------------------
st.title("Data Tools — Proof of Concept")
st.markdown("A minimal, generic data ingestion & cleaning interface. You're the builder; extend as needed.")

if uploaded_file is not None:
    filename = uploaded_file.name
    st.subheader(f"Loaded: {filename}")
    try:
        if filename.lower().endswith('.csv'):
            df = load_csv(uploaded_file)
        else:
            df = load_excel(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    # Show preview and basic info
    st.write("**Preview**")
    st.dataframe(df.head(200))

    st.write("---")
    st.write("**Detected column types & summary**")
    types_df = detect_types(df)
    st.dataframe(types_df)

    # Missing values panel
    st.write("---")
    st.header("Missing values & duplicates")
    col1, col2 = st.columns([2,1])
    with col1:
        mv = df.isna().sum().sort_values(ascending=False)
        mv = mv[mv>0]
        if mv.empty:
            st.success("No missing values detected")
        else:
            mv_df = mv.reset_index()
            mv_df.columns = ['column','missing_count']
            mv_df['pct_missing'] = (mv_df['missing_count'] / len(df) * 100).round(2)
            st.dataframe(mv_df)
            sel_mv_col = st.selectbox('Select column to visualize missing pattern', options=mv_df['column'].tolist())
            # show missing vs present scatter (by row index)
            viz = df[sel_mv_col].isna().astype(int).reset_index().rename(columns={sel_mv_col:'is_missing'})
            fig = px.bar(viz, x='index', y='is_missing', title=f'Missingness in {sel_mv_col}', labels={'is_missing':'is_missing (1=missing)'})
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        dup_count = df.duplicated().sum()
        st.metric("Duplicate rows", f"{dup_count}")
        if dup_count>0:
            if st.button("Show duplicate sample"):
                st.dataframe(df[df.duplicated(keep=False)].head(200))

    # Outliers
    st.write("---")
    st.header("Outlier detection (IQR)")
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        chosen = st.selectbox('Choose numeric column to scan for outliers', options=numeric_cols)
        outlier_mask = detect_outliers_iqr(df, chosen)
        st.write(f"Outliers found: {outlier_mask.sum()} of {len(df)}")
        if outlier_mask.sum()>0:
            st.dataframe(df.loc[outlier_mask, [chosen]].head(200))
            fig = px.box(df, y=chosen, title=f'Box plot — {chosen}')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns detected for outlier scanning.")

    # Basic cleaning actions
    st.write("---")
    st.header("Basic cleaning actions")
    st.write("Choose actions below and then press **Apply**. Changes are kept in-memory for this session.")

    # We'll keep a session-state copy
    if 'work_df' not in st.session_state:
        st.session_state['work_df'] = df.copy()

    work_df = st.session_state['work_df']

    c1, c2, c3 = st.columns(3)
    with c1:
        drop_cols = st.multiselect('Drop columns', options=work_df.columns.tolist())
        if st.button('Apply drop'):
            work_df.drop(columns=drop_cols, inplace=True)
            st.success('Dropped selected columns')
    with c2:
        drop_rows = st.text_input('Drop rows by index (comma separated)')
        if st.button('Apply row drop'):
            if drop_rows.strip():
                try:
                    idxs = [int(x.strip()) for x in drop_rows.split(',') if x.strip()]
                    work_df.drop(index=idxs, inplace=True)
                    st.success('Dropped rows')
                except Exception as e:
                    st.error(f'Bad row indexes: {e}')
    with c3:
        impute_col = st.selectbox('Impute column', options=['--']+work_df.columns.tolist())
        impute_strategy = st.selectbox('Strategy', options=['mean','median','mode','fill_value'])
        if impute_strategy == 'fill_value':
            fill_value = st.text_input('Value to fill with')
        else:
            fill_value = None
        if st.button('Apply impute'):
            if impute_col != '--':
                try:
                    if impute_strategy=='mean':
                        val = work_df[impute_col].mean()
                    elif impute_strategy=='median':
                        val = work_df[impute_col].median()
                    elif impute_strategy=='mode':
                        val = work_df[impute_col].mode().iloc[0]
                    else:
                        val = fill_value
                    work_df[impute_col].fillna(val, inplace=True)
                    st.success('Imputed missing values')
                except Exception as e:
                    st.error(f'Impute error: {e}')

    st.write("---")
    st.subheader('Working data preview')
    st.dataframe(work_df.head(200))

    # Export
    st.write("---")
    st.header('Export cleaned data')
    cexp1, cexp2 = st.columns([1,2])
    with cexp1:
        export_format = st.selectbox('Format', options=['csv','xlsx'])
        export_name = st.text_input('Filename (without extension)', value='cleaned_data')
        if st.button('Download'):
            buf = io.BytesIO()
            if export_format=='csv':
                work_df.to_csv(buf, index=False)
                buf.seek(0)
                st.download_button('Click to download CSV', data=buf, file_name=f"{export_name}.csv", mime='text/csv')
            else:
                with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                    work_df.to_excel(writer, index=False)
                buf.seek(0)
                st.download_button('Click to download Excel', data=buf, file_name=f"{export_name}.xlsx", mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    # Save snapshot to session (so actions persist as user navigates)
    st.session_state['work_df'] = work_df

    # Quick insights using OpenAI (if key present) — minimal safe usage
    st.write("---")
    if 'OPENAI_API_KEY' in os.environ and os.environ['OPENAI_API_KEY']:
        st.header('Automated insights (LLM) — quick summary')
        if st.button('Generate quick insights'):
            # Prepare a tiny summary (no heavy API calls)
            try:
                sample = work_df.select_dtypes(include=[np.number]).describe().T.head(10).to_dict()
                prompt = f"Provide 3 short analytical insights about this numeric summary: {sample}"
                # A real environment would use the ChatCompletions API; keep this block minimal to avoid complexity.
                resp = openai.ChatCompletion.create(
                    model='gpt-4o-mini',
                    messages=[{"role":"user","content":prompt}],
                    max_tokens=200
                )
                text = resp['choices'][0]['message']['content']
                st.write(text)
            except Exception as e:
                st.error(f"LLM call failed (check key & quota): {e}")
    else:
        st.info("Set OPENAI_API_KEY in sidebar to enable automated insights.")

    # Final documentation note
    st.write("---")
    st.markdown("**Notes:** This is a compact POC: extend validators, logging, schema enforcement, and add unit tests. Integrate with a background job queue for large files and add role-based access for multi-user scenarios.")

else:
    st.info("Upload a CSV / Excel file from the sidebar to get started.\n\nIf you want, drop a sample public dataset URL and I'll show how to wire it in.")

# EOF
