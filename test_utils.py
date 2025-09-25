import pandas as pd
from utils import detect_types_fast as detect_types, detect_outliers_iqr

def test_detect_types_numeric():
    df = pd.DataFrame({'a':[1,2,3], 'b':['x','y','z']})
    meta = detect_types(df)
    assert 'a' in meta['column'].values
    assert 'numeric' in meta['semantic_type'].values

def test_outliers_iqr():
    df = pd.DataFrame({'v':[1,2,3,1000]})
    mask = detect_outliers_iqr(df, 'v')
    assert mask.sum() >= 1
