import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Deviation score calculation function

def calculate_deviation_score(series):
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series([50] * len(series), index=series.index)
    return 10 * (series - mean) / std + 50

# Preprocessing function

def preprocess_boatrace_dataframe(df):
    # Columns to convert to numeric
    rate_cols = ["全国勝率", "全国2連率", "当地勝率", "当地2連率", "モーター2連率", "ボート2連率"]  # National Win Rate, National 2-Connection Rate, Local Win Rate, Local 2-Connection Rate, Motor 2-Connection Rate, Boat 2-Connection Rate
    for col in rate_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert odds if present
    if "単勝オッズ" in df.columns:  # Single Win Odds
        df["単勝オッズ"] = pd.to_numeric(df["単勝オッズ"], errors="coerce")

    # Fill missing values
    df["年齢"] = df["年齢"].fillna(df["年齢"].median())  # Age
    df["体重"] = df["体重"].fillna(df["体重"].median())  # Weight
    for col in rate_cols:
        df[col] = df[col].fillna(df[col].median())

    # Convert to numeric type (展示タイム)
    if "展示タイム" in df.columns:  # Exhibition Time
        df["展示タイム"] = pd.to_numeric(df["展示タイム"], errors="coerce")  # Convert strings to NaN

    # Clip outliers
    df["モーター2連率"] = df["モーター2連率"].clip(0, 100)  # Motor 2-Connection Rate
    if "展示タイム" in df.columns:  # Exhibition Time
        df["展示タイム"] = df["展示タイム"].clip(6.3, 7.2)

    # Extract month and weekday if present
    if "日付" in df.columns:  # Date
        df["日付"] = pd.to_datetime(df["日付"], errors="coerce")
        df["月"] = df["日付"].dt.month  # Month
        df["曜日"] = df["日付"].dt.weekday  # Weekday

    # Encode categorical columns
    cat_cols = ["支部", "級別"]  # Branch, Class
    for col in cat_cols:
        df[col] = df[col].astype(str).fillna("不明")  # Unknown
        df[col] = LabelEncoder().fit_transform(df[col])

    # Dummy processing for venue (set to 0 if not present)
    if "会場" not in df.columns:  # Venue
        df["会場"] = 0
    else:
        df["会場"] = LabelEncoder().fit_transform(df["会場"].astype(str))

    # Deviation score calculation target
    dev_cols = ["全国勝率", "全国2連率", "当地勝率", "当地2連率", "モーター2連率", "ボート2連率", "展示タイム"]  # National Win Rate, National 2-Connection Rate, Local Win Rate, Local 2-Connection Rate, Motor 2-Connection Rate, Boat 2-Connection Rate, Exhibition Time
    for col in dev_cols:
        df[f"{col}_dev"] = df.groupby("レースID")[col].transform(calculate_deviation_score)  # Race ID

    return df
