import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 偏差値計算関数

def calculate_deviation_score(series):
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series([50] * len(series), index=series.index)
    return 10 * (series - mean) / std + 50

# 前処理関数

def preprocess_boatrace_dataframe(df):
    # 数値変換対象
    rate_cols = ["全国勝率", "全国2連率", "当地勝率", "当地2連率", "モーター2連率", "ボート2連率"]
    for col in rate_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # オッズ変換（存在する場合）
    if "単勝オッズ" in df.columns:
        df["単勝オッズ"] = pd.to_numeric(df["単勝オッズ"], errors="coerce")

    # 欠損補完
    df["年齢"] = df["年齢"].fillna(df["年齢"].median())
    df["体重"] = df["体重"].fillna(df["体重"].median())
    for col in rate_cols:
        df[col] = df[col].fillna(df[col].median())

    # 数値型への変換（展示タイム）
    if "展示タイム" in df.columns:
        df["展示タイム"] = pd.to_numeric(df["展示タイム"], errors="coerce")  # 文字列は欠損値に変換

    # 異常値クリップ
    df["モーター2連率"] = df["モーター2連率"].clip(0, 100)
    if "展示タイム" in df.columns:
        df["展示タイム"] = df["展示タイム"].clip(6.3, 7.2)

    # 月・曜日抽出（存在すれば）
    if "日付" in df.columns:
        df["日付"] = pd.to_datetime(df["日付"], errors="coerce")
        df["月"] = df["日付"].dt.month
        df["曜日"] = df["日付"].dt.weekday

    # カテゴリ列エンコーディング
    cat_cols = ["支部", "級別"]
    for col in cat_cols:
        df[col] = df[col].astype(str).fillna("不明")
        df[col] = LabelEncoder().fit_transform(df[col])

    # 会場のダミー処理（存在しない場合は仮で0）
    if "会場" not in df.columns:
        df["会場"] = 0
    else:
        df["会場"] = LabelEncoder().fit_transform(df["会場"].astype(str))

    # 偏差値化対象
    dev_cols = ["全国勝率", "全国2連率", "当地勝率", "当地2連率", "モーター2連率", "ボート2連率", "展示タイム"]
    for col in dev_cols:
        df[f"{col}_dev"] = df.groupby("レースID")[col].transform(calculate_deviation_score)

    return df
