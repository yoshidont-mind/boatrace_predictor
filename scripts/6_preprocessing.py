import pandas as pd
import numpy as np
import time
import pytz
from datetime import datetime
import sys

# 偏差値計算関数（レース内での比較）
def calculate_deviation_score(series):
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series([50] * len(series), index=series.index)  # 全員同じ場合は50に固定
    return 10 * (series - mean) / std + 50

# 前処理関数
def preprocess_boatrace_dataframe(df):
    print(f"前処理を開始します。データサイズ: {df.shape[0]}行 × {df.shape[1]}列")
    start_time = time.time()
    
    # 必要な列がすべて揃っていることを前提とする
    print("ステップ1: 数値型変換を開始します...")
    # 数値型変換（パーセンテージなど）
    rate_columns = ["全国勝率", "全国2連率", "当地勝率", "当地2連率", "ボート2連率"]
    for col in rate_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"  - レート列({', '.join(rate_columns)})の数値変換が完了しました")

    # 「単勝オッズ」が存在する場合、数値型に変換
    if "単勝オッズ" in df.columns:
        df["単勝オッズ"] = pd.to_numeric(df["単勝オッズ"], errors="coerce")
        print("  - 「単勝オッズ」列の数値変換が完了しました")

    print("ステップ2: 欠損値補完を開始します...")
    # 欠損補完（基本的には中央値で対応）
    df["年齢"].fillna(df["年齢"].median(), inplace=True)
    df["体重"].fillna(df["体重"].median(), inplace=True)
    df["全国勝率"].fillna(df["全国勝率"].median(), inplace=True)
    df["全国2連率"].fillna(df["全国2連率"].median(), inplace=True)
    df["当地勝率"].fillna(df["当地勝率"].median(), inplace=True)
    df["当地2連率"].fillna(df["当地2連率"].median(), inplace=True)
    df["ボート2連率"].fillna(df["ボート2連率"].median(), inplace=True)
    print("  - 欠損値の補完が完了しました")

    print("ステップ3: 異常値の処理を開始します...")
    # モーター2連率：異常値のクリップ
    df["モーター2連率"] = df["モーター2連率"].clip(0, 100)
    df["展示タイム"] = df["展示タイム"].clip(6.3, 7.2)
    print("  - 異常値のクリップ処理が完了しました")

    print("ステップ4: 日付処理を開始します...")
    # 日付処理
    df["日付"] = pd.to_datetime(df["日付"], errors="coerce")
    df["月"] = df["日付"].dt.month
    df["曜日"] = df["日付"].dt.weekday
    print("  - 日付から月・曜日の抽出が完了しました")
    print("日付のデータ型: ", df["日付"].dtype)

    print("ステップ5: カテゴリカル変数のエンコーディングを開始します...")
    # カテゴリ列のエンコーディング（Label EncodingでOK）
    from sklearn.preprocessing import LabelEncoder
    category_cols = ["支部", "級別", "天候", "風向"]
    for col in category_cols:
        df[col] = df[col].astype(str).fillna("不明")
        df[col] = LabelEncoder().fit_transform(df[col])
    print(f"  - カテゴリ列({', '.join(category_cols)})のエンコーディングが完了しました")

    print("ステップ6: 目的変数の作成を開始します...")
    # 目的変数（単勝＝1着予測）
    df["is_win"] = df["着"].apply(lambda x: 1 if x == 1 else 0)
    print("  - 目的変数(is_win)の作成が完了しました")

    print("ステップ7: 偏差値化処理を開始します...")
    # 偏差値化する列（レース内で比較する特徴量）
    score_cols = ["全国勝率", "全国2連率", "当地勝率", "当地2連率", "モーター2連率", "ボート2連率", "展示タイム"]

    # レース単位で偏差値化
    for col in score_cols:
        new_col = f"{col}_dev"
        df[new_col] = df.groupby("レースID")[col].transform(calculate_deviation_score)
        print(f"  - {col}の偏差値化が完了しました → {new_col}")

    print("ステップ8: 不要な列の削除を開始します...")
    # 使用しない列を削除（例：選手名、文字列系IDなど）
    drop_cols = ["選手名"]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")
    print(f"  - 不要列({', '.join(drop_cols)})の削除が完了しました")

    elapsed_time = time.time() - start_time
    print(f"前処理が完了しました。処理時間: {elapsed_time:.2f}秒")
    print(f"処理後のデータサイズ: {df.shape[0]}行 × {df.shape[1]}列")
    
    return df

print("ボートレース予測のための前処理スクリプトを実行します...")

# コマンドライン引数からデータファイルのパスを取得
if len(sys.argv) < 2:
    print("エラー: データファイルのパスを指定してください。")
    sys.exit(1)

data_file_path = sys.argv[1]

# データ読み込みと前処理の実行
print("データを読み込んでいます...")
df = pd.read_csv(data_file_path)
print(f"データ読み込み完了。サイズ: {df.shape[0]}行 × {df.shape[1]}列")

print("前処理を実行します...")
df_preprocessed = preprocess_boatrace_dataframe(df)

# 処理後のデータの先頭を表示
print("処理後のデータサンプル:")
print(df_preprocessed.head())

# 処理後のデータをCSVファイルとして保存
print("前処理済みデータをCSVファイルに保存しています...")
# データに含まれる最新の日付を取得
latest_date = df_preprocessed["日付"].max().strftime('%Y%m%d')

# 出力ファイル名を設定
output_path = f"/Users/yoshidont_mind/Library/Mobile Documents/com~apple~CloudDocs/Desktop/CST term4/COMP 4949 Big Data Analytics Methods/boat_race_prediction/data/preprocessed/preprocessed_{latest_date}.csv"

df_preprocessed.to_csv(output_path, index=False)
print(f"最新日付: {latest_date}")
print(f"保存完了: {output_path}")
print("全ての処理が正常に完了しました！")
