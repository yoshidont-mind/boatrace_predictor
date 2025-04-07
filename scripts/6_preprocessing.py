import pandas as pd
import numpy as np
import time
import pytz
from datetime import datetime
import sys

# Deviation score calculation function (comparison within the race)
def calculate_deviation_score(series):
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series([50] * len(series), index=series.index)  # Fix to 50 if all are the same
    return 10 * (series - mean) / std + 50

# Preprocessing function
def preprocess_boatrace_dataframe(df):
    print(f"Starting preprocessing. Data size: {df.shape[0]} rows × {df.shape[1]} columns")
    start_time = time.time()
    
    # Assume all necessary columns are present
    print("Step 1: Starting numeric conversion...")
    # Numeric conversion (percentages, etc.)
    rate_columns = ["全国勝率", "全国2連率", "当地勝率", "当地2連率", "ボート2連率"]  # National Win Rate, National 2-Connection Rate, Local Win Rate, Local 2-Connection Rate, Boat 2-Connection Rate
    for col in rate_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"  - Numeric conversion of rate columns ({', '.join(rate_columns)}) completed")

    # If "単勝オッズ" exists, convert to numeric
    if "単勝オッズ" in df.columns:  # Win Odds
        df["単勝オッズ"] = pd.to_numeric(df["単勝オッズ"], errors="coerce")
        print("  - Numeric conversion of '単勝オッズ' column completed")

    print("Step 2: Starting missing value imputation...")
    # Missing value imputation (basically using the median)
    df["年齢"].fillna(df["年齢"].median(), inplace=True)  # Age
    df["体重"].fillna(df["体重"].median(), inplace=True)  # Weight
    df["全国勝率"].fillna(df["全国勝率"].median(), inplace=True)  # National Win Rate
    df["全国2連率"].fillna(df["全国2連率"].median(), inplace=True)  # National 2-Connection Rate
    df["当地勝率"].fillna(df["当地勝率"].median(), inplace=True)  # Local Win Rate
    df["当地2連率"].fillna(df["当地2連率"].median(), inplace=True)  # Local 2-Connection Rate
    df["ボート2連率"].fillna(df["ボート2連率"].median(), inplace=True)  # Boat 2-Connection Rate
    print("  - Missing value imputation completed")

    print("Step 3: Starting outlier processing...")
    # Motor 2連率: Clip outliers
    df["モーター2連率"] = df["モーター2連率"].clip(0, 100)  # Motor 2-Connection Rate
    df["展示タイム"] = df["展示タイム"].clip(6.3, 7.2)  # Exhibition Time
    print("  - Outlier clipping completed")

    print("Step 4: Starting date processing...")
    # Date processing
    df["日付"] = pd.to_datetime(df["日付"], errors="coerce")  # Date
    df["月"] = df["日付"].dt.month  # Month
    df["曜日"] = df["日付"].dt.weekday  # Weekday
    print("  - Extraction of month and weekday from date completed")
    print("Date data type: ", df["日付"].dtype)

    print("Step 5: Starting categorical variable encoding...")
    # Encoding of categorical columns (Label Encoding is OK)
    from sklearn.preprocessing import LabelEncoder
    category_cols = ["支部", "級別", "天候", "風向"]  # Branch, Class, Weather, Wind Direction
    for col in category_cols:
        df[col] = df[col].astype(str).fillna("不明")  # Unknown
        df[col] = LabelEncoder().fit_transform(df[col])
    print(f"  - Encoding of categorical columns ({', '.join(category_cols)}) completed")

    print("Step 6: Starting creation of target variable...")
    # Target variable (win prediction)
    df["is_win"] = df["着"].apply(lambda x: 1 if x == 1 else 0)  # Finish Position
    print("  - Creation of target variable (is_win) completed")

    print("Step 7: Starting deviation score processing...")
    # Columns to be deviation scored (features compared within the race)
    score_cols = ["全国勝率", "全国2連率", "当地勝率", "当地2連率", "モーター2連率", "ボート2連率", "展示タイム"]  # National Win Rate, National 2-Connection Rate, Local Win Rate, Local 2-Connection Rate, Motor 2-Connection Rate, Boat 2-Connection Rate, Exhibition Time

    # Deviation scoring by race
    for col in score_cols:
        new_col = f"{col}_dev"
        df[new_col] = df.groupby("レースID")[col].transform(calculate_deviation_score)  # Race ID
        print(f"  - Deviation scoring of {col} completed → {new_col}")

    print("Step 8: Starting removal of unnecessary columns...")
    # Remove unused columns (e.g., player name, string IDs, etc.)
    drop_cols = ["選手名"]  # Player Name
    df.drop(columns=drop_cols, inplace=True, errors="ignore")
    print(f"  - Removal of unnecessary columns ({', '.join(drop_cols)}) completed")

    elapsed_time = time.time() - start_time
    print(f"Preprocessing completed. Processing time: {elapsed_time:.2f} seconds")
    print(f"Data size after processing: {df.shape[0]} rows × {df.shape[1]} columns")
    
    return df

print("Executing preprocessing script for boat race prediction...")

# Get the data file path from command line arguments
if len(sys.argv) < 2:
    print("Error: Please specify the data file path.")
    sys.exit(1)

data_file_path = sys.argv[1]

# Load data and execute preprocessing
print("Loading data...")
df = pd.read_csv(data_file_path)
print(f"Data loading completed. Size: {df.shape[0]} rows × {df.shape[1]} columns")

print("Executing preprocessing...")
df_preprocessed = preprocess_boatrace_dataframe(df)

# Display the head of the processed data
print("Sample of processed data:")
print(df_preprocessed.head())

# Save the processed data to a CSV file
print("Saving preprocessed data to CSV file...")
# Get the latest date included in the data
latest_date = df_preprocessed["日付"].max().strftime('%Y%m%d')  # Date

# Set the output file name
output_path = f"/Users/yoshidont_mind/Library/Mobile Documents/com~apple~CloudDocs/Desktop/CST term4/COMP 4949 Big Data Analytics Methods/boat_race_prediction/data/preprocessed/preprocessed_{latest_date}.csv"

df_preprocessed.to_csv(output_path, index=False)
print(f"Latest date: {latest_date}")
print(f"Save completed: {output_path}")
print("All processing completed successfully!")
