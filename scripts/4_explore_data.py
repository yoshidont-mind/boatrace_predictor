#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ボートレースデータの探索的分析を行うスクリプト。
各列のデータ型、欠損値の数、統計情報を表形式で出力します。

使用方法:
    python scripts/4_explore_data.py [入力ファイルパス]

例:
    python scripts/4_explore_data.py data/merged/boat_race_extracted.csv
    python scripts/4_explore_data.py data/merged/boat_race_merged_data.csv
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


def analyze_data(input_file):
    """
    CSVファイルを読み込み、探索的データ分析を行います。
    
    Args:
        input_file (str): 分析するCSVファイルのパス
        
    Returns:
        bool: 処理が成功した場合はTrue、それ以外はFalse
    """
    try:
        # ファイルの存在確認
        if not os.path.exists(input_file):
            print(f"エラー: 入力ファイル '{input_file}' が存在しません。")
            return False
            
        # CSVファイルの読み込み
        print(f"ファイル '{input_file}' を読み込んでいます...")
        df = pd.read_csv(input_file)
        
        # 基本情報の表示
        print("\n===== データの基本情報 =====")
        print(f"行数: {df.shape[0]}")
        print(f"列数: {df.shape[1]}")
        print(f"メモリ使用量: {df.memory_usage().sum() / 1024**2:.2f} MB")
        
        # 列名の一覧表示
        print("\n===== 列名の一覧 =====")
        for i, col in enumerate(df.columns, 1):
            print(f"{i:2d}. {col}")
        
        # データ型の分析
        print("\n===== 各列のデータ型 =====")
        dtype_df = pd.DataFrame({
            '列名': df.columns,
            'データ型': df.dtypes.astype(str),
            'メモリ使用量(KB)': df.memory_usage(deep=True)[1:] / 1024
        })
        print(tabulate(dtype_df, headers='keys', tablefmt='grid', showindex=False))
        
        # 欠損値の分析
        print("\n===== 欠損値の分析 =====")
        null_df = pd.DataFrame({
            '列名': df.columns,
            '欠損値数': df.isnull().sum(),
            '欠損値の割合(%)': df.isnull().sum() / len(df) * 100
        }).sort_values('欠損値数', ascending=False)
        print(tabulate(null_df, headers='keys', tablefmt='grid', showindex=False))
        
        # 数値列の統計情報
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            print("\n===== 数値列の統計情報 =====")
            stats_df = pd.DataFrame({
                '列名': numeric_cols,
                '最小値': [df[col].min() for col in numeric_cols],
                '第1四分位数': [df[col].quantile(0.25) for col in numeric_cols],
                '中央値': [df[col].median() for col in numeric_cols],
                '平均値': [df[col].mean() for col in numeric_cols],
                '第3四分位数': [df[col].quantile(0.75) for col in numeric_cols],
                '最大値': [df[col].max() for col in numeric_cols],
                '標準偏差': [df[col].std() for col in numeric_cols],
                '分散': [df[col].var() for col in numeric_cols]
            })
            print(tabulate(stats_df, headers='keys', tablefmt='grid', showindex=False, floatfmt='.3f'))
            
            # 外れ値の分析
            print("\n===== 外れ値の分析（数値列のみ） =====")
            outliers_df = pd.DataFrame(columns=['列名', '外れ値の数', '外れ値の割合(%)', '外れ値の基準'])
            
            for col in numeric_cols:
                if df[col].nunique() > 1:  # 一意な値が1つだけの列は除外
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                    outlier_count = len(outliers)
                    outlier_percent = outlier_count / len(df) * 100
                    
                    outliers_df = pd.concat([outliers_df, pd.DataFrame({
                        '列名': [col],
                        '外れ値の数': [outlier_count],
                        '外れ値の割合(%)': [outlier_percent],
                        '外れ値の基準': [f"< {lower_bound:.3f} または > {upper_bound:.3f}"]
                    })], ignore_index=True)
            
            print(tabulate(outliers_df, headers='keys', tablefmt='grid', showindex=False))
        
        # カテゴリ列の分析（文字列列）
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        if cat_cols:
            print("\n===== カテゴリカル列の分析 =====")
            cat_df = pd.DataFrame({
                '列名': cat_cols,
                '一意な値の数': [df[col].nunique() for col in cat_cols],
                '最頻値': [df[col].mode()[0] if not df[col].mode().empty else 'なし' for col in cat_cols],
                '最頻値の出現回数': [df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0 for col in cat_cols],
                '最頻値の割合(%)': [df[col].value_counts().iloc[0] / len(df) * 100 if not df[col].value_counts().empty else 0 for col in cat_cols]
            })
            print(tabulate(cat_df, headers='keys', tablefmt='grid', showindex=False))
            
            # 各カテゴリカル列の上位値の表示
            for col in cat_cols:
                if df[col].nunique() <= 20:  # 一意な値が20以下の場合のみ表示
                    print(f"\n列 '{col}' の値の分布:")
                    value_counts = df[col].value_counts()
                    value_percent = value_counts / len(df) * 100
                    value_df = pd.DataFrame({
                        '値': value_counts.index,
                        '出現回数': value_counts.values,
                        '割合(%)': value_percent.values
                    })
                    print(tabulate(value_df.head(10), headers='keys', tablefmt='grid', showindex=False))
                else:
                    print(f"\n列 '{col}' の上位10件:")
                    value_counts = df[col].value_counts()
                    value_percent = value_counts / len(df) * 100
                    value_df = pd.DataFrame({
                        '値': value_counts.index,
                        '出現回数': value_counts.values,
                        '割合(%)': value_percent.values
                    })
                    print(tabulate(value_df.head(10), headers='keys', tablefmt='grid', showindex=False))
        
        # 日付列の分析
        try:
            if 'date' in df.columns:
                print("\n===== 日付列の分析 =====")
                df['date'] = pd.to_datetime(df['date'])
                date_range = df['date'].max() - df['date'].min()
                print(f"日付の範囲: {df['date'].min()} から {df['date'].max()} ({date_range.days} 日間)")
                print(f"一意な日付の数: {df['date'].nunique()}")
                
                # 月ごとのデータ数
                monthly_counts = df['date'].dt.to_period('M').value_counts().sort_index()
                print("\n月ごとのデータ数:")
                for period, count in monthly_counts.items():
                    print(f"{period}: {count} 件")
        except:
            print("日付列の分析中にエラーが発生しました。")
        
        print("\n分析が完了しました。")
        return True
        
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    メイン関数。コマンドライン引数を処理し、データ分析を実行します。
    """
    # コマンドライン引数の処理
    if len(sys.argv) < 2:
        print("使用方法: python scripts/4_explore_data.py [入力ファイルパス]")
        print("例: python scripts/4_explore_data.py data/merged/boat_race_extracted.csv")
        return False
    
    input_file = sys.argv[1]
    
    # データ分析の実行
    success = analyze_data(input_file)
    return success


if __name__ == "__main__":
    main() 