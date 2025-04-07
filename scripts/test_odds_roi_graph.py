import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import random

# グラフ保存ディレクトリの確認
graphs_dir = "graphs"
if not os.path.exists(graphs_dir):
    os.makedirs(graphs_dir)
    print(f"Created directory for saving graphs: {graphs_dir}")

# テスト用データを生成
def generate_test_data(n_samples=1000, strategy_name="Test Strategy"):
    """テスト用のデータを生成する関数"""
    np.random.seed(42)  # 再現性のため
    
    # オッズの範囲をランダムに生成
    odds_actual = np.concatenate([
        np.random.uniform(1.1, 1.5, size=int(n_samples*0.2)),  # ~1.5倍
        np.random.uniform(1.5, 2.0, size=int(n_samples*0.2)),  # 1.5~2倍
        np.random.uniform(2.0, 3.0, size=int(n_samples*0.2)),  # 2~3倍
        np.random.uniform(3.0, 5.0, size=int(n_samples*0.15)),  # 3~5倍
        np.random.uniform(5.0, 10.0, size=int(n_samples*0.1)),  # 5~10倍
        np.random.uniform(10.0, 50.0, size=int(n_samples*0.1)),  # 10~50倍
        np.random.uniform(50.0, 100.0, size=int(n_samples*0.05))  # 50倍~
    ])
    
    # オッズが低いものほど的中率が高くなるようにする
    win_prob = 1.0 / odds_actual * 0.9  # 理論上の確率よりも少し低めに
    is_win = np.random.random(size=len(odds_actual)) < win_prob
    
    # データフレーム作成
    df = pd.DataFrame({
        "odds_actual": odds_actual,
        "is_win": is_win,
        "日付": pd.date_range(start="2025-01-01", periods=len(odds_actual))
    })
    
    return df

# 4つの戦略のテストデータを生成
strategy_results = {
    "Top1 Strategy": generate_test_data(n_samples=800, strategy_name="Top1"),
    "Top3 Strategy": generate_test_data(n_samples=1200, strategy_name="Top3"),
    "Kelly Strategy": generate_test_data(n_samples=600, strategy_name="Kelly"),
    "Threshold Strategy": generate_test_data(n_samples=400, strategy_name="Threshold")
}

# 各戦略の回収率を少し調整して特徴を出す
for i, (strategy, df) in enumerate(strategy_results.items()):
    # 戦略ごとに特徴を持たせる
    if strategy == "Top1 Strategy":
        # 低オッズが少し強い
        df.loc[df["odds_actual"] < 2.0, "is_win"] = np.random.random(size=len(df[df["odds_actual"] < 2.0])) < 0.6
    elif strategy == "Top3 Strategy":
        # 中オッズが少し強い
        df.loc[(df["odds_actual"] >= 2.0) & (df["odds_actual"] < 5.0), "is_win"] = np.random.random(
            size=len(df[(df["odds_actual"] >= 2.0) & (df["odds_actual"] < 5.0)])) < 0.4
    elif strategy == "Kelly Strategy":
        # 全体的に安定している
        df["is_win"] = np.random.random(size=len(df)) < 0.35 / np.sqrt(df["odds_actual"])
    else:  # Threshold Strategy
        # 高オッズが強い
        df.loc[df["odds_actual"] > 10.0, "is_win"] = np.random.random(size=len(df[df["odds_actual"] > 10.0])) < 0.15

def compare_roi_by_odds_range(results_dict):
    """
    オッズ区間ごとの回収率を戦略間で比較するグラフを作成する
    
    Parameters:
    -----------
    results_dict : dict
        戦略名をキー、データフレームを値とする辞書
    """
    # 各戦略のオッズ区間別回収率を保存するための辞書
    roi_by_odds = {}
    
    # 各戦略のデータを処理
    for strategy_name, strategy_df in results_dict.items():
        # 元のデータを変更しないようにコピーを作成
        df = strategy_df.copy()
        
        # オッズ区間カラムを追加
        df["Odds Range"] = pd.cut(df["odds_actual"], 
                                 bins=[0, 1.5, 2, 3, 5, 10, 50, 1000], 
                                 labels=["~1.5", "1.5~2", "2~3", "3~5", "5~10", "10~50", "50+"])
        
        # オッズ区間ごとの回収率を計算
        roi_data = df.groupby("Odds Range").apply(
            lambda x: pd.Series({
                "Bet Count": len(x),
                "Win Count": x["is_win"].sum(),
                "Win Rate": x["is_win"].mean() * 100,
                "Investment": len(x) * 100,
                "Return": (x["is_win"] * x["odds_actual"] * 100).sum(),
                "ROI (%)": (x["is_win"] * x["odds_actual"] * 100).sum() / (len(x) * 100) * 100 if len(x) > 0 else 0
            })
        )
        
        # この戦略のROIデータを保存
        roi_by_odds[strategy_name] = roi_data
    
    # 全戦略のROIデータを統合
    combined_roi = pd.DataFrame()
    for strategy_name, roi_data in roi_by_odds.items():
        if not roi_data.empty:
            temp_df = roi_data["ROI (%)"].reset_index()
            temp_df.columns = ["Odds Range", strategy_name]
            
            if combined_roi.empty:
                combined_roi = temp_df
            else:
                combined_roi = pd.merge(combined_roi, temp_df, on="Odds Range", how="outer")
    
    # NaN値を0で埋める（ベットのないオッズ区間用）
    # Categorical列以外の数値列のみfillnaを適用
    for col in combined_roi.columns:
        if col != "Odds Range":
            combined_roi[col] = combined_roi[col].fillna(0)
    
    # グラフをプロット
    plt.figure(figsize=(12, 8))
    
    # バーの幅を設定
    bar_width = 0.15
    index = np.arange(len(combined_roi["Odds Range"]))
    
    # 各戦略のバーをプロット
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, (strategy_name, color) in enumerate(zip(roi_by_odds.keys(), colors)):
        if strategy_name in combined_roi.columns:
            plt.bar(index + i*bar_width, combined_roi[strategy_name], 
                   bar_width, label=strategy_name, color=color)
    
    # 回収率100%のラインを追加
    plt.axhline(y=100, color='r', linestyle='-', alpha=0.3)
    
    # ラベルとタイトルを追加
    plt.xlabel('Odds Range')
    plt.ylabel('ROI (%)')
    plt.title('Return on Investment by Odds Range Across Strategies')
    plt.xticks(index + bar_width * (len(roi_by_odds) - 1) / 2, combined_roi["Odds Range"])
    plt.legend()
    
    # バーの上に値のラベルを追加
    for i, (strategy_name, color) in enumerate(zip(roi_by_odds.keys(), colors)):
        if strategy_name in combined_roi.columns:
            for j, value in enumerate(combined_roi[strategy_name]):
                if value > 0:  # 正の値のみラベルを表示
                    plt.text(j + i*bar_width, value + 5, f'{value:.0f}%', 
                            ha='center', va='bottom', color=color, fontweight='bold')
    
    plt.tight_layout()
    
    # グラフを保存
    latest_date = df["日付"].max().strftime('%Y%m%d') if "日付" in df.columns else "unknown"
    graph_path = os.path.join("graphs", f"roi_by_odds_range_{latest_date}.png")
    plt.savefig(graph_path, dpi=300)
    print(f"\nROI by Odds Range graph saved: {graph_path}")
    # plt.show()
    
    return combined_roi

# オッズ区間別回収率の比較グラフを作成して保存
compare_roi_by_odds_range(strategy_results)
print("グラフ生成完了！")
