import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
import os
import sys

# ===== 日本語フォント（macOS用） =====
mpl.rcParams['font.family'] = ['Hiragino Sans GB', 'AppleGothic']

# ===== 1. データ読み込み =====
# コマンドライン引数からデータファイルのパスを取得
if len(sys.argv) < 2:
    print("エラー: データファイルのパスを指定してください。")
    sys.exit(1)

data_file_path = sys.argv[1]

df = pd.read_csv(data_file_path)

# 日付カラムをdatetime型に変換
df["日付"] = pd.to_datetime(df["日付"])

# ===== 2. 特徴量・目的変数 =====
feature_cols = [
    "支部", "級別", "艇番", "会場", "風量", "波", "月", "曜日",
    "全国勝率_dev", "全国2連率_dev", "当地勝率_dev", "当地2連率_dev",
    "モーター2連率_dev", "ボート2連率_dev", "展示タイム_dev",
]
X = df[feature_cols]
y = df["is_win"]

# ===== 3. 時系列分割 =====
df = df.sort_values("日付")
# 日付単位で分割（同じ日はtrain/testに跨がないように）
unique_dates = df["日付"].dt.date.unique()
unique_dates.sort()
split_idx = int(len(unique_dates) * 0.8)  # 80%をトレーニングデータに
split_date = unique_dates[split_idx]
print(f"データ分割日: {split_date} （この日より前がトレーニングデータ、この日以降がテストデータ）")

train_base_df = df[df["日付"].dt.date < split_date].copy()
test_all_df = df[df["日付"].dt.date >= split_date].copy()
print(f"トレーニングベースデータ: {len(train_base_df)}行 ({train_base_df['日付'].min()} 〜 {train_base_df['日付'].max()})")
print(f"テスト対象データ: {len(test_all_df)}行 ({test_all_df['日付'].min()} 〜 {test_all_df['日付'].max()})")

# ウォークフォワード方式のパラメータ設定
params = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "random_state": 42
}

# ===== 4 & 5. ウォークフォワード方式によるモデル評価（実運用に近い方法） =====
print("\n===== ウォークフォワード方式による評価 =====")
print("各レース日ごとにその時点までのデータでモデルを構築し、当日のレースのみ予測します")

# テスト期間の各日付を取得
test_dates = sorted(test_all_df["日付"].dt.date.unique())

# 全予測結果を保存するDataFrame
all_predictions = []

for i, test_date in enumerate(test_dates):
    # 進捗表示
    print(f"\n評価中: {test_date} ({i+1}/{len(test_dates)})")
    
    # その日のデータ
    test_df = test_all_df[test_all_df["日付"].dt.date == test_date].copy()
    
    # その日より前のデータ全て（訓練用）
    train_df = df[df["日付"].dt.date < test_date].copy()
    
    print(f"  訓練データ: {len(train_df)}行, テストデータ: {len(test_df)}行")
    
    # 訓練データとテストデータを準備
    X_train = train_df[feature_cols]
    y_train = train_df["is_win"]
    X_test = test_df[feature_cols]
    y_test = test_df["is_win"]
    
    # 訓練データを学習用と検証用に分割（検証用は最新の20%）
    train_rows = len(train_df)
    if train_rows > 1000:  # 十分なデータがある場合のみ分割
        val_size = int(train_rows * 0.2)
        train_idx = train_rows - val_size
        
        X_train_train = X_train.iloc[:train_idx]
        y_train_train = y_train.iloc[:train_idx]
        X_train_val = X_train.iloc[train_idx:]
        y_train_val = y_train.iloc[train_idx:]
        
        # LightGBMデータセット作成
        train_data = lgb.Dataset(X_train_train, label=y_train_train)
        val_data = lgb.Dataset(X_train_val, label=y_train_val)
        
        # モデル学習（早期停止用の検証データを使用）
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=100,
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
        )
    else:
        # データが少ない場合はearly stoppingなしで学習
        train_data = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(
            params,
            train_data,
            num_boost_round=50  # early stoppingがないので少なめに
        )
    
    # 予測
    test_df["pred_proba"] = model.predict(X_test)
    test_df["pred_label"] = (test_df["pred_proba"] > 0.5).astype(int)
    
    # 簡易評価（当日分のみ）
    if len(test_df) > 0:
        daily_accuracy = accuracy_score(test_df["is_win"], test_df["pred_label"])
        print(f"  当日の予測精度: {daily_accuracy:.4f}")
    
    # 結果を保存
    all_predictions.append(test_df)

# 全予測結果を結合
test_all_predictions = pd.concat(all_predictions)

# 結果の評価
y_test = test_all_predictions["is_win"]
y_pred_proba = test_all_predictions["pred_proba"]
y_pred_label = test_all_predictions["pred_label"]

print("\n=== ウォークフォワード方式による総合評価結果 ===")
print("AUC:", round(roc_auc_score(y_test, y_pred_proba), 4))
print("Accuracy:", round(accuracy_score(y_test, y_pred_label), 4))
print(classification_report(y_test, y_pred_label))

# テストデータを更新（ウォークフォワード方式の結果を使用）
test_df = test_all_predictions

# ===== 6. 特徴量重要度 =====
lgb.plot_importance(model, max_num_features=15, figsize=(10, 6))
plt.title("Feature Importance")
# plt.show()

# ===== 7. 期待値計算 =====
# オッズ変動を考慮した補正関数（高オッズほど大きく下げる）
def adjusted_odds(raw_odds, weight=0.2):
    """
    実際の運用時のオッズ変動を考慮した補正関数
    高オッズほど大きく下げる（逆に低オッズは変化少）
    
    Parameters:
    -----------
    raw_odds : float
        元のオッズ値
    weight : float, default=0.2
        補正の強さ（大きいほど補正が強くなる）
    
    Returns:
    --------
    float
        補正後のオッズ値
    """
    return raw_odds * (1 - weight * (1 / raw_odds))

# テストデータの期待値計算（予測確率はウォークフォワード予測の結果を使用）
test_df["odds_actual"] = test_df["単勝オッズ"] / 100
test_df["odds_adjusted"] = test_df["odds_actual"].apply(adjusted_odds)
test_df["expected_return_raw"] = test_df["pred_proba"] * test_df["odds_actual"]  # 従来の計算方法
test_df["expected_return"] = test_df["pred_proba"] * test_df["odds_adjusted"]  # オッズ変動考慮版

# 補正前後の期待値の比較
print("\n=== オッズ変動補正の効果 ===")
odds_comparison = test_df.groupby(pd.cut(test_df["odds_actual"], 
                         bins=[0, 1.5, 2, 3, 5, 10, 50, 1000], 
                         labels=["〜1.5倍", "1.5〜2倍", "2〜3倍", "3〜5倍", "5〜10倍", "10〜50倍", "50倍〜"])).agg({
    "odds_actual": "mean",
    "odds_adjusted": "mean",
    "expected_return_raw": "mean",
    "expected_return": "mean",
    "レースID": "count"
}).rename(columns={"レースID": "件数"})

odds_comparison["補正率"] = (odds_comparison["odds_adjusted"] / odds_comparison["odds_actual"] - 1) * 100
odds_comparison["期待値変化率"] = (odds_comparison["expected_return"] / odds_comparison["expected_return_raw"] - 1) * 100

print(odds_comparison[["件数", "odds_actual", "odds_adjusted", "補正率", "expected_return_raw", "expected_return", "期待値変化率"]])

# ===== 8. 推奨買い目表示 =====
def recommend_bets(df, threshold=1.0):
    return df[df["expected_return"] > threshold].sort_values("expected_return", ascending=False)

print("\n=== 期待値が高いおすすめ買い目 ===")
print(recommend_bets(test_df)[["レースID", "艇番", "pred_proba", "odds_actual", "expected_return"]].head(10))

# ===== 8.5 払戻金分布の可視化 =====
def visualize_returns(df):
    # 払戻金の計算
    df = df.copy()
    df["払戻"] = df["is_win"] * df["odds_actual"] * 100
    
    # 払戻金の分布を可視化
    plt.figure(figsize=(10, 6))
    sns.histplot(df["払戻"], bins=50)
    plt.title("払戻金の分布")
    plt.xlabel("払戻金額（円）")
    plt.ylabel("頻度")
    # plt.show()
    
    # オッズ別の回収率分析
    df["オッズ区間"] = pd.cut(df["odds_actual"], 
                          bins=[0, 1.5, 2, 3, 5, 10, 50, 1000], 
                          labels=["〜1.5倍", "1.5〜2倍", "2〜3倍", "3〜5倍", "5〜10倍", "10〜50倍", "50倍〜"])
    
    odds_roi = df.groupby("オッズ区間").apply(
        lambda x: pd.Series({
            "ベット数": len(x),
            "的中数": x["is_win"].sum(),
            "的中率": x["is_win"].mean() * 100,
            "投資額": len(x) * 100,
            "払戻額": (x["is_win"] * x["odds_actual"] * 100).sum(),
            "回収率": (x["is_win"] * x["odds_actual"] * 100).sum() / (len(x) * 100) * 100
        })
    )
    
    print("\n=== オッズ別回収率 ===")
    print(odds_roi)
    
    # オッズ別回収率の可視化
    plt.figure(figsize=(12, 6))
    sns.barplot(x=odds_roi.index, y=odds_roi["回収率"])
    plt.title("オッズ別の回収率")
    plt.ylabel("回収率（%）")
    plt.axhline(y=100, color='r', linestyle='-', alpha=0.3)  # 100%ライン
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.show()
    
    # 大当たりの影響分析
    threshold = 1000  # 1000円以上を大当たりと定義
    big_wins = df[df["払戻"] >= threshold]
    
    if len(big_wins) > 0:
        print(f"\n=== 大当たり（{threshold}円以上）の影響 ===")
        total_return = df["払戻"].sum()
        big_win_return = big_wins["払戻"].sum()
        total_cost = len(df) * 100
        
        print(f"大当たり回数: {len(big_wins)}回")
        print(f"大当たりによる払戻合計: ¥{big_win_return:,.0f} ({big_win_return/total_return*100:.1f}%の寄与率)")
        
        # 大当たりを除いた場合の回収率
        roi_with_big = total_return / total_cost
        roi_without_big = (total_return - big_win_return) / total_cost
        
        print(f"全体の回収率: {roi_with_big*100:.2f}%")
        print(f"大当たりを除いた回収率: {roi_without_big*100:.2f}%")
        
        # 大当たり一覧
        print("\n=== 大当たり詳細 ===")
        print(big_wins[["レースID", "艇番", "odds_actual", "払戻"]].sort_values("払戻", ascending=False))

# テストデータの払戻金分布を可視化
print("\n=== テストデータの払戻金分析 ===")
visualize_returns(test_df)

# ===== 9. バックテスト：期待値最大の1艇に100円ずつ =====
def simulate_top1(df):
    df = df.copy()
    # オッズ調整後の期待値でソート
    top_bets = df.groupby("レースID").apply(lambda x: x.sort_values("expected_return", ascending=False).head(1)).reset_index(drop=True)
    total_bets = len(top_bets)
    total_cost = 100 * total_bets
    # 実際の払戻計算は調整前のオッズを使用（実際の払戻はオッズ板の値で計算されるため）
    total_return = (top_bets["is_win"] * top_bets["odds_actual"] * 100).sum()
    roi = total_return / total_cost
    print("\n=== バックテスト（Top1艇） ===")
    print(f"ベット数: {total_bets} 回")
    print(f"投資金額: ¥{total_cost:,.0f}")
    print(f"払戻金額: ¥{total_return:,.0f}")
    print(f"回収率 (ROI): {roi * 100:.2f}%")
    
    # Top1の的中率とオッズ分布
    print(f"的中率: {top_bets['is_win'].mean() * 100:.2f}%")
    print(f"平均オッズ: {top_bets['odds_actual'].mean():.2f}倍")
    
    # 払戻金の可視化
    top_bets["払戻"] = top_bets["is_win"] * top_bets["odds_actual"] * 100
    visualize_returns(top_bets)
    
    return roi, top_bets

simulate_top1(test_df)

# ===== 10. 分散投資：Top3艇に100円ずつ =====
def simulate_top3(df):
    df = df.copy()
    # オッズ調整後の期待値でソート
    top3_bets = df.groupby("レースID").apply(lambda x: x.sort_values("expected_return", ascending=False).head(3)).reset_index(drop=True)
    total_bets = len(top3_bets)
    total_cost = 100 * total_bets
    # 実際の払戻計算は調整前のオッズを使用
    total_return = (top3_bets["is_win"] * top3_bets["odds_actual"] * 100).sum()
    roi = total_return / total_cost
    print("\n=== 分散投資（Top3艇） ===")
    print(f"ベット数: {total_bets} 回（各レース3艇）")
    print(f"投資金額: ¥{total_cost:,.0f}")
    print(f"払戻金額: ¥{total_return:,.0f}")
    print(f"回収率 (ROI): {roi * 100:.2f}%")
    
    # Top3の的中率とオッズ分布
    print(f"的中率: {top3_bets['is_win'].mean() * 100:.2f}%")
    print(f"平均オッズ: {top3_bets['odds_actual'].mean():.2f}倍")
    
    # 払戻金の可視化
    top3_bets["払戻"] = top3_bets["is_win"] * top3_bets["odds_actual"] * 100
    visualize_returns(top3_bets)
    
    return roi, top3_bets

simulate_top3(test_df)

# ===== 11. ケリー基準でベット額計算（Top1艇） =====
def kelly_bet_fraction(p, b):
    # bが0以下の場合は0を返す（オッズが1.0以下の場合）
    if b <= 0:
        return 0
    return max(0, (p * (b + 1) - 1) / b)

def simulate_kelly(df):
    df = df.copy()
    # オッズ調整後の期待値でソート
    top1 = df.groupby("レースID").apply(lambda x: x.sort_values("expected_return", ascending=False).head(1)).reset_index(drop=True)
    # ケリー基準計算時は調整後のオッズを使用（より現実的な勝率・オッズ関係を反映）
    top1["kelly"] = top1.apply(lambda row: kelly_bet_fraction(row["pred_proba"], row["odds_adjusted"] - 1), axis=1)
    top1["bet_amt"] = top1["kelly"] * 100  # 100円が元本と仮定
    top1["bet_amt"] = top1["bet_amt"].clip(0, 100)  # 過剰な賭けを防ぐ
    total_cost = top1["bet_amt"].sum()
    # 実際の払戻計算は調整前のオッズを使用
    total_return = (top1["is_win"] * top1["odds_actual"] * top1["bet_amt"]).sum()
    roi = total_return / total_cost
    print("\n=== ケリー基準ベット（Top1艇） ===")
    print(f"合計投資: ¥{total_cost:,.0f}")
    print(f"払戻合計: ¥{total_return:,.0f}")
    print(f"回収率 (ROI): {roi * 100:.2f}%")
    
    # Kelly戦略の詳細
    print(f"的中率: {top1['is_win'].mean() * 100:.2f}%")
    print(f"平均オッズ: {top1['odds_actual'].mean():.2f}倍")
    print(f"平均調整オッズ: {top1['odds_adjusted'].mean():.2f}倍")
    print(f"平均ベット額: ¥{top1['bet_amt'].mean():.2f}")
    
    # 払戻金の可視化
    top1["払戻"] = top1["is_win"] * top1["odds_actual"] * top1["bet_amt"]
    visualize_returns(top1)
    
    return roi, top1

# ===== 11.5 閾値ベースの戦略（勝率30%以上かつ期待値1.8以上） =====
def simulate_threshold_strategy(df):
    df = df.copy()
    # 勝率30%以上かつ調整後期待値1.8以上という条件にマッチする艇を抽出
    threshold_bets = df[(df["pred_proba"] >= 0.3) & (df["expected_return"] >= 1.8)].copy()
    
    # 結果集計
    total_bets = len(threshold_bets)
    total_cost = 100 * total_bets
    total_return = (threshold_bets["is_win"] * threshold_bets["odds_actual"] * 100).sum()
    roi = total_return / total_cost if total_cost > 0 else 0
    
    print("\n=== 閾値ベース戦略（勝率30%以上かつ期待値1.8以上） ===")
    print(f"ベット数: {total_bets} 回")
    print(f"投資金額: ¥{total_cost:,.0f}")
    print(f"払戻金額: ¥{total_return:,.0f}")
    print(f"回収率 (ROI): {roi * 100:.2f}%")
    
    if total_bets > 0:
        # 的中率とオッズの分析
        print(f"的中率: {threshold_bets['is_win'].mean() * 100:.2f}%")
        print(f"平均オッズ: {threshold_bets['odds_actual'].mean():.2f}倍")
        print(f"平均期待値: {threshold_bets['expected_return'].mean():.2f}")
        
        # 1日あたりの平均ベット数
        days_count = threshold_bets["日付"].dt.date.nunique()
        print(f"対象期間: {days_count}日")
        print(f"1日あたり平均ベット数: {total_bets / days_count:.1f}回")
        
        # 払戻金の可視化
        threshold_bets["払戻"] = threshold_bets["is_win"] * threshold_bets["odds_actual"] * 100
        visualize_returns(threshold_bets)
    else:
        print("条件に合致するベットがありませんでした。")
    
    return roi, threshold_bets

simulate_kelly(test_df)

# 閾値ベース戦略の実行
simulate_threshold_strategy(test_df)

# ===== 12. 全データを使った再学習とモデル保存 =====
print("\n=== 全データを使った再学習 ===")
# 全データで再学習
X_all = df[feature_cols]
y_all = df["is_win"]

# 全データを訓練用と検証用に分割
all_rows = len(df)
val_size = int(all_rows * 0.2)
train_idx = all_rows - val_size

X_all_train = X_all.iloc[:train_idx]
y_all_train = y_all.iloc[:train_idx]
X_all_val = X_all.iloc[train_idx:]
y_all_val = y_all.iloc[train_idx:]

full_train_data = lgb.Dataset(X_all_train, label=y_all_train)
full_val_data = lgb.Dataset(X_all_val, label=y_all_val)

final_model = lgb.train(
    params,
    full_train_data,
    valid_sets=[full_val_data],
    num_boost_round=100,
    callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
)

# モデル保存ディレクトリの確認
models_dir = "models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"モデル保存用ディレクトリを作成しました: {models_dir}")

# データに含まれる最新の日付を取得
latest_date = df["日付"].max().strftime('%Y%m%d')

# モデルを保存
model_path = os.path.join(models_dir, f"model_{latest_date}.pkl")
final_model.save_model(model_path)
print(f"モデルを保存しました: {model_path}")

# 特徴量名も保存しておく（予測時に必要）
# feature_cols_path = os.path.join(models_dir, "feature_cols.txt")
# with open(feature_cols_path, "w") as f:
#     f.write("\n".join(feature_cols))
# print(f"特徴量リストを保存しました: {feature_cols_path}")
