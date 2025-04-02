import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import matplotlib as mpl
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
split_date = df["日付"].quantile(0.8)  # 上位20%をテストデータに
train_df = df[df["日付"] < split_date].copy()
test_df = df[df["日付"] >= split_date].copy()

X_train = train_df[feature_cols]
y_train = train_df["is_win"]
X_test = test_df[feature_cols]
y_test = test_df["is_win"]

# ===== 4. モデル学習 =====
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test)

params = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "random_state": 42
}

model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, valid_data],
    valid_names=["train", "valid"],
    num_boost_round=100,
    callbacks=[lgb.early_stopping(stopping_rounds=10)]
)

# ===== 5. モデル評価 =====
y_pred_proba = model.predict(X_test)
y_pred_label = (y_pred_proba > 0.5).astype(int)

print("=== モデル評価結果 ===")
print("AUC:", round(roc_auc_score(y_test, y_pred_proba), 4))
print("Accuracy:", round(accuracy_score(y_test, y_pred_label), 4))
print(classification_report(y_test, y_pred_label))

# ===== 6. 特徴量重要度 =====
lgb.plot_importance(model, max_num_features=15, figsize=(10, 6))
plt.title("Feature Importance")
plt.show()

# ===== 7. 期待値計算 =====
df["pred_proba"] = model.predict(df[feature_cols])
df["odds_actual"] = df["単勝オッズ"] / 100
df["expected_return"] = df["pred_proba"] * df["odds_actual"]

test_df["pred_proba"] = model.predict(test_df[feature_cols])
test_df["odds_actual"] = test_df["単勝オッズ"] / 100
test_df["expected_return"] = test_df["pred_proba"] * test_df["odds_actual"]

# ===== 8. 推奨買い目表示 =====
def recommend_bets(df, threshold=1.0):
    return df[df["expected_return"] > threshold].sort_values("expected_return", ascending=False)

print("\n=== 期待値が高いおすすめ買い目 ===")
print(recommend_bets(test_df)[["レースID", "艇番", "pred_proba", "odds_actual", "expected_return"]].head(10))

# ===== 9. バックテスト：期待値最大の1艇に100円ずつ =====
def simulate_top1(df):
    df = df.copy()
    top_bets = df.groupby("レースID").apply(lambda x: x.sort_values("expected_return", ascending=False).head(1)).reset_index(drop=True)
    total_bets = len(top_bets)
    total_cost = 100 * total_bets
    total_return = (top_bets["is_win"] * top_bets["odds_actual"] * 100).sum()
    roi = total_return / total_cost
    print("\n=== バックテスト（Top1艇） ===")
    print(f"ベット数: {total_bets} 回")
    print(f"投資金額: ¥{total_cost:,.0f}")
    print(f"払戻金額: ¥{total_return:,.0f}")
    print(f"回収率 (ROI): {roi * 100:.2f}%")
    return roi, top_bets

simulate_top1(test_df)

# ===== 10. 分散投資：Top3艇に100円ずつ =====
def simulate_top3(df):
    df = df.copy()
    top3_bets = df.groupby("レースID").apply(lambda x: x.sort_values("expected_return", ascending=False).head(3)).reset_index(drop=True)
    total_bets = len(top3_bets)
    total_cost = 100 * total_bets
    total_return = (top3_bets["is_win"] * top3_bets["odds_actual"] * 100).sum()
    roi = total_return / total_cost
    print("\n=== 分散投資（Top3艇） ===")
    print(f"ベット数: {total_bets} 回（各レース3艇）")
    print(f"投資金額: ¥{total_cost:,.0f}")
    print(f"払戻金額: ¥{total_return:,.0f}")
    print(f"回収率 (ROI): {roi * 100:.2f}%")
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
    top1 = df.groupby("レースID").apply(lambda x: x.sort_values("expected_return", ascending=False).head(1)).reset_index(drop=True)
    top1["kelly"] = top1.apply(lambda row: kelly_bet_fraction(row["pred_proba"], row["odds_actual"] - 1), axis=1)
    top1["bet_amt"] = top1["kelly"] * 100  # 100円が元本と仮定
    top1["bet_amt"] = top1["bet_amt"].clip(0, 100)  # 過剰な賭けを防ぐ
    total_cost = top1["bet_amt"].sum()
    total_return = (top1["is_win"] * top1["odds_actual"] * top1["bet_amt"]).sum()
    roi = total_return / total_cost
    print("\n=== ケリー基準ベット（Top1艇） ===")
    print(f"合計投資: ¥{total_cost:,.0f}")
    print(f"払戻合計: ¥{total_return:,.0f}")
    print(f"回収率 (ROI): {roi * 100:.2f}%")
    return roi, top1

simulate_kelly(test_df)

# ===== 12. 全データを使った再学習とモデル保存 =====
print("\n=== 全データを使った再学習 ===")
# 全データで再学習
X_all = df[feature_cols]
y_all = df["is_win"]

full_data = lgb.Dataset(X_all, label=y_all)

final_model = lgb.train(
    params,
    full_data,
    num_boost_round=model.best_iteration,  # 上で学習した最適なイテレーション回数を使用
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
