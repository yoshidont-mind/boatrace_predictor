import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

# ===== 1. データ読み込み =====
df = pd.read_csv("/Users/yoshidont_mind/Library/Mobile Documents/com~apple~CloudDocs/Desktop/CST term4/COMP 4949 Big Data Analytics Methods/boat_race_prediction/data/preprocessed/preprocessed_6.csv")  # ファイルパス調整してね
df["日付"] = pd.to_datetime(df["レースID"].astype(str).str[:8], format="%Y%m%d")
df = df.sort_values("日付").reset_index(drop=True)

# ===== 2. 運用対象：直近3ヶ月（例） =====
start_date = pd.to_datetime("2025-03-01")
end_date = pd.to_datetime("2025-03-28")
target_dates = df[(df["日付"] >= start_date) & (df["日付"] <= end_date)]["日付"].sort_values().unique()

# ===== 3. 特徴量定義 =====
feature_cols = [
    "支部", "級別", "艇番", "会場", "風量", "波", "月", "曜日",
    "全国勝率_dev", "全国2連率_dev", "当地勝率_dev", "当地2連率_dev",
    "モーター2連率_dev", "ボート2連率_dev", "展示タイム_dev"
]

# ===== 4. 安全なケリー関数（ハーフケリー＋上限） =====
def safe_kelly(p, b, max_kelly=0.1):
    if b <= 0:
        return 0
    k = (p * (b + 1) - 1) / b
    return max(0, min(k * 0.5, max_kelly))  # ハーフケリー + 上限

# ===== 5. 初期資金設定 =====
capital = 30000
capital_history = []

# ===== 6. 日毎にループ（未来データ不使用） =====
for i, current_date in enumerate(target_dates):
    train_df = df[df["日付"] < current_date].copy()
    test_df = df[df["日付"] == current_date].copy()

    if len(train_df) == 0 or len(test_df) == 0 or capital <= 0:
        capital_history.append((current_date, capital))
        continue

    print(f"[{i+1}/{len(target_dates)}] {current_date.date()}｜資金: ¥{int(capital):,}")

    # モデル学習
    X_train = train_df[feature_cols]
    y_train = train_df["is_win"]
    model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        random_state=42,
        n_estimators=100
    )
    model.fit(X_train, y_train)

    # 予測・ケリー係数計算
    test_df = test_df.copy()
    test_df["pred_proba"] = model.predict_proba(test_df[feature_cols])[:, 1]
    test_df["odds_actual"] = test_df["単勝オッズ"] / 100
    test_df["kelly_frac"] = test_df.apply(
        lambda row: safe_kelly(row["pred_proba"], row["odds_actual"] - 1), axis=1
    )

    # 各レース1艇ベット（期待値最大の艇）
    top1_bets = test_df.groupby("レースID").apply(
        lambda x: x.sort_values("kelly_frac", ascending=False).head(1)
    ).reset_index(drop=True)

    # ベット額 = ケリー率 × 資金（100円単位に丸め・最大2000円制限）
    top1_bets["bet_amt"] = (top1_bets["kelly_frac"] * capital / 100).round() * 100
    top1_bets["bet_amt"] = top1_bets["bet_amt"].clip(lower=0, upper=2000)

    # 払戻金計算
    top1_bets["payout"] = top1_bets["is_win"] * top1_bets["odds_actual"] * top1_bets["bet_amt"]

    total_bet = top1_bets["bet_amt"].sum()
    total_return = top1_bets["payout"].sum()

    # 資金更新
    capital = capital - total_bet + total_return
    capital_history.append((current_date, capital))

# ===== 7. 資金推移グラフ描画 =====
capital_df = pd.DataFrame(capital_history, columns=["日付", "資金"])
plt.figure(figsize=(12, 6))
plt.plot(capital_df["日付"], capital_df["資金"], marker="o")
plt.title("資金推移（ハーフケリー戦略・制限付き）")
plt.xlabel("日付")
plt.ylabel("資金（円）")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
