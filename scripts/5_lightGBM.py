import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
import os
import sys

# Add the project root directory to the Python path to import modules from there
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import translation function only for feature importance visualization
from utils import get_japanese_to_english_columns_mapping

# ===== Japanese Font (for macOS) =====
mpl.rcParams['font.family'] = ['Hiragino Sans GB', 'AppleGothic']

# ===== 1. Load Data =====
# Get the data file path from command line arguments
if len(sys.argv) < 2:
    print("Error: Please specify the data file path.")
    sys.exit(1)

data_file_path = sys.argv[1]

df = pd.read_csv(data_file_path)

# Convert date column to datetime type
df["日付"] = pd.to_datetime(df["日付"])  # Date

# ===== 2. Features and Target Variable =====
feature_cols = [
    "支部", "級別", "艇番", "会場", "風量", "波", "月", "曜日",  # Branch, Class, Boat Number, Venue, Wind Speed, Wave, Month, Weekday
    "全国勝率_dev", "全国2連率_dev", "当地勝率_dev", "当地2連率_dev",  # National Win Rate (Deviation), National 2-Connection Rate (Deviation), Local Win Rate (Deviation), Local 2-Connection Rate (Deviation)
    "モーター2連率_dev", "ボート2連率_dev", "展示タイム_dev",  # Motor 2-Connection Rate (Deviation), Boat 2-Connection Rate (Deviation), Exhibition Time (Deviation)
]
X = df[feature_cols]
y = df["is_win"]

# ===== 3. Time Series Split =====
df = df.sort_values("日付")  # Date
# Split by date (ensure the same day is not split between train/test)
unique_dates = df["日付"].dt.date.unique()  # Date
unique_dates.sort()
split_idx = int(len(unique_dates) * 0.8)  # 80% for training data
split_date = unique_dates[split_idx]
print(f"Split date: {split_date} (Data before this date is for training, data from this date onwards is for testing)")

train_base_df = df[df["日付"].dt.date < split_date].copy()  # Date
test_all_df = df[df["日付"].dt.date >= split_date].copy()  # Date
print(f"Training base data: {len(train_base_df)} rows ({train_base_df['日付'].min()} 〜 {train_base_df['日付'].max()})")  # Date
print(f"Test target data: {len(test_all_df)} rows ({test_all_df['日付'].min()} 〜 {test_all_df['日付'].max()})")  # Date

# Walk-forward method parameters
params = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "random_state": 42
}

# ===== 4 & 5. Model Evaluation using Walk-Forward Method =====
print("\n===== Evaluation using Walk-Forward Method =====")
print("For each race day, build a model with data up to that point and predict only the races of that day")

# Get all test dates
test_dates = sorted(test_all_df["日付"].dt.date.unique())  # Date

# DataFrame to save all predictions
all_predictions = []

for i, test_date in enumerate(test_dates):
    # Progress display
    print(f"\nEvaluating: {test_date} ({i+1}/{len(test_dates)})")
    
    # Data of the day
    test_df = test_all_df[test_all_df["日付"].dt.date == test_date].copy()  # Date
    
    # All data before the day (for training)
    train_df = df[df["日付"].dt.date < test_date].copy()  # Date
    
    print(f"  Training data: {len(train_df)} rows, Test data: {len(test_df)} rows")
    
    # Prepare training and test data
    X_train = train_df[feature_cols]
    y_train = train_df["is_win"]
    X_test = test_df[feature_cols]
    y_test = test_df["is_win"]
    
    # Split training data into training and validation (latest 20% for validation)
    train_rows = len(train_df)
    if train_rows > 1000:  # Split only if there is enough data
        val_size = int(train_rows * 0.2)
        train_idx = train_rows - val_size
        
        X_train_train = X_train.iloc[:train_idx]
        y_train_train = y_train.iloc[:train_idx]
        X_train_val = X_train.iloc[train_idx:]
        y_train_val = y_train.iloc[train_idx:]
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train_train, label=y_train_train)
        val_data = lgb.Dataset(X_train_val, label=y_train_val)
        
        # Train model (using validation data for early stopping)
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=100,
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
        )
    else:
        # Train without early stopping if data is limited
        train_data = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(
            params,
            train_data,
            num_boost_round=50  # Fewer rounds without early stopping
        )
    
    # Predict
    test_df["pred_proba"] = model.predict(X_test)
    test_df["pred_label"] = (test_df["pred_proba"] > 0.5).astype(int)
    
    # Simple evaluation (for the day)
    if len(test_df) > 0:
        daily_accuracy = accuracy_score(test_df["is_win"], test_df["pred_label"])
        print(f"  Daily prediction accuracy: {daily_accuracy:.4f}")
    
    # Save results
    all_predictions.append(test_df)

# Combine all predictions
test_all_predictions = pd.concat(all_predictions)

# Evaluate results
y_test = test_all_predictions["is_win"]
y_pred_proba = test_all_predictions["pred_proba"]
y_pred_label = test_all_predictions["pred_label"]

print("\n=== Overall Evaluation Results using Walk-Forward Method ===")
print("AUC:", round(roc_auc_score(y_test, y_pred_proba), 4))
print("Accuracy:", round(accuracy_score(y_test, y_pred_label), 4))
print(classification_report(y_test, y_pred_label))

# Update test data (using walk-forward results)
test_df = test_all_predictions

# ===== 6. Feature Importance =====
# Check graph save directory
graphs_dir = "graphs"
if not os.path.exists(graphs_dir):
    os.makedirs(graphs_dir)
    print(f"Created directory for saving graphs: {graphs_dir}")

# Get column name mapping dictionary
column_mapping = get_japanese_to_english_columns_mapping()

# Create custom feature importance graph with English column names
plt.figure(figsize=(10, 6))

# Get feature importance from the model
importance = model.feature_importance(importance_type='split')
feature_names = model.feature_name()

# Create a dataframe for easier manipulation
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values('Importance', ascending=False)

# Limit to top 15 features
importance_df = importance_df.head(15)

# Translate feature names to English (if mapping exists)
translated_names = []
for feature in importance_df['Feature']:
    if feature in column_mapping:
        translated_names.append(column_mapping[feature])
    else:
        # Try to match partial names (for columns like x_dev)
        found = False
        for jp_name, en_name in column_mapping.items():
            if feature.startswith(jp_name):
                # Replace Japanese part with English part
                translated_name = feature.replace(jp_name, en_name)
                translated_names.append(translated_name)
                found = True
                break
        if not found:
            translated_names.append(feature)  # Keep original if no mapping found

importance_df['Translated_Feature'] = translated_names

# Plot with translated feature names
plt.barh(range(len(importance_df)), importance_df['Importance'], align='center')
plt.yticks(range(len(importance_df)), importance_df['Translated_Feature'])
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.gca().invert_yaxis()  # Highest importance at the top

# Get the latest date in the data and use it in the file name
latest_date = df["日付"].max().strftime('%Y%m%d')  # Date
feature_importance_path = os.path.join(graphs_dir, f"feature_importance_{latest_date}.png")

# Save graph
plt.tight_layout()
plt.savefig(feature_importance_path, dpi=300)
print(f"Saved feature importance graph (with English labels): {feature_importance_path}")
# plt.show()

# ===== 7. Expected Value Calculation =====
# Adjustment function considering odds fluctuation (lower high odds more)
def adjusted_odds(raw_odds, weight=0.2):
    """
    Adjustment function considering actual odds fluctuation
    Lower high odds more (less change for low odds)
    
    Parameters:
    -----------
    raw_odds : float
        Original odds value
    weight : float, default=0.2
        Strength of adjustment (higher value means stronger adjustment)
    
    Returns:
    --------
    float
        Adjusted odds value
    """
    return raw_odds * (1 - weight * (1 / raw_odds))

# Calculate expected value for test data (using walk-forward prediction results)
test_df["odds_actual"] = test_df["単勝オッズ"] / 100  # Win Odds
test_df["odds_adjusted"] = test_df["odds_actual"].apply(adjusted_odds)
test_df["expected_return_raw"] = test_df["pred_proba"] * test_df["odds_actual"]  # Traditional calculation method
test_df["expected_return"] = test_df["pred_proba"] * test_df["odds_adjusted"]  # Considering odds fluctuation

# Compare expected values before and after adjustment
print("\n=== Effect of Odds Fluctuation Adjustment ===")
odds_comparison = test_df.groupby(pd.cut(test_df["odds_actual"], 
                         bins=[0, 1.5, 2, 3, 5, 10, 50, 1000], 
                         labels=["〜1.5倍", "1.5〜2倍", "2〜3倍", "3〜5倍", "5〜10倍", "10〜50倍", "50倍〜"])).agg({
    "odds_actual": "mean",
    "odds_adjusted": "mean",
    "expected_return_raw": "mean",
    "expected_return": "mean",
    "レースID": "count"  # Race ID
}).rename(columns={"レースID": "件数"})  # Count

odds_comparison["補正率"] = (odds_comparison["odds_adjusted"] / odds_comparison["odds_actual"] - 1) * 100  # Adjustment Rate
odds_comparison["期待値変化率"] = (odds_comparison["expected_return"] / odds_comparison["expected_return_raw"] - 1) * 100  # Expected Value Change Rate

print(odds_comparison[["件数", "odds_actual", "odds_adjusted", "補正率", "expected_return_raw", "expected_return", "期待値変化率"]])  # Count, Adjustment Rate, Expected Value Change Rate

# ===== 8. Recommended Bets Display =====
def recommend_bets(df, threshold=1.0):
    return df[df["expected_return"] > threshold].sort_values("expected_return", ascending=False)

print("\n=== Recommended Bets with High Expected Value ===")
print(recommend_bets(test_df)[["レースID", "艇番", "pred_proba", "odds_actual", "expected_return"]].head(10))  # Race ID, Boat Number

# ===== 8.5 Visualization of Payout Distribution =====
def visualize_returns(df):
    # Calculate payout
    df = df.copy()
    df["払戻"] = df["is_win"] * df["odds_actual"] * 100  # Payout
    
    # Visualize payout distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df["払戻"], bins=50)  # Payout
    plt.title("Payout Distribution")
    plt.xlabel("Payout Amount (Yen)")
    plt.ylabel("Frequency")
    # plt.show()
    
    # ROI analysis by odds
    df["オッズ区間"] = pd.cut(df["odds_actual"], 
                          bins=[0, 1.5, 2, 3, 5, 10, 50, 1000], 
                          labels=["〜1.5倍", "1.5〜2倍", "2〜3倍", "3〜5倍", "5〜10倍", "10〜50倍", "50倍〜"])  # Odds Range
    
    odds_roi = df.groupby("オッズ区間").apply(
        lambda x: pd.Series({
            "ベット数": len(x),  # Number of Bets
            "的中数": x["is_win"].sum(),  # Number of Hits
            "的中率": x["is_win"].mean() * 100,  # Hit Rate
            "投資額": len(x) * 100,  # Investment Amount
            "払戻額": (x["is_win"] * x["odds_actual"] * 100).sum(),  # Payout Amount
            "回収率": (x["is_win"] * x["odds_actual"] * 100).sum() / (len(x) * 100) * 100  # ROI
        })
    )
    
    print("\n=== ROI by Odds ===")
    print(odds_roi)
    
    # Visualize ROI by odds
    plt.figure(figsize=(12, 6))
    sns.barplot(x=odds_roi.index, y=odds_roi["回収率"])  # ROI
    plt.title("ROI by Odds")
    plt.ylabel("ROI (%)")
    plt.axhline(y=100, color='r', linestyle='-', alpha=0.3)  # 100% line
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.show()
    
    # Big win impact analysis
    threshold = 1000  # Define big win as 1000 yen or more
    big_wins = df[df["払戻"] >= threshold]  # Payout
    
    if len(big_wins) > 0:
        print(f"\n=== Impact of Big Wins (¥{threshold} or more) ===")
        total_return = df["払戻"].sum()  # Payout
        big_win_return = big_wins["払戻"].sum()  # Payout
        total_cost = len(df) * 100
        
        print(f"Number of big wins: {len(big_wins)}")
        print(f"Total payout from big wins: ¥{big_win_return:,.0f} ({big_win_return/total_return*100:.1f}% contribution)")
        
        # ROI excluding big wins
        roi_with_big = total_return / total_cost
        roi_without_big = (total_return - big_win_return) / total_cost
        
        print(f"Overall ROI: {roi_with_big*100:.2f}%")
        print(f"ROI excluding big wins: {roi_without_big*100:.2f}%")
        
        # Big win details
        print("\n=== Big Win Details ===")
        print(big_wins[["レースID", "艇番", "odds_actual", "払戻"]].sort_values("払戻", ascending=False))  # Race ID, Boat Number, Payout

# Visualize payout distribution for test data
print("\n=== Payout Analysis for Test Data ===")
visualize_returns(test_df)

# ===== 9. Backtest: Bet 100 yen on the top 1 boat with the highest expected value =====
def simulate_top1(df):
    df = df.copy()
    # Sort by adjusted expected value
    top_bets = df.groupby("レースID").apply(lambda x: x.sort_values("expected_return", ascending=False).head(1)).reset_index(drop=True)  # Race ID
    total_bets = len(top_bets)
    total_cost = 100 * total_bets
    # Calculate actual payout using unadjusted odds (actual payout is calculated using odds board values)
    total_return = (top_bets["is_win"] * top_bets["odds_actual"] * 100).sum()
    roi = total_return / total_cost
    print("\n=== Backtest (Top 1 Boat) ===")
    print(f"Number of bets: {total_bets}")
    print(f"Total investment: ¥{total_cost:,.0f}")
    print(f"Total payout: ¥{total_return:,.0f}")
    print(f"ROI: {roi * 100:.2f}%")
    
    # Top 1 hit rate and odds distribution
    print(f"Hit rate: {top_bets['is_win'].mean() * 100:.2f}%")
    print(f"Average odds: {top_bets['odds_actual'].mean():.2f}")
    
    # Visualize payout
    top_bets["払戻"] = top_bets["is_win"] * top_bets["odds_actual"] * 100  # Payout
    visualize_returns(top_bets)
    
    return roi, top_bets

simulate_top1(test_df)

# ===== 10. Diversified Investment: Bet 100 yen on the top 3 boats =====
def simulate_top3(df):
    df = df.copy()
    # Sort by adjusted expected value
    top3_bets = df.groupby("レースID").apply(lambda x: x.sort_values("expected_return", ascending=False).head(3)).reset_index(drop=True)  # Race ID
    total_bets = len(top3_bets)
    total_cost = 100 * total_bets
    # Calculate actual payout using unadjusted odds
    total_return = (top3_bets["is_win"] * top3_bets["odds_actual"] * 100).sum()
    roi = total_return / total_cost
    print("\n=== Diversified Investment (Top 3 Boats) ===")
    print(f"Number of bets: {total_bets} (3 boats per race)")
    print(f"Total investment: ¥{total_cost:,.0f}")
    print(f"Total payout: ¥{total_return:,.0f}")
    print(f"ROI: {roi * 100:.2f}%")
    
    # Top 3 hit rate and odds distribution
    print(f"Hit rate: {top3_bets['is_win'].mean() * 100:.2f}%")
    print(f"Average odds: {top3_bets['odds_actual'].mean():.2f}")
    
    # Visualize payout
    top3_bets["払戻"] = top3_bets["is_win"] * top3_bets["odds_actual"] * 100  # Payout
    visualize_returns(top3_bets)
    
    return roi, top3_bets

simulate_top3(test_df)

# ===== 11. Kelly Criterion Bet Calculation (Top 1 Boat) =====
def kelly_bet_fraction(p, b):
    # Return 0 if b is 0 or less (odds of 1.0 or less)
    if b <= 0:
        return 0
    return max(0, (p * (b + 1) - 1) / b)

def simulate_kelly(df):
    df = df.copy()
    # Sort by adjusted expected value
    top1 = df.groupby("レースID").apply(lambda x: x.sort_values("expected_return", ascending=False).head(1)).reset_index(drop=True)  # Race ID
    # Use adjusted odds for Kelly criterion calculation (reflects more realistic win rate-odds relationship)
    top1["kelly"] = top1.apply(lambda row: kelly_bet_fraction(row["pred_proba"], row["odds_adjusted"] - 1), axis=1)
    top1["bet_amt"] = top1["kelly"] * 100  # Assume 100 yen as the principal
    top1["bet_amt"] = top1["bet_amt"].clip(0, 100)  # Prevent excessive betting
    total_cost = top1["bet_amt"].sum()
    # Calculate actual payout using unadjusted odds
    total_return = (top1["is_win"] * top1["odds_actual"] * top1["bet_amt"]).sum()
    roi = total_return / total_cost
    print("\n=== Kelly Criterion Bet (Top 1 Boat) ===")
    print(f"Total investment: ¥{total_cost:,.0f}")
    print(f"Total payout: ¥{total_return:,.0f}")
    print(f"ROI: {roi * 100:.2f}%")
    
    # Kelly strategy details
    print(f"Hit rate: {top1['is_win'].mean() * 100:.2f}%")
    print(f"Average odds: {top1['odds_actual'].mean():.2f}")
    print(f"Average adjusted odds: {top1['odds_adjusted'].mean():.2f}")
    print(f"Average bet amount: ¥{top1['bet_amt'].mean():.2f}")
    
    # Visualize payout
    top1["払戻"] = top1["is_win"] * top1["odds_actual"] * top1["bet_amt"]  # Payout
    visualize_returns(top1)
    
    return roi, top1

# ===== 11.5 Threshold-Based Strategy (Win rate 30% or more and expected value 1.8 or more) =====
def simulate_threshold_strategy(df):
    df = df.copy()
    # Extract boats that match the condition of win rate 30% or more and adjusted expected value 1.8 or more
    threshold_bets = df[(df["pred_proba"] >= 0.3) & (df["expected_return"] >= 1.8)].copy()
    
    # Summarize results
    total_bets = len(threshold_bets)
    total_cost = 100 * total_bets
    total_return = (threshold_bets["is_win"] * threshold_bets["odds_actual"] * 100).sum()
    roi = total_return / total_cost if total_cost > 0 else 0
    
    print("\n=== Threshold-Based Strategy (Win rate 30% or more and expected value 1.8 or more) ===")
    print(f"Number of bets: {total_bets}")
    print(f"Total investment: ¥{total_cost:,.0f}")
    print(f"Total payout: ¥{total_return:,.0f}")
    print(f"ROI: {roi * 100:.2f}%")
    
    if total_bets > 0:
        # Analyze hit rate and odds
        print(f"Hit rate: {threshold_bets['is_win'].mean() * 100:.2f}%")
        print(f"Average odds: {threshold_bets['odds_actual'].mean():.2f}")
        print(f"Average expected value: {threshold_bets['expected_return'].mean():.2f}")
        
        # Average number of bets per day
        days_count = threshold_bets["日付"].dt.date.nunique()  # Date
        print(f"Target period: {days_count} days")
        print(f"Average number of bets per day: {total_bets / days_count:.1f}")
        
        # Visualize payout
        threshold_bets["払戻"] = threshold_bets["is_win"] * threshold_bets["odds_actual"] * 100  # Payout
        visualize_returns(threshold_bets)
    else:
        print("No bets matched the conditions.")
    
    return roi, threshold_bets

simulate_kelly(test_df)

# Execute threshold-based strategy
simulate_threshold_strategy(test_df)

# ===== 12. Asset Transition Simulation =====
def run_simulation(test_df, initial_capital=3000, daily_budget=1000):
    """
    Run asset transition simulation based on four strategies and visualize the results

    Parameters:
    -----------
    test_df : DataFrame
        Test data frame
    initial_capital : int
        Initial capital (yen)
    daily_budget : int
        Daily investment budget (yen)
    """
    print("\n===== Asset Transition Simulation =====")
    print(f"Initial capital: ¥{initial_capital:,}")
    print(f"Daily investment budget: ¥{daily_budget:,}")
    
    # Get the last 30 days of the dataset
    test_dates = sorted(test_df["日付"].dt.date.unique())  # Date
    if len(test_dates) > 30:
        test_dates = test_dates[-30:]
        print(f"Simulation for the last 30 days: {test_dates[0]} 〜 {test_dates[-1]}")
        test_df = test_df[test_df["日付"].dt.date.isin(test_dates)].copy()  # Date
    else:
        print(f"Simulation for the entire test period: {test_dates[0]} 〜 {test_dates[-1]}")
    
    # List of strategy names
    strategies = ["Top1 Strategy", "Top3 Strategy", "Kelly Strategy", "Threshold Strategy"]
    
    # Dictionaries to record asset transition and investment amount (by strategy)
    capital_history = {strategy: [] for strategy in strategies}
    total_investment = {strategy: 0 for strategy in strategies}
    
    # Update assets for each date
    for date in test_dates:
        day_df = test_df[test_df["日付"].dt.date == date].copy()  # Date
        
        # ===== Identify investment targets for each strategy =====
        
        # Top1 Strategy: Top 1 boat with the highest expected value for each race
        top1_bets = day_df.groupby("レースID").apply(
            lambda x: x.sort_values("expected_return", ascending=False).head(1)
        ).reset_index(drop=True)  # Race ID
        
        # Top3 Strategy: Top 3 boats with the highest expected value for each race
        top3_bets = day_df.groupby("レースID").apply(
            lambda x: x.sort_values("expected_return", ascending=False).head(3)
        ).reset_index(drop=True)  # Race ID
        
        # Kelly Strategy: Top 1 boat with the highest expected value for each race
        kelly_bets = day_df.groupby("レースID").apply(
            lambda x: x.sort_values("expected_return", ascending=False).head(1)
        ).reset_index(drop=True)  # Race ID
        
        # Calculate Kelly ratio
        kelly_bets["kelly"] = kelly_bets.apply(
            lambda row: kelly_bet_fraction(row["pred_proba"], row["odds_adjusted"] - 1) * 0.5, axis=1
        )
        
        # Threshold Strategy: Boats with win rate 30% or more and expected value 1.8 or more
        threshold_bets = day_df[(day_df["pred_proba"] >= 0.3) & (day_df["expected_return"] >= 1.8)].copy()
        
        # ===== Allocate investment amount (adjust daily budget to match total investment for each strategy) =====
        
        # Number of bets for each strategy
        bet_counts = {
            "Top1 Strategy": len(top1_bets),
            "Top3 Strategy": len(top3_bets),
            "Kelly Strategy": len(kelly_bets),
            "Threshold Strategy": len(threshold_bets)
        }
        
        # Calculate investment unit (divide budget by number of bets, round to minimum 100 yen)
        bet_units = {}
        for strategy, count in bet_counts.items():
            if count > 0:
                # Investment amount per bet = daily budget ÷ number of bets
                unit = daily_budget / count
                # Minimum 100 yen, round to 100 yen units
                bet_units[strategy] = max(100, int(unit / 100) * 100)
            else:
                bet_units[strategy] = 0
        
        # ===== Execute investment and update assets for each strategy =====
        for strategy in strategies:
            # Get previous day's capital
            if len(capital_history[strategy]) == 0:
                current_capital = initial_capital
            else:
                current_capital = capital_history[strategy][-1][1]
            
            # Bet targets and unit investment amount
            if strategy == "Top1 Strategy":
                bets = top1_bets.copy()
                bet_amount = bet_units[strategy]
                
            elif strategy == "Top3 Strategy":
                bets = top3_bets.copy()
                # For Top3, divide unit amount by 3 as it bets on 3 boats per race
                bet_amount = max(100, int(bet_units[strategy] / 3 / 100) * 100)
                
            elif strategy == "Kelly Strategy":
                bets = kelly_bets.copy()
                if len(bets) > 0:
                    # Determine investment amount based on Kelly ratio for each race
                    kelly_sum = bets["kelly"].sum()
                    if kelly_sum > 0:
                        # Distribute budget based on Kelly ratio (minimum 100 yen)
                        bets["bet_ratio"] = bets["kelly"] / kelly_sum
                        bets["bet_amt"] = (bets["bet_ratio"] * daily_budget / 100).round() * 100
                        bets["bet_amt"] = bets["bet_amt"].clip(lower=100, upper=1000)
                    else:
                        # If Kelly ratio is 0, distribute evenly
                        bets["bet_amt"] = max(100, int(daily_budget / len(bets) / 100) * 100)
                bet_amount = None  # Different amount for each race
                
            elif strategy == "Threshold Strategy":
                bets = threshold_bets.copy()
                bet_amount = bet_units[strategy]
                
            # Execute investment
            if len(bets) > 0:
                # Check capital
                if strategy == "Kelly Strategy":
                    total_bet = bets["bet_amt"].sum()
                else:
                    total_bet = len(bets) * bet_amount
                
                # If capital is insufficient
                if current_capital < total_bet:
                    if strategy == "Kelly Strategy":
                        # Scale down to match capital
                        scale_factor = current_capital / total_bet
                        bets["bet_amt"] = (bets["bet_amt"] * scale_factor / 100).round() * 100
                        bets["bet_amt"] = bets["bet_amt"].clip(lower=100)
                        # If still insufficient, sort by expected return
                        while bets["bet_amt"].sum() > current_capital:
                            if len(bets) <= 1:
                                bets = pd.DataFrame()  # Empty
                                break
                            bets = bets.sort_values("expected_return", ascending=False).iloc[:-1]
                        total_bet = bets["bet_amt"].sum()
                    else:
                        # Reduce number of bets
                        max_bets = int(current_capital / bet_amount)
                        if max_bets == 0:
                            bets = pd.DataFrame()  # Empty
                            total_bet = 0
                        else:
                            bets = bets.sort_values("expected_return", ascending=False).head(max_bets)
                            total_bet = len(bets) * bet_amount
                
                # Calculate payout
                if len(bets) > 0:
                    if strategy == "Kelly Strategy":
                        total_return = (bets["is_win"] * bets["odds_actual"] * bets["bet_amt"]).sum()
                    else:
                        total_return = (bets["is_win"] * bets["odds_actual"] * bet_amount).sum()
                    
                    # Update capital
                    current_capital = current_capital - total_bet + total_return
                    
                    # Update cumulative investment amount
                    total_investment[strategy] += total_bet
            
            # Update asset history
            capital_history[strategy].append((date, current_capital))

    # Convert results to DataFrame
    result_dfs = {}
    for strategy, history in capital_history.items():
        result_dfs[strategy] = pd.DataFrame(history, columns=["日付", "資金"])  # Date, Capital
    
    # Display final capital
    print("\n===== Simulation Results =====")
    for strategy, df in result_dfs.items():
        final_capital = df["資金"].iloc[-1]  # Capital
        roi = (final_capital - initial_capital) / initial_capital * 100
        print(f"{strategy}: Final Capital ¥{int(final_capital):,} (ROI: {roi:.2f}%)")
    
    # Plot graph
    plt.figure(figsize=(12, 6))
    for strategy, df in result_dfs.items():
        plt.plot(df["日付"], df["資金"], marker="o", label=strategy)  # Date, Capital
    
    plt.title("Asset Transition Simulation (Comparison of 4 Strategies)")
    plt.xlabel("Date")
    plt.ylabel("Capital (Yen)")
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save graph
    latest_date = test_df["日付"].max().strftime('%Y%m%d')  # Date
    graph_path = os.path.join("graphs", f"asset_simulation_{latest_date}.png")
    plt.savefig(graph_path, dpi=300)
    print(f"\nGraph saved: {graph_path}")
    # plt.show()

# Execute asset transition simulation
run_simulation(test_df)

# ===== 13. Retraining with All Data and Model Saving =====
print("\n=== Retraining with All Data ===")
# Retrain with all data
X_all = df[feature_cols]
y_all = df["is_win"]

# Split all data into training and validation sets
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

# Check model saving directory
models_dir = "models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"Created directory for model saving: {models_dir}")

# Get the latest date included in the data
latest_date = df["日付"].max().strftime('%Y%m%d')  # Date

# Save model
model_path = os.path.join(models_dir, f"model_{latest_date}.pkl")
final_model.save_model(model_path)
print(f"Model saved: {model_path}")

# Also save feature names (needed for prediction)
# feature_cols_path = os.path.join(models_dir, "feature_cols.txt")
# with open(feature_cols_path, "w") as f:
#     f.write("\n".join(feature_cols))
# print(f"Feature list saved: {feature_cols_path}")
