import pandas as pd
import pickle
from datetime import datetime
from pathlib import Path
from utils import preprocess_boatrace_dataframe
from pyjpboatrace import PyJPBoatrace
import traceback

# ====== 単一レースの予測関数 ======
def predict_single_race(race_id):
    try:
        date_str = race_id[:8]  # yyyymmdd
        stadium_id = int(race_id[8:10])
        race_no = int(race_id[10:])
        target_date = datetime.strptime(date_str, "%Y%m%d").date()

        scraper = PyJPBoatrace()
        race_info = scraper.get_race_info(d=target_date, stadium=stadium_id, race=race_no)

        race_entries = []
        for boat_no in range(1, 7):
            boat_key = f"boat{boat_no}"
            boat_data = race_info.get(boat_key, {})
            race_entries.append({
                "レースID": race_id,
                "日付": target_date,
                "艇番": boat_no,
                "選手名": boat_data.get("name", ""),
                "級別": boat_data.get("class", ""),
                "支部": boat_data.get("branch", ""),
                "全国勝率": boat_data.get("global_win_pt", None),
                "当地勝率": boat_data.get("local_win_pt", None),
                "モーター番号": boat_data.get("motor", None),
                "モーター2連率": boat_data.get("motor_in2nd", None),
                "ボート番号": boat_data.get("boat", None),
                "ボート2連率": boat_data.get("boat_in2nd", None),
                "展示タイム": boat_data.get("display_time", None),
                "体重": boat_data.get("weight", None),
                "年齢": boat_data.get("age", None),
            })

        df_race = pd.DataFrame(race_entries)

        # ====== 前処理 ======
        df_processed = preprocess_boatrace_dataframe(df_race.copy())

        # ====== モデル読み込み ======
        model_path = Path("models/model_20250401.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # ====== 予測 ======
        feature_cols = [
            "支部", "級別", "艇番", "会場", "風量", "波", "月", "曜日",
            "全国勝率_dev", "全国2連率_dev", "当地勝率_dev", "当地2連率_dev",
            "モーター2連率_dev", "ボート2連率_dev", "展示タイム_dev"
        ]

        df_processed["pred_proba"] = model.predict_proba(df_processed[feature_cols])[:, 1]
        df_processed["期待値"] = df_processed["pred_proba"] * df_processed["単勝オッズ"] / 100

        df_result = df_processed[["艇番", "pred_proba", "単勝オッズ", "期待値"]].sort_values("期待値", ascending=False)
        df_result.columns = ["艇番", "勝率(予測)", "単勝オッズ", "期待値"]

        return df_result, datetime.now()

    except Exception as e:
        print(f"エラー: {e}")
        traceback.print_exc()
        return None, datetime.now()
