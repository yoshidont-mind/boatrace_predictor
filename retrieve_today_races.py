import pandas as pd
from pyjpboatrace import PyJPBoatrace
from pyjpboatrace.const import STADIUMS_MAP
from datetime import datetime, date

# ====== 本日のレース一覧を取得する関数 ======
def get_today_races():
    today = datetime.today()
    today_str = today.strftime("%Y-%m-%d")
    today_date = date(today.year, today.month, today.day)

    scraper = PyJPBoatrace()
    stadiums_info = scraper.get_stadiums(d=today_date)
    stadium_names = [name for name in stadiums_info.keys() if name != 'date']

    stadium_id_map = {name: id for id, name in STADIUMS_MAP}
    entries = []

    for stadium in stadium_names:
        stadium_id = stadium_id_map.get(stadium)
        if stadium_id is None:
            continue
        try:
            races = scraper.get_12races(d=today_date, stadium=stadium_id)
            for race_key in races:
                if race_key in ['date', 'stadium']:
                    continue
                race_no = int(race_key.replace("R", ""))
                race_id = today.strftime("%Y%m%d") + f"{stadium_id:02}{race_no:02}"
                
                # 締切予定時刻とステータスの取得 (参照: 9_retrieve_data.py)
                race_data = races.get(race_key, {})
                deadline_time = race_data.get('vote_limit')
                race_status = race_data.get('status', "投票")  # ステータス情報を取得
                
                # 時刻部分だけを抽出 (例: "2025-04-01 15:17:00" -> "15:17")
                if deadline_time and isinstance(deadline_time, str):
                    try:
                        deadline_dt = datetime.strptime(deadline_time, "%Y-%m-%d %H:%M:%S")
                        deadline_time = deadline_dt.strftime("%H:%M")
                    except Exception as e:
                        print(f"⚠️ 締切時刻のフォーマット変換でエラー: {e}")
                        deadline_time = "不明"
                
                entries.append({
                    "レースID": race_id,
                    "場": stadium,
                    "R": race_no,
                    "締切予定時刻": deadline_time,
                    "ステータス": race_status
                })
        except Exception as e:
            print(f"{stadium} のレース情報取得に失敗: {e}")
            continue

    df = pd.DataFrame(entries)
    return df
