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
                closing_time = races[race_key].get("deadline")
                entries.append({
                    "レースID": race_id,
                    "場": stadium,
                    "R": race_no,
                    "締切予定時刻": closing_time,
                    "ステータス": "未予測"
                })
        except Exception as e:
            print(f"{stadium} のレース情報取得に失敗: {e}")
            continue

    df = pd.DataFrame(entries)
    return df
