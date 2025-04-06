import pandas as pd
from pyjpboatrace import PyJPBoatrace
from datetime import datetime, date
from pyjpboatrace.const import STADIUMS_MAP
import traceback
import pytz

# ===== 1. スクレーパー初期化 =====
print("スクレーパー初期化中...")
scraper = PyJPBoatrace()
# スクレイピングにはシステム時刻を使用
today_datetime = datetime.today()
today = today_datetime.strftime("%Y-%m-%d")
today_date = date(today_datetime.year, today_datetime.month, today_datetime.day)

# 表示用にJST時刻を取得
jst = pytz.timezone('Asia/Tokyo')
jst_now = datetime.now(pytz.UTC).astimezone(jst)
jst_today = jst_now.strftime("%Y-%m-%d")

print(f"システム日付: {today_date}")
print(f"JST日付: {jst_today}")

# ===== 2. 今日開催中のレース場一覧を取得 =====
print("レース場一覧を取得中...")
stadiums_info = scraper.get_stadiums(d=today_date)
stadium_names = [name for name in stadiums_info.keys() if name != 'date']
print(f"🎯 本日({jst_today}) 開催中のレース場：")
print(stadium_names)

# レース場名 → 場IDへの変換
# 修正: STADIUMS_MAPを正しく使用する
stadium_id_map = {}
for stadium_id, stadium_name in STADIUMS_MAP:
    stadium_id_map[stadium_name] = stadium_id
print(f"レース場IDマップ: {stadium_id_map}")

# ===== 3. 各レース場の詳細出走表を取得 =====
all_entries = []

for stadium_name in stadium_names:
    try:
        print(f"\n📍 {stadium_name} の処理を開始...")
        stadium_id = stadium_id_map.get(stadium_name)
        if stadium_id is None:
            print(f"⚠️ {stadium_name} のIDが見つかりません。スキップします。")
            continue

        # レース一覧とそれぞれの締切時刻を取得
        races = scraper.get_12races(d=today_date, stadium=stadium_id)
        print(f"📋 {stadium_name} (ID:{stadium_id}) のレース一覧: {races}")

        for race_key in races:
            try:
                if race_key in ['date', 'stadium']:
                    continue

                race_no = int(race_key.replace('R', ''))
                print(f"⛵ {stadium_name} - {race_key} を取得中...")

                # レース情報取得
                race_info = scraper.get_race_info(d=today_date, stadium=stadium_id, race=race_no)
                
                # 締切予定時刻とステータスの取得 (get_12racesの結果から)
                race_data = races.get(race_key, {})
                deadline_time = race_data.get('vote_limit')
                race_status = race_data.get('status')  # ステータス情報を取得
                
                # 時刻部分だけを抽出 (例: "2025-04-01 15:17:00" -> "15:17")
                if deadline_time and isinstance(deadline_time, str):
                    try:
                        deadline_dt = datetime.strptime(deadline_time, "%Y-%m-%d %H:%M:%S")
                        deadline_time = deadline_dt.strftime("%H:%M")
                    except Exception as e:
                        print(f"⚠️ 締切時刻のフォーマット変換でエラー: {e}")
                
                print(f"📅 締切予定時刻: {deadline_time}")
                print(f"🔄 レースステータス: {race_status}")

                for boat_no in range(1, 7):
                    boat_key = f'boat{boat_no}'
                    boat_data = race_info.get(boat_key, {})
                    # print(f"🚤 艇番: {boat_no}, データ: {boat_data}")

                    # 進入/ST情報の抽出（start_displayから）
                    course_info = None
                    st_time = None
                    start_display = race_info.get('start_display', {})
                    for course, info in start_display.items():
                        if info.get("boat") == boat_no:
                            course_info = course.replace("course", "")
                            st_time = info.get("ST")
                            break

                    # 修正: 正しいキー名を使用する
                    entry = {
                        "日付": today,
                        "レース場": stadium_name,
                        "レース番号": race_no,
                        "締切予定時刻": deadline_time,  # 締切予定時刻を追加
                        "ステータス": race_status,  # ステータスを追加
                        "艇番": boat_no,
                        "選手名": boat_data.get("name", ""),
                        "級別": boat_data.get("class", ""),
                        "支部": boat_data.get("branch", ""),
                        "全国勝率": boat_data.get("global_win_pt", None),  # 修正: win_rate → global_win_pt
                        "全国2連率": boat_data.get("global_in2nd", None),  # 追加: 全国2連率
                        "当地勝率": boat_data.get("local_win_pt", None),   # 修正: local_win_rate → local_win_pt
                        "当地2連率": boat_data.get("local_in2nd", None),   # 追加: 当地2連率
                        "モーター番号": boat_data.get("motor", None),      # 修正: motor_no → motor
                        "モーター2連率": boat_data.get("motor_in2nd", None), # 修正: motor_2rate → motor_in2nd
                        "ボート番号": boat_data.get("boat", None),         # 修正: boat_no → boat
                        "ボート2連率": boat_data.get("boat_in2nd", None),  # 修正: boat_2rate → boat_in2nd
                        "体重": boat_data.get("weight", None),
                        "年齢": boat_data.get("age", None),
                        "進入": course_info,
                    }
                    all_entries.append(entry)
            except Exception as e:
                print(f"⚠️ {stadium_name} {race_key} の処理でエラー: {e}")
                print(traceback.format_exc())
    except Exception as e:
        print(f"⚠️ {stadium_name} の全体処理でエラー: {e}")
        print(traceback.format_exc())

# ===== 4. DataFrame化 =====
print("\n📊 DataFrame作成中...")
df_today = pd.DataFrame(all_entries)

if not df_today.empty:
    print("✅ 取得成功！先頭5行を表示:")
    print(df_today.head())
else:
    print("❌ データが取得できませんでした")

# CSV出力
# JSTの日時をファイル名用にフォーマット (YYYYMMDDhhmm)
jst_timestamp = datetime.now(pytz.UTC).astimezone(jst).strftime("%Y%m%d%H%M")
csv_filename = f"races_{jst_timestamp}.csv"
base_dir = "/Users/yoshidont_mind/Library/Mobile Documents/com~apple~CloudDocs/Desktop/CST term4/COMP 4949 Big Data Analytics Methods/boat_race_prediction/data/race_info"
csv_path = f"{base_dir}/{csv_filename}"

df_today.to_csv(csv_path, index=False)
print(f"💾 CSVとして保存しました: {csv_path}")

# クローズ処理
scraper.close()
print("✅ スクレーパー終了")
