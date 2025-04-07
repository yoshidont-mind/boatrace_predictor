import pandas as pd
from pyjpboatrace import PyJPBoatrace
from datetime import datetime, date
from pyjpboatrace.const import STADIUMS_MAP
import traceback
import pytz

# ===== 1. Initialize Scraper =====
print("Initializing scraper...")
scraper = PyJPBoatrace()
# Use system time for scraping
today_datetime = datetime.today()
today = today_datetime.strftime("%Y-%m-%d")
today_date = date(today_datetime.year, today_datetime.month, today_datetime.day)

# Get JST time for display
jst = pytz.timezone('Asia/Tokyo')
jst_now = datetime.now(pytz.UTC).astimezone(jst)
jst_today = jst_now.strftime("%Y-%m-%d")

print(f"System date: {today_date}")
print(f"JST date: {jst_today}")

# ===== 2. Get list of racecourses holding races today =====
print("Retrieving list of racecourses...")
stadiums_info = scraper.get_stadiums(d=today_date)
stadium_names = [name for name in stadiums_info.keys() if name != 'date']
print(f"🎯 Racecourses holding races today ({jst_today}):")
print(stadium_names)

# Convert racecourse name to ID
# Fix: Use STADIUMS_MAP correctly
stadium_id_map = {}
for stadium_id, stadium_name in STADIUMS_MAP:
    stadium_id_map[stadium_name] = stadium_id
print(f"Racecourse ID map: {stadium_id_map}")

# ===== 3. Retrieve detailed entry list for each racecourse =====
all_entries = []

for stadium_name in stadium_names:
    try:
        print(f"\n📍 Processing {stadium_name}...")
        stadium_id = stadium_id_map.get(stadium_name)
        if stadium_id is None:
            print(f"⚠️ ID for {stadium_name} not found. Skipping.")
            continue

        # Retrieve list of races and their respective closing times
        races = scraper.get_12races(d=today_date, stadium=stadium_id)
        print(f"📋 List of races for {stadium_name} (ID:{stadium_id}): {races}")

        for race_key in races:
            try:
                if race_key in ['date', 'stadium']:
                    continue

                race_no = int(race_key.replace('R', ''))
                print(f"⛵ Retrieving {stadium_name} - {race_key}...")

                # Retrieve race information
                race_info = scraper.get_race_info(d=today_date, stadium=stadium_id, race=race_no)
                
                # Retrieve closing time and status from get_12races result
                race_data = races.get(race_key, {})
                deadline_time = race_data.get('vote_limit')
                race_status = race_data.get('status')  # Retrieve status information
                
                # Extract only the time part (e.g., "2025-04-01 15:17:00" -> "15:17")
                if deadline_time and isinstance(deadline_time, str):
                    try:
                        deadline_dt = datetime.strptime(deadline_time, "%Y-%m-%d %H:%M:%S")
                        deadline_time = deadline_dt.strftime("%H:%M")
                    except Exception as e:
                        print(f"⚠️ Error converting closing time format: {e}")
                
                print(f"📅 Scheduled closing time: {deadline_time}")
                print(f"🔄 Race status: {race_status}")

                for boat_no in range(1, 7):
                    boat_key = f'boat{boat_no}'
                    boat_data = race_info.get(boat_key, {})
                    # print(f"🚤 Boat number: {boat_no}, Data: {boat_data}")

                    # Extract entry/ST information from start_display
                    course_info = None
                    st_time = None
                    start_display = race_info.get('start_display', {})
                    for course, info in start_display.items():
                        if info.get("boat") == boat_no:
                            course_info = course.replace("course", "")
                            st_time = info.get("ST")
                            break

                    # Fix: Use correct key names
                    entry = {
                        "日付": today,
                        "レース場": stadium_name,
                        "レース番号": race_no,
                        "締切予定時刻": deadline_time,  # Add scheduled closing time
                        "ステータス": race_status,  # Add status
                        "艇番": boat_no,
                        "選手名": boat_data.get("name", ""),
                        "級別": boat_data.get("class", ""),
                        "支部": boat_data.get("branch", ""),
                        "全国勝率": boat_data.get("global_win_pt", None),
                        "全国2連率": boat_data.get("global_in2nd", None),  
                        "当地勝率": boat_data.get("local_win_pt", None),  
                        "当地2連率": boat_data.get("local_in2nd", None),  
                        "モーター番号": boat_data.get("motor", None),      
                        "モーター2連率": boat_data.get("motor_in2nd", None), 
                        "ボート番号": boat_data.get("boat", None),         
                        "ボート2連率": boat_data.get("boat_in2nd", None),  
                        "体重": boat_data.get("weight", None),
                        "年齢": boat_data.get("age", None),
                        "進入": course_info,
                    }
                    all_entries.append(entry)
            except Exception as e:
                print(f"⚠️ Error processing {stadium_name} {race_key}: {e}")
                print(traceback.format_exc())
    except Exception as e:
        print(f"⚠️ Error processing {stadium_name}: {e}")
        print(traceback.format_exc())

# ===== 4. Convert to DataFrame =====
print("\n📊 Creating DataFrame...")
df_today = pd.DataFrame(all_entries)

if not df_today.empty:
    print("✅ Data retrieval successful! Displaying first 5 rows:")
    print(df_today.head())
else:
    print("❌ No data retrieved")

# Output to CSV
# Format JST datetime for filename (YYYYMMDDhhmm)
jst_timestamp = datetime.now(pytz.UTC).astimezone(jst).strftime("%Y%m%d%H%M")
csv_filename = f"races_{jst_timestamp}.csv"
base_dir = "/Users/yoshidont_mind/Library/Mobile Documents/com~apple~CloudDocs/Desktop/CST term4/COMP 4949 Big Data Analytics Methods/boat_race_prediction/data/race_info"
csv_path = f"{base_dir}/{csv_filename}"

df_today.to_csv(csv_path, index=False)
print(f"💾 Saved as CSV: {csv_path}")

# Close scraper
scraper.close()
print("✅ Scraper closed")
