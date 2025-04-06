import pandas as pd
import pickle
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
from pathlib import Path
import pytz
from utils import preprocess_boatrace_dataframe
from pyjpboatrace import PyJPBoatrace
import traceback

# List of Boat Race Venue Codes
boatrace_venues = {
    '01': '桐生',
    '02': '戸田',
    '03': '江戸川',
    '04': '平和島',
    '05': '多摩川',
    '06': '浜名湖',
    '07': '蒲郡',
    '08': '常滑',
    '09': '津',
    '10': '三国',
    '11': 'びわこ',
    '12': '住之江',
    '13': '尼崎',
    '14': '鳴門',
    '15': '丸亀',
    '16': '児島',
    '17': '宮島',
    '18': '徳山',
    '19': '下関',
    '20': '若松',
    '21': '芦屋',
    '22': '福岡',
    '23': '唐津',
    '24': '大村'
}

# Get venue code from venue name (reverse lookup dictionary)
venue_to_code = {v: k for k, v in boatrace_venues.items()}

# Build URL for beforeinfo page
def build_beforeinfo_url(date, venue_code, race_number):
    base_url = "https://www.boatrace.jp/owpc/pc/race/beforeinfo"
    url = f"{base_url}?rno={race_number}&jcd={venue_code}&hd={date}"
    return url

# Build URL for odds info page
def build_odds_url(date, venue_code, race_number):
    base_url = "https://www.boatrace.jp/owpc/pc/race/oddstf"
    url = f"{base_url}?rno={race_number}&jcd={venue_code}&hd={date}"
    return url

# Function to fetch HTML
def fetch_html(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.text, datetime.now()
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to fetch HTML: {e}")
        return None, datetime.now()

# Function to scrape boat info
def scrape_boat_info(soup):
    boat_info = []
    
    try:
        # Get boat info
        tbody_elements = soup.select('tbody.is-fs12')
        
        for tbody in tbody_elements:
            boat_number_element = tbody.select_one('td.is-fs14')
            if boat_number_element:
                boat_number = boat_number_element.text.strip()
                
                # Get exhibition time (5th column from left)
                # Get the 5th td element from the first row of each boat's tbody
                first_row = tbody.select('tr')[0]
                td_elements = first_row.select('td')
                
                # Exhibition time (5th column from left)
                # rowspan is used, so we can't access it directly by index
                exhibition_time = "N/A"
                
                # 展示タイムは通常、rowspan="4"属性を持つtd要素
                for td in tbody.select('td[rowspan="4"]'):
                    # Find td elements containing only numbers (exhibition time)
                    if re.match(r'^\d+\.\d+$', td.text.strip()):
                        exhibition_time = td.text.strip()
                        break
                
                # Get entry info (2nd row, 2nd column)
                rows = tbody.select('tr')
                entry = "N/A"
                if len(rows) > 1:
                    entry_cells = rows[1].select('td')
                    if len(entry_cells) > 1:
                        entry = entry_cells[1].text.strip()
                
                boat_info.append({
                    '艇番': boat_number,  # Boat Number
                    '展示タイム': exhibition_time,  # Exhibition Time
                    '進入': entry,  # Entry
                    '単勝オッズ': "N/A"  # オッズ情報は別途取得  # Win Odds (odds info retrieved separately)
                })
    except Exception as e:
        print(f"Error scraping boat info: {e}")
    
    # If no data is retrieved, create empty data
    if not boat_info:
        for i in range(1, 7):
            boat_info.append({
                '艇番': str(i),  # Boat Number
                '展示タイム': "N/A",  # Exhibition Time
                '進入': "N/A",  # Entry
                '単勝オッズ': "N/A"  # Win Odds
            })
    
    # Sort by boat number
    boat_info.sort(key=lambda x: int(x['艇番']))
    
    return boat_info

# Function to scrape single odds info
def scrape_odds_info(soup):
    odds_info = {}
    
    try:
        # Find single odds table
        odds_tables = soup.find_all('table')
        
        for table in odds_tables:
            # Check if the table header contains "Single Odds"
            header = table.find_previous('div', string=lambda text: text and '単勝オッズ' in text if text else False)
            if header or table.find('th', string=lambda text: text and '単勝オッズ' in text if text else False):
                # Process each row
                rows = table.find_all('tr')
                for row in rows:
                    # First cell is boat number, second cell is odds value
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        # Get boat number (only numbers)
                        boat_number_cell = cells[0]
                        boat_number = boat_number_cell.text.strip()
                        
                        # Extract only numbers
                        if boat_number.isdigit() and 1 <= int(boat_number) <= 6:
                            # Get odds value
                            odds_value = cells[-1].text.strip()  # Last cell is odds value
                            odds_info[boat_number] = odds_value
        
        # Try another method (if the table structure is different)
        if not odds_info:
            # Use the class name of the boat number to find the specific
            for i in range(1, 7):
                # Find the cell with the class name corresponding to the color of each boat
                boat_cell = soup.select_one(f'td.is-boatColor{i}')
                if boat_cell:
                    # Find the cell containing the odds value in the same row
                    row = boat_cell.parent
                    if row:
                        # The last cell is often the odds value
                        odds_cell = row.select_one('td:last-child')
                        if odds_cell:
                            odds_value = odds_cell.text.strip()
                            # Check if the odds value is in numeric format
                            if re.match(r'^\d+(\.\d+)?$', odds_value):
                                odds_info[str(i)] = odds_value
    
    except Exception as e:
        print(f"Error scraping odds info: {e}")
    
    return odds_info

# Function to scrape weather info
def scrape_weather_info(soup):
    weather_info = {
        '天候': 'N/A',  # Weather Condition
        '風向': 'N/A',  # Wind Direction
        '風量': 'N/A',  # Wind Speed
        '波': 'N/A'  # Wave Height
    }
    
    try:
        # Get weather info
        weather_section = soup.select_one('.weather1')
        if weather_section:
            # Weather (rain, cloudy, sunny, etc.)
            weather_unit = weather_section.select_one('.weather1_bodyUnit.is-weather')
            if weather_unit:
                weather_label = weather_unit.select_one('.weather1_bodyUnitLabelTitle')
                if weather_label:
                    weather_info['天候'] = weather_label.text.strip()
            
            # Wind direction
            wind_direction_element = weather_section.select_one('.weather1_bodyUnit.is-windDirection .weather1_bodyUnitImage')
            if wind_direction_element:
                # Extract wind direction from class name (e.g. is-wind14 → 14)
                wind_direction_class = wind_direction_element.get('class', [])
                wind_direction = next((cls.replace('is-wind', '') for cls in wind_direction_class if cls.startswith('is-wind')), "N/A")
                weather_info['風向'] = wind_direction
            
            # Wind speed
            wind_speed_element = weather_section.select_one('.weather1_bodyUnit.is-wind .weather1_bodyUnitLabelData')
            if wind_speed_element:
                weather_info['風量'] = wind_speed_element.text.strip()
            
            # Wave height
            wave_element = weather_section.select_one('.weather1_bodyUnit.is-wave .weather1_bodyUnitLabelData')
            if wave_element:
                weather_info['波'] = wave_element.text.strip()
    except Exception as e:
        print(f"Error scraping weather info: {e}")
    
    return weather_info

# Function to predict single race
def predict_single_race(race_id):
    try:
        date_str = race_id[:8]  # yyyymmdd
        stadium_id = int(race_id[8:10])
        race_no = int(race_id[10:])
        target_date = datetime.strptime(date_str, "%Y%m%d").date()
        stadium_name = boatrace_venues.get(f"{stadium_id:02}", "不明")

        # Set JST timezone
        jst = pytz.timezone('Asia/Tokyo')
        
        # ====== Get race basic info ======
        scraper = PyJPBoatrace()
        race_info = scraper.get_race_info(d=target_date, stadium=stadium_id, race=race_no)

        # ====== Get before info and odds (scraping) ======
        # Build URL for beforeinfo page
        venue_code = f"{stadium_id:02}"
        beforeinfo_url = build_beforeinfo_url(date_str, venue_code, race_no)
        odds_url = build_odds_url(date_str, venue_code, race_no)
        
        # Get HTML
        beforeinfo_html, _ = fetch_html(beforeinfo_url)
        odds_html, prediction_time = fetch_html(odds_url)
        
        # If HTML is not retrieved, raise an error
        if not beforeinfo_html or not odds_html:
            raise Exception("Failed to retrieve before info or odds info")
        
        # Parse HTML with BeautifulSoup
        beforeinfo_soup = BeautifulSoup(beforeinfo_html, 'html.parser')
        odds_soup = BeautifulSoup(odds_html, 'html.parser')
        
        # Get before info for each boat
        boat_info = scrape_boat_info(beforeinfo_soup)
        
        # Get single odds info
        odds_info = scrape_odds_info(odds_soup)
        
        # Integrate single odds info into boat_info
        for boat in boat_info:
            boat_number = boat['艇番']
            if boat_number in odds_info:
                boat['単勝オッズ'] = odds_info[boat_number]
        
        # Get weather info
        weather_info = scrape_weather_info(beforeinfo_soup)
        
        # ====== Convert race basic info to DataFrame ======
        race_entries = []
        for boat_no in range(1, 7):
            boat_key = f"boat{boat_no}"
            boat_data = race_info.get(boat_key, {})
            
            # Get the corresponding boat info from boat_info
            boat_info_item = next((b for b in boat_info if int(b['艇番']) == boat_no), {})
            
            race_entries.append({
                "レースID": race_id,  # Race ID
                "日付": target_date,  # Date
                "会場": stadium_name, # Add venue name  # Stadium/Venue
                "艇番": boat_no,  # Boat Number
                "選手名": boat_data.get("name", ""),  # Racer Name
                "級別": boat_data.get("class", ""),  # Class
                "支部": boat_data.get("branch", ""),  # Branch
                "全国勝率": boat_data.get("global_win_pt", None),  # National Win Rate
                "全国2連率": boat_data.get("global_in2nd", None), # Add national 2-win rate  # National 2nd Place Rate
                "当地勝率": boat_data.get("local_win_pt", None),  # Local Win Rate
                "当地2連率": boat_data.get("local_in2nd", None), # Add local 2-win rate  # Local 2nd Place Rate
                "モーター番号": boat_data.get("motor", None),  # Motor Number
                "モーター2連率": boat_data.get("motor_in2nd", None),  # Motor 2nd Place Rate
                "ボート番号": boat_data.get("boat", None),  # Boat Number
                "ボート2連率": boat_data.get("boat_in2nd", None),  # Boat 2nd Place Rate
                "展示タイム": boat_info_item.get('展示タイム', None),  # Exhibition Time
                "進入": boat_info_item.get('進入', None),  # Entry
                "単勝オッズ": float(boat_info_item.get('単勝オッズ', 0)) if boat_info_item.get('単勝オッズ', 'N/A') != 'N/A' else None,  # Win Odds
                "体重": boat_data.get("weight", None),  # Weight
                "年齢": boat_data.get("age", None),  # Age
            })

        df_race = pd.DataFrame(race_entries)
        
        # Add weather info to each row
        for key, value in weather_info.items():
            df_race[key] = value
            
        # Convert "風量" to numeric type (e.g. "3m" → 3)
        if "風量" in df_race.columns:  # Wind Speed
            df_race["風量"] = df_race["風量"].astype(str).str.replace('m', '').str.replace('M', '')
            df_race["風量"] = pd.to_numeric(df_race["風量"], errors="coerce")
        
        # Convert "波" to numeric type (e.g. "3cm" → 3)
        if "波" in df_race.columns:  # Wave Height
            df_race["波"] = df_race["波"].astype(str).str.replace('cm', '').str.replace('CM', '')
            df_race["波"] = pd.to_numeric(df_race["波"], errors="coerce")

        # Check for missing important data
        for col in ["全国勝率", "全国2連率", "当地勝率", "当地2連率", "モーター2連率", "ボート2連率"]:  # National Win Rate, National 2nd Place Rate, Local Win Rate, Local 2nd Place Rate, Motor 2nd Place Rate, Boat 2nd Place Rate
            if col not in df_race.columns or df_race[col].isnull().all():
                print(f"Missing data for important feature '{col}'")
                if col not in df_race.columns:
                    # Add feature if it doesn't exist
                    df_race[col] = 0
                    print(f"'{col}' added (value: 0)")
        
        # If date info is missing, add it
        if "日付" in df_race.columns:  # Date
            if not pd.api.types.is_datetime64_any_dtype(df_race["日付"]):
                df_race["日付"] = pd.to_datetime(df_race["日付"], errors="coerce")
        else:
            # Set current date
            df_race["日付"] = pd.to_datetime(target_date)
        
        print("Starting preprocessing...")
        # ====== Preprocessing ======
        df_processed = preprocess_boatrace_dataframe(df_race.copy())
        print("Preprocessing completed")

        # ====== Load model ======
        try:
            # Select the latest model before the race date
            race_date = datetime.strptime(date_str, "%Y%m%d").date()
            model_files = sorted(list(Path("models").glob("model_*.pkl")))
            
            # Extract date from model file name and select the latest model before the race date
            valid_models = []
            for model_file in model_files:
                try:
                    # Extract YYYYMMDD part from file name
                    date_part = model_file.stem.split('_')[1]
                    model_date = datetime.strptime(date_part, "%Y%m%d").date()
                    
                    # Only use models before the race date
                    if model_date < race_date:
                        valid_models.append((model_date, model_file))
                except (IndexError, ValueError):
                    pass
            
            if not valid_models:
                raise Exception(f"No model found for dates before race date ({race_date})")
            
            # Sort by date in descending order and get the latest model
            valid_models.sort(reverse=True)
            latest_model_date, model_path = valid_models[0]
            print(f"Using model from {latest_model_date} for race date ({race_date}): {model_path}")
            
            # Try to load LightGBM directly instead of pickle
            import lightgbm as lgb
            model = lgb.Booster(model_file=str(model_path))
            print("Model loaded (LightGBM direct read)")
        except Exception as e:
            print(f"Failed to load model (LightGBM): {e}")
            print("Loading model as pickle file as backup...")
            
            try:
                # Specify protocol for reading old pickle files
                with open(model_path, "rb") as f:
                    model = pickle.load(f, encoding='latin1')
                print("Model loaded (pickle)")
            except Exception as e:
                print(f"Failed to load model: {e}")
                raise Exception(f"Failed to load model file: {e}")

        # ====== Prediction ======
        # Prepare features in the same order as the model's features
        feature_cols = [
            "支部", "級別", "艇番", "会場", "風量", "波", "月", "曜日",  # Branch, Class, Boat Number, Stadium, Wind Speed, Wave Height, Month, Day of Week
            "全国勝率_dev", "全国2連率_dev", "当地勝率_dev", "当地2連率_dev",  # National Win Rate (dev), National 2nd Place Rate (dev), Local Win Rate (dev), Local 2nd Place Rate (dev)
            "モーター2連率_dev", "ボート2連率_dev", "展示タイム_dev"  # Motor 2nd Place Rate (dev), Boat 2nd Place Rate (dev), Exhibition Time (dev)
        ]

        # Check if each feature exists
        print("Checking features...")
        for feature in feature_cols:
            if feature not in df_processed.columns:
                print(f"Warning: Feature '{feature}' does not exist in the dataframe")
                # If feature is missing, fill with an appropriate value
                if feature.endswith('_dev'):
                    df_processed[feature] = 50  # Median value for deviation score
                else:
                    df_processed[feature] = 0
                print(f"Feature '{feature}' added")

        # Prediction (LightGBM uses predict() instead of predict_proba())
        print("Running prediction...")
        try:
            # If using Booster object, use predict()
            df_processed["pred_proba"] = model.predict(df_processed[feature_cols])
            print("Prediction completed")
        except Exception as e:
            print(f"Error during prediction: {e}")
            raise

        # Normalize win rate (total to 1)
        print("Normalizing win rate...")
        sum_proba = df_processed["pred_proba"].sum()
        if sum_proba > 0:
            df_processed["pred_proba"] = df_processed["pred_proba"] / sum_proba
        print("Normalization of win rate completed")

        # Calculate expected value
        df_processed["期待値"] = df_processed["pred_proba"] * df_processed["単勝オッズ"]  # Expected Value

        # Create result dataframe
        df_result = df_processed[["艇番", "選手名", "pred_proba", "単勝オッズ", "期待値"]].sort_values("pred_proba", ascending=False)  # Boat Number, Racer Name, Prediction Probability, Win Odds, Expected Value
        df_result.columns = ["艇番", "選手名", "勝率(予測)", "単勝オッズ", "期待値"]  # Boat Number, Racer Name, Win Rate (Predicted), Win Odds, Expected Value

        # Convert win rate to %
        df_result["勝率(予測)"] = (df_result["勝率(予測)"] * 100).round(1).astype(str) + '%'  # Win Rate (Predicted)
        
        # Make expected value more readable
        df_result["期待値"] = df_result["期待値"].round(2)  # Expected Value

        # Round single odds to 1 decimal place
        df_result["単勝オッズ"] = df_result["単勝オッズ"].round(1)  # Win Odds

        # Convert prediction time to JST
        prediction_time_jst = prediction_time.astimezone(jst) if prediction_time.tzinfo else jst.localize(prediction_time)

        return df_result, prediction_time_jst

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return None, datetime.now()
