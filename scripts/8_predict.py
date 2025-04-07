#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import requests
from bs4 import BeautifulSoup
import re
import os
import pandas as pd
import numpy as np
import glob
import lightgbm as lgb
import time
import pytz
from datetime import datetime

# Boat race venue code list
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

# Function to get venue code from venue name
def get_venue_code(venue_name):
    # Create reverse dictionary
    venue_to_code = {v: k for k, v in boatrace_venues.items()}
    
    # Get code from venue name
    if venue_name in venue_to_code:
        return venue_to_code[venue_name]
    else:
        print(f"Error: Venue '{venue_name}' not found.")
        print("Valid venue names: " + ", ".join(boatrace_venues.values()))
        sys.exit(1)

# Function to build URL for before info page
def build_beforeinfo_url(date, venue_code, race_number):
    base_url = "https://www.boatrace.jp/owpc/pc/race/beforeinfo"
    url = f"{base_url}?rno={race_number}&jcd={venue_code}&hd={date}"
    return url

# Function to build URL for odds info page
def build_odds_url(date, venue_code, race_number):
    base_url = "https://www.boatrace.jp/owpc/pc/race/oddstf"
    url = f"{base_url}?rno={race_number}&jcd={venue_code}&hd={date}"
    return url

# Function to fetch HTML
def fetch_html(url, use_local_file=None):
    # Get current time in JST
    jst = pytz.timezone('Asia/Tokyo')
    fetch_time = datetime.now(jst)
    
    if use_local_file and os.path.exists(use_local_file):
        print(f"Loading HTML from local file {use_local_file}")
        try:
            with open(use_local_file, 'r', encoding='utf-8') as f:
                return f.read(), fetch_time
        except Exception as e:
            print(f"Failed to load local file: {e}")
            print("Fetching HTML from URL")
    
    try:
        print(f"Fetching HTML from URL: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise exception for errors
        return response.text, fetch_time
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to fetch HTML: {e}")
        sys.exit(1)

# Function to scrape boat info
def scrape_boat_info(soup):
    boat_info = []
    
    try:
        # Get info for each boat
        tbody_elements = soup.select('tbody.is-fs12')
        
        for tbody in tbody_elements:
            boat_number_element = tbody.select_one('td.is-fs14')
            if boat_number_element:
                boat_number = boat_number_element.text.strip()
                
                # Get exhibition time (5th column from the left)
                # First row of each boat's tbody, 5th td element
                first_row = tbody.select('tr')[0]
                td_elements = first_row.select('td')
                
                # Exhibition time is after boat number, photo, player name, and weight
                # rowspan is used, so direct index access may not work
                exhibition_time = "N/A"
                
                # Exhibition time is usually in td element with rowspan="4"
                for td in tbody.select('td[rowspan="4"]'):
                    # Find td element with only numeric value (exhibition time)
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
                    '単勝オッズ': "N/A"  # Win Odds
                })
    except Exception as e:
        print(f"Error occurred while scraping boat info: {e}")
        # Continue processing even if an error occurs
    
    # Create empty data if no data was obtained
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

# Function to scrape odds info
def scrape_odds_info(soup):
    odds_info = {}
    
    try:
        # Find table with win odds
        # Look for table after heading "単勝オッズ"
        odds_tables = soup.find_all('table')
        
        for table in odds_tables:
            # Check if table header contains "単勝オッズ"
            header = table.find_previous('div', string=lambda text: text and '単勝オッズ' in text if text else False)
            if header or table.find('th', string=lambda text: text and '単勝オッズ' in text if text else False):
                # Process each row
                rows = table.find_all('tr')
                for row in rows:
                    # First cell is boat number, second cell is odds value
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        # Get boat number (numeric only)
                        boat_number_cell = cells[0]
                        boat_number = boat_number_cell.text.strip()
                        
                        # Extract numeric value only
                        if boat_number.isdigit() and 1 <= int(boat_number) <= 6:
                            # Get odds value
                            odds_value = cells[-1].text.strip()  # Last cell is odds value
                            odds_info[boat_number] = odds_value
        
        # Try another method (if table structure is different)
        if not odds_info:
            # Directly find win odds table
            for table in odds_tables:
                # Check first row of table
                first_row = table.find('tr')
                if first_row:
                    # Check if there is a column header "単勝オッズ"
                    headers = first_row.find_all('th')
                    for i, header in enumerate(headers):
                        if header.text.strip() == '単勝オッズ':
                            # Process each row
                            for row in table.find_all('tr')[1:]:  # Skip header row
                                cells = row.find_all('td')
                                if len(cells) > i:
                                    # Get boat number
                                    boat_number = cells[0].text.strip()
                                    if boat_number.isdigit() and 1 <= int(boat_number) <= 6:
                                        # Get odds value
                                        odds_value = cells[i].text.strip()
                                        odds_info[boat_number] = odds_value
        
        # Try another method (using class names)
        if not odds_info:
            # Identify by class name corresponding to each boat color
            for i in range(1, 7):
                # Find cell with class name corresponding to each boat color
                boat_cell = soup.select_one(f'td.is-boatColor{i}')
                if boat_cell:
                    # Find cell with odds value in the same row
                    row = boat_cell.parent
                    if row:
                        # Last cell is often the odds value
                        odds_cell = row.select_one('td:last-child')
                        if odds_cell:
                            odds_value = odds_cell.text.strip()
                            # Check if it is a numeric format
                            if re.match(r'^\d+(\.\d+)?$', odds_value):
                                odds_info[str(i)] = odds_value
    
    except Exception as e:
        print(f"Error occurred while scraping odds info: {e}")
    
    return odds_info

# Function to scrape weather info
def scrape_weather_info(soup):
    weather_info = {
        '天候': 'N/A',  # Weather
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
                    weather_info['天候'] = weather_label.text.strip()  # Weather
            
            # Wind direction
            wind_direction_element = weather_section.select_one('.weather1_bodyUnit.is-windDirection .weather1_bodyUnitImage')
            if wind_direction_element:
                # Extract wind direction from class name (e.g., is-wind14 → 14)
                wind_direction_class = wind_direction_element.get('class', [])
                wind_direction = next((cls.replace('is-wind', '') for cls in wind_direction_class if cls.startswith('is-wind')), "N/A")
                weather_info['風向'] = wind_direction  # Wind Direction
            
            # Wind speed
            wind_speed_element = weather_section.select_one('.weather1_bodyUnit.is-wind .weather1_bodyUnitLabelData')
            if wind_speed_element:
                weather_info['風量'] = wind_speed_element.text.strip()  # Wind Speed
            
            # Wave height
            wave_element = weather_section.select_one('.weather1_bodyUnit.is-wave .weather1_bodyUnitLabelData')
            if wave_element:
                weather_info['波'] = wave_element.text.strip()  # Wave Height
    except Exception as e:
        print(f"Error occurred while scraping weather info: {e}")
        # Continue processing even if an error occurs
    
    return weather_info

# Function to get the latest race info file
def get_latest_race_info_file():
    data_dir = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/Desktop/CST term4/COMP 4949 Big Data Analytics Methods/boat_race_prediction/data/race_info")
    race_files = glob.glob(os.path.join(data_dir, "races_*.csv"))
    
    if not race_files:
        print("Error: No race info files found.")
        sys.exit(1)
    
    # Select the latest file by filename date
    latest_file = max(race_files)
    print(f"Latest race info file: {os.path.basename(latest_file)}")
    return latest_file

# Function to get the latest model file
def get_latest_model_file():
    model_dir = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/Desktop/CST term4/COMP 4949 Big Data Analytics Methods/boat_race_prediction/models")
    model_files = glob.glob(os.path.join(model_dir, "model_*.pkl"))
    
    if not model_files:
        print("Error: No model files found.")
        sys.exit(1)
    
    # Select the latest file by filename date
    latest_file = max(model_files)
    print(f"Latest model file: {os.path.basename(latest_file)}")
    return latest_file

# Function to get race base info
def get_race_base_info(date, venue_name, race_number):
    # Get the latest race info file
    race_info_file = get_latest_race_info_file()
    
    # Read the CSV file
    try:
        race_info_df = pd.read_csv(race_info_file)
    except Exception as e:
        print(f"Failed to read race info file: {e}")
        sys.exit(1)
    
    # Convert date to unified format (2025/04/02 or 2025-04-02 → 20250402)
    if '/' in race_info_df['日付'].iloc[0]:
        race_info_df['日付'] = race_info_df['日付'].str.replace('/', '')  # Date
    elif '-' in race_info_df['日付'].iloc[0]:
        race_info_df['日付'] = race_info_df['日付'].str.replace('-', '')  # Date
    
    # Filter by specified conditions
    filtered_df = race_info_df[
        (race_info_df['日付'] == date) &  # Date
        (race_info_df['レース場'] == venue_name) &  # Venue
        (race_info_df['レース番号'] == int(race_number))  # Race Number
    ]
    
    if filtered_df.empty:
        print(f"Error: No info found for specified race ({date}, {venue_name}, {race_number}).")
        sys.exit(1)
    
    return filtered_df

# Function to calculate deviation score (comparison within race)
def calculate_deviation_score(series):
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series([50] * len(series), index=series.index)  # Fix to 50 if all are the same
    return 10 * (series - mean) / std + 50

# Function to preprocess data
def preprocess_data(merged_df):
    df = merged_df.copy()
    print(f"Starting preprocessing. Data size: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Convert to numeric type (percentage, etc.)
    rate_columns = ["全国勝率", "全国2連率", "当地勝率", "当地2連率", "ボート2連率", "モーター2連率"]
    for col in rate_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Convert "単勝オッズ" to numeric type if it exists
    if "単勝オッズ" in df.columns:
        df["単勝オッズ"] = pd.to_numeric(df["単勝オッズ"], errors="coerce")  # Win Odds
    
    # Convert "展示タイム" to numeric type
    if "展示タイム" in df.columns:
        df["展示タイム"] = pd.to_numeric(df["展示タイム"], errors="coerce")  # Exhibition Time
        
    # Convert "風量" to numeric type ("3m" → 3)
    if "風量" in df.columns:
        df["風量"] = df["風量"].astype(str).str.replace('m', '').str.replace('M', '')
        df["風量"] = pd.to_numeric(df["風量"], errors="coerce")  # Wind Speed
    
    # Convert "波" to numeric type ("3cm" → 3)
    if "波" in df.columns:
        df["波"] = df["波"].astype(str).str.replace('cm', '').str.replace('CM', '')
        df["波"] = pd.to_numeric(df["波"], errors="coerce")  # Wave Height
    
    # Convert "会場" from categorical to numeric type
    if "会場" in df.columns:
        # Skip if already numeric
        if not pd.api.types.is_numeric_dtype(df["会場"]):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df["会場"] = le.fit_transform(df["会場"].astype(str))  # Venue
    else:
        # Generate "会場" from "レース場" if "会場" is missing but "レース場" exists
        if "レース場" in df.columns:
            df["会場"] = df["レース場"]  # Venue
            print("Generated '会場' from 'レース場'")
            if not pd.api.types.is_numeric_dtype(df["会場"]):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df["会場"] = le.fit_transform(df["会場"].astype(str))  # Venue
    
    # Date processing
    if "日付" in df.columns:
        df["日付"] = pd.to_datetime(df["日付"], format="%Y%m%d", errors="coerce")  # Date
        df["月"] = df["日付"].dt.month  # Month
        df["曜日"] = df["日付"].dt.weekday  # Day of the Week
    
    # Encoding categorical columns (Label Encoding is OK)
    from sklearn.preprocessing import LabelEncoder
    category_cols = ["支部", "級別", "天候", "風向"]
    for col in category_cols:
        if col in df.columns:
            # Skip if already numeric and integer type
            if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_integer_dtype(df[col]):
                continue
            df[col] = df[col].astype(str).fillna("不明")  # Unknown
            df[col] = LabelEncoder().fit_transform(df[col])
    
    # Fill missing values (use median)
    median_values = {
        "年齢": 35,  # Age
        "体重": 53.0,  # Weight
        "全国勝率": 5.50,  # Global Win Rate
        "全国2連率": 30.0,  # Global In 2nd Rate
        "当地勝率": 5.50,  # Local Win Rate
        "当地2連率": 30.0,  # Local In 2nd Rate
        "モーター2連率": 30.0,  # Motor In 2nd Rate
        "ボート2連率": 30.0,  # Boat In 2nd Rate
        "展示タイム": 6.70  # Exhibition Time
    }
    
    for col, median_val in median_values.items():
        if col in df.columns:
            df[col].fillna(median_val, inplace=True)
    
    # Columns to be standardized (compared within race)
    score_cols = ["全国勝率", "全国2連率", "当地勝率", "当地2連率", "モーター2連率", "ボート2連率", "展示タイム"]
    
    # Standardize within race
    for col in score_cols:
        if col in df.columns:
            new_col = f"{col}_dev"
            # Fill missing values with 0 before grouping (to prevent skipping calculation)
            df[col] = df[col].fillna(0)
            df[new_col] = df.groupby("レースID")[col].transform(calculate_deviation_score)  # Race ID
            
    # Explicitly check for existence and create if missing
    required_cols = ["風量", "波"]
    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: Feature '{col}' not found in DataFrame. Filling with 0.")
            df[col] = 0

    # Force recalculation of standardized features
    score_cols = ["全国勝率", "全国2連率", "当地勝率", "当地2連率", "モーター2連率", "ボート2連率", "展示タイム"]
    for base_col in score_cols:
        dev_col = f"{base_col}_dev"
        if base_col not in df.columns:
            print(f"Feature '{dev_col}' missing original feature '{base_col}', filling with 50")
            df[dev_col] = 50  # Median of standardized score
    
    print(f"Preprocessing completed. Data size after processing: {df.shape[0]} rows × {df.shape[1]} columns")
    return df

# Function to predict race result using model
def predict_race_result(preprocessed_df):
    # Get the latest model file
    model_file = get_latest_model_file()
    
    # Load the model
    try:
        model = lgb.Booster(model_file=model_file)
    except Exception as e:
        print(f"Failed to load model file: {e}")
        sys.exit(1)
    
    # Prepare features in the same order as the model
    feature_names = model.feature_name()
    
    # Check for feature existence and fill missing values
    for feature in feature_names:
        if feature not in preprocessed_df.columns:
            print(f"Warning: Feature '{feature}' not found in DataFrame. Filling with 0.")
            preprocessed_df[feature] = 0
    
    # Extract features for prediction
    X_pred = preprocessed_df[feature_names]
    
    # Execute prediction
    probabilities = model.predict(X_pred)
    
    # Normalize win rates (sum to 1)
    normalized_probs = probabilities / probabilities.sum()
    
    # Combine boat number and prediction results
    results = pd.DataFrame({
        '艇番': preprocessed_df['艇番'],  # Boat Number
        '勝率予測': normalized_probs,  # Win Probability
        '単勝オッズ': preprocessed_df['単勝オッズ']  # Win Odds
    })
    
    # Calculate expected value (win rate prediction * odds)
    results['期待値'] = results['勝率予測'] * results['単勝オッズ']  # Expected Value
    
    # Sort by win rate prediction in descending order
    results = results.sort_values('勝率予測', ascending=False)
    
    return results

def main():
    try:
        # Parse command line arguments
        if len(sys.argv) != 4:
            print("Usage: python 8_predict.py <date(YYYYMMDD)> <venue name> <race number>")
            print("Example: python 8_predict.py 20250402 Kiryu 12")
            sys.exit(1)
        
        date = sys.argv[1]
        venue_name = sys.argv[2]
        race_number = sys.argv[3]
        
        # Validate date format
        if not re.match(r'^\d{8}$', date):
            print("Error: Date must be specified as an 8-digit number (YYYYMMDD).")
            sys.exit(1)
        
        # Validate race number
        try:
            race_number = int(race_number)
            if race_number < 1 or race_number > 12:
                print("Error: Race number must be between 1 and 12.")
                sys.exit(1)
        except ValueError:
            print("Error: Race number must be a number.")
            sys.exit(1)
        
        # Get venue code
        venue_code = get_venue_code(venue_name)
        
        # Build URL for before info page
        beforeinfo_url = build_beforeinfo_url(date, venue_code, race_number)
        
        # Build URL for odds info page
        odds_url = build_odds_url(date, venue_code, race_number)
        
        # Path to sample HTML file (for testing)
        beforeinfo_html_path = "/home/ubuntu/upload/beforeinfo.html"
        odds_html_path = "/home/ubuntu/upload/odds.html"
        
        # Fetch before info HTML
        beforeinfo_html, _ = fetch_html(beforeinfo_url, use_local_file=beforeinfo_html_path if os.path.exists(beforeinfo_html_path) else None)
        
        # Fetch odds info HTML (use this time as prediction time)
        odds_html, prediction_time = fetch_html(odds_url, use_local_file=odds_html_path if os.path.exists(odds_html_path) else None)
        
        # Parse HTML with BeautifulSoup
        beforeinfo_soup = BeautifulSoup(beforeinfo_html, 'html.parser')
        odds_soup = BeautifulSoup(odds_html, 'html.parser')
        
        # Get before info for each boat
        boat_info = scrape_boat_info(beforeinfo_soup)
        
        # Get win odds information
        odds_info = scrape_odds_info(odds_soup)
        
        # Integrate win odds information into boat_info
        for boat in boat_info:
            boat_number = boat['艇番']  # Boat Number
            if boat_number in odds_info:
                boat['単勝オッズ'] = odds_info[boat_number]  # Win Odds
        
        # Get weather information
        weather_info = scrape_weather_info(beforeinfo_soup)
        
        print("\n【Fetched Before Info】")
        print(f"Before Info URL: {beforeinfo_url}")
        print(f"Odds Info URL: {odds_url}")
        
        print("\n【Weather Information】")
        for key, value in weather_info.items():
            print(f"{key}: {value}")
        
        print("\n【Before Info for Each Boat】")
        for boat in boat_info:
            print(f"Boat Number {boat['艇番']}: Exhibition Time {boat['展示タイム']}, Course {boat['進入']}, Win Odds {boat['単勝オッズ']}")
        
        # Get basic race information
        race_base_info = get_race_base_info(date, venue_name, race_number)
        
        print("\n【Fetched Basic Race Information】")
        print(f"Number of Rows: {len(race_base_info)}")
        
        # Get scheduled deadline time
        if '締切予定時刻' in race_base_info.columns:
            deadline_time_str = race_base_info['締切予定時刻'].iloc[0]  # Scheduled Deadline Time
            
            # Convert scheduled deadline time to datetime object
            # Process according to the format
            jst = pytz.timezone('Asia/Tokyo')
            
            # First create the date part
            deadline_date = datetime.strptime(date, '%Y%m%d').date()
            
            try:
                # If in "HH:MM" format
                if ':' in deadline_time_str:
                    deadline_time = datetime.strptime(deadline_time_str, '%H:%M').time()
                # If in "HHMM" format
                elif len(deadline_time_str) == 4 and deadline_time_str.isdigit():
                    deadline_time = datetime.strptime(deadline_time_str, '%H%M').time()
                else:
                    raise ValueError(f"Unknown time format: {deadline_time_str}")
                
                # Combine date and time to create datetime object
                deadline_datetime = datetime.combine(deadline_date, deadline_time)
                deadline_datetime = jst.localize(deadline_datetime)
                
                # Calculate remaining time until deadline (minutes)
                time_remaining = (deadline_datetime - prediction_time).total_seconds() / 60
            except Exception as e:
                print(f"Failed to parse deadline time: {e}")
                deadline_time_str = "Unknown"
                time_remaining = None
        else:
            print("Warning: Scheduled deadline time is not included in the race information")
            deadline_time_str = "Unknown"
            time_remaining = None
        
        # Convert boat_info to DataFrame
        boat_info_df = pd.DataFrame(boat_info)
        
        # Convert boat number to numeric type (for merging)
        boat_info_df['艇番'] = pd.to_numeric(boat_info_df['艇番'])  # Boat Number
        
        # Merge before info and basic race information
        merged_df = pd.merge(
            race_base_info,
            boat_info_df,
            on='艇番',  # Boat Number
            how='inner'
        )
        
        print(f"\n【Merged Information】")
        print(f"Merged Data: {merged_df.shape[0]} rows × {merged_df.shape[1]} columns")
        
        # Check if the merged data has information for 6 boats
        if len(merged_df) != 6:
            print(f"Warning: Merged data does not have information for 6 boats ({len(merged_df)} boats). Please check the data.")
        
        # Add weather information to each row
        for key, value in weather_info.items():
            merged_df[key] = value
        
        # Copy "レース場" to "会場" (to match the feature name in the model)
        if 'レース場' in merged_df.columns and '会場' not in merged_df.columns:
            merged_df['会場'] = merged_df['レース場']  # Venue
        
        # Generate race ID (format: date_venue_race number)
        merged_df['レースID'] = int(f"{date}_{venue_code}_{race_number}")  # Race ID
        
        # Data preprocessing
        preprocessed_df = preprocess_data(merged_df)
        
        # Prediction using the model
        prediction_results = predict_race_result(preprocessed_df)
        
        # Create a string for the prediction time
        prediction_time_str = prediction_time.strftime('%H:%M')
        
        # Display time information
        print(f"\n【Prediction Results ({venue_name} {race_number})】")
        print(f"Prediction Time: {prediction_time_str} JST")
        
        if deadline_time_str != "Unknown":
            print(f"Scheduled Deadline Time: {deadline_time_str}")
            
            if time_remaining is not None:
                if time_remaining > 0:
                    print(f"Approximately {int(time_remaining)} minutes remaining until the deadline")
                else:
                    print(f"Approximately {abs(int(time_remaining))} minutes have passed since the deadline")
        
        # Display prediction results
        for _, row in prediction_results.iterrows():
            print(f"Boat Number {int(row['艇番'])}: Win Probability {row['勝率予測']:.2%}, Win Odds {row['単勝オッズ']:.1f}, Expected Value {row['期待値']:.2f}")
        
        # Calculate prediction confidence (sum of win probabilities of the top 2 boats)
        top2_probs_sum = prediction_results.iloc[:2]['勝率予測'].sum()
        print(f"\n【Prediction Confidence】: {top2_probs_sum:.2%}")
        
        # Extract boats with expected value greater than 1.0 (positive expected value boats)
        plus_ev_boats = prediction_results[prediction_results['期待値'] > 1.0]
        if not plus_ev_boats.empty:
            print("\n【Recommended Bets】(Boats with expected value greater than 1.0)")
            for _, row in plus_ev_boats.iterrows():
                print(f"Boat Number {int(row['艇番'])}: Expected Value {row['期待値']:.2f}")
        else:
            print("\n【Recommended Bets】: None (No boats with expected value greater than 1.0)")
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
