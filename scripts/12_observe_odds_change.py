#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import requests
from bs4 import BeautifulSoup
import re
import os
import time
import datetime
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import pytz
import locale

# Locale setting (Japanese environment)
try:
    locale.setlocale(locale.LC_ALL, 'ja_JP.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_ALL, 'ja_JP.utf8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, 'Japanese_Japan.932')
        except locale.Error:
            print("Warning: Could not set Japanese locale.")
            pass

# Set matplotlib RC parameters
plt.rcParams['axes.unicode_minus'] = False  # Correctly display minus sign

# Font setting (for Japanese display) - revised version
# Ensure to find and set Japanese fonts on MacOS
def set_japanese_font():
    # Function to debug fonts
    def debug_fonts():
        print("Available fonts:")
        for font in sorted(fm.findSystemFonts()):
            try:
                font_name = fm.FontProperties(fname=font).get_name()
                if any(jp_char in font_name for jp_char in ['ヒラギノ', '角ゴ', '丸ゴ', 'ゴシック', 'メイリオ', '明朝']):
                    print(f"  - {font_name} ({font})")
            except:
                pass
    
    # Common Japanese fonts for Mac
    font_candidates = [
        '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc',
        '/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc',
        '/System/Library/Fonts/ヒラギノ角ゴシック W5.ttc',
        '/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc',
        '/System/Library/Fonts/ヒラギノ角ゴシック W7.ttc',
        '/System/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc',
        '/System/Library/Fonts/AppleSDGothicNeo.ttc',
        '/Library/Fonts/Osaka.ttf',
        # Path for Catalina and later
        '/System/Library/Fonts/Supplemental/ヒラギノ角ゴシック W3.ttc',
        '/System/Library/Fonts/Supplemental/ヒラギノ角ゴシック W4.ttc',
        '/System/Library/Fonts/Supplemental/ヒラギノ角ゴシック W5.ttc',
        '/System/Library/Fonts/Supplemental/ヒラギノ角ゴシック W6.ttc',
        '/System/Library/Fonts/Supplemental/ヒラギノ角ゴシック W7.ttc',
        '/System/Library/Fonts/Supplemental/ヒラギノ角ゴシック W8.ttc',
        '/System/Library/Fonts/Supplemental/ヒラギノ角ゴシック W9.ttc',
        '/System/Library/Fonts/Supplemental/ヒラギノ丸ゴ ProN W4.ttc',
    ]
    
    # Find existing font files
    for font_path in font_candidates:
        if os.path.exists(font_path):
            try:
                fm.fontManager.addfont(font_path)
                print(f"Registered font: {font_path}")
                # Set font
                rcParams['font.family'] = 'sans-serif'
                if 'Hiragino' in font_path or 'ヒラギノ' in font_path:
                    rcParams['font.sans-serif'] = ['Hiragino Sans', 'Hiragino Kaku Gothic ProN', 'Hiragino Maru Gothic ProN', 'sans-serif']
                elif 'AppleSDGothicNeo' in font_path:
                    rcParams['font.sans-serif'] = ['AppleGothic', 'sans-serif']
                elif 'Osaka' in font_path:
                    rcParams['font.sans-serif'] = ['Osaka', 'sans-serif']
                print(f"Set Japanese font: {rcParams['font.sans-serif'][0]}")
                return True
            except Exception as e:
                print(f"Error registering font: {e}")
                continue
    
    # Search for Japanese fonts in the system
    try:
        for font in fm.findSystemFonts():
            try:
                font_name = fm.FontProperties(fname=font).get_name()
                if any(jp_font in font_name for jp_font in ['Hiragino', 'ヒラギノ', 'Gothic', 'ゴシック', 'Meiryo', 'メイリオ']):
                    fm.fontManager.addfont(font)
                    rcParams['font.family'] = 'sans-serif'
                    rcParams['font.sans-serif'] = [font_name, 'sans-serif']
                    print(f"Found Japanese font: {font_name} ({font})")
                    return True
            except:
                continue
    except Exception as e:
        print(f"Error searching for fonts: {e}")
    
    # Additional font settings for matplotlib 3.2.0 and later
    try:
        rcParams['axes.formatter.use_locale'] = True  # Use locale settings
        # Final fallback settings
        rcParams['font.family'] = 'sans-serif'
        # Common Japanese fonts for Mac/Linux/Windows
        rcParams['font.sans-serif'] = ['AppleGothic', 'Hiragino Sans', 'Hiragino Kaku Gothic ProN', 
                                      'Osaka', 'Yu Gothic', 'MS Gothic', 'Noto Sans CJK JP', 
                                      'Droid Sans Japanese', 'Meiryo', 'sans-serif']
        print("Using default Japanese font settings.")
    except Exception as e:
        print(f"Error in fallback settings: {e}")
    
    # Debug output for font detection results
    debug_fonts()
    
    return False

# Apply Japanese font settings
set_japanese_font()

# List of boatrace venue codes
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

# Function to build odds information page URL
def build_odds_url(date, venue_code, race_number):
    base_url = "https://www.boatrace.jp/owpc/pc/race/oddstf"
    url = f"{base_url}?rno={race_number}&jcd={venue_code}&hd={date}"
    return url

# Function to fetch HTML
def fetch_html(url):
    try:
        print(f"Fetching HTML from URL: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise exception for errors
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to fetch HTML: {e}")
        return None

# Function to scrape odds information
def scrape_odds_info(soup):
    odds_info = {}
    
    try:
        # Find the win odds table
        # Look for the table after the heading "単勝オッズ"
        odds_tables = soup.find_all('table')
        
        for table in odds_tables:
            # Check if the table header contains "単勝オッズ"
            header = table.find_previous('div', string=lambda text: text and '単勝オッズ' in text if text else False)
            if header or table.find('th', string=lambda text: text and '単勝オッズ' in text if text else False):
                # Process each row
                rows = table.find_all('tr')
                for row in rows:
                    # First cell is boat number, second cell is odds value
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        # Get boat number (only digits)
                        boat_number_cell = cells[0]
                        boat_number = boat_number_cell.text.strip()
                        
                        # Extract only digits
                        if boat_number.isdigit() and 1 <= int(boat_number) <= 6:
                            # Get odds value
                            odds_value = cells[-1].text.strip()  # Last cell is odds value
                            odds_info[boat_number] = odds_value
        
        # Try another method (if table structure is different)
        if not odds_info:
            # Directly find the win odds table
            for table in odds_tables:
                # Check the first row of the table
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
            # Identify by class name for boat number
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
        print(f"Error occurred while scraping odds information: {e}")
    
    return odds_info

# Function to get deadline time (JST)
def get_deadline_time(date, venue_name, race_number):
    # Find the latest CSV file from the race_info directory
    data_dir = 'data/race_info'
    pattern = os.path.join(data_dir, 'races_*.csv')
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"Error: No CSV files found in {data_dir} directory.")
        sys.exit(1)
    
    # Sort by date in the file name (get the latest one)
    latest_csv = max(csv_files)
    print(f"Latest race information file: {latest_csv}")
    
    # Read the CSV file
    df = pd.read_csv(latest_csv)
    
    # Search for rows matching the date, venue, and race number
    date_obj = datetime.datetime.strptime(date, '%Y%m%d')
    date_formatted = date_obj.strftime('%Y-%m-%d')
    
    race_info = df[(df['日付'] == date_formatted) & 
                    (df['レース場'] == venue_name) & 
                    (df['レース番号'] == int(race_number))]
    
    if race_info.empty:
        print(f"Error: No race information found for date={date_formatted}, venue={venue_name}, race number={race_number}.")
        sys.exit(1)
    
    # Get the scheduled deadline time
    deadline_time_str = race_info.iloc[0]['締切予定時刻']
    
    # Convert time string to datetime object
    deadline_time = datetime.datetime.strptime(f"{date_formatted} {deadline_time_str}", '%Y-%m-%d %H:%M')
    
    # Set timezone (Japan Standard Time)
    jst = pytz.timezone('Asia/Tokyo')
    deadline_time = jst.localize(deadline_time)
    
    return deadline_time

# Function to fetch and record odds information
def fetch_and_record_odds(date, venue_code, race_number, deadline_time, venue_name):
    odds_url = build_odds_url(date, venue_code, race_number)
    
    # DataFrame to save results
    columns = ['timestamp', 'Boat 1', 'Boat 2', 'Boat 3', 'Boat 4', 'Boat 5', 'Boat 6']
    odds_data = pd.DataFrame(columns=columns)
    
    # Set timezone (JST)
    jst = pytz.timezone('Asia/Tokyo')
    
    # Current time (convert from system time to JST)
    now = datetime.datetime.now(jst)
    
    print(f"Current time (JST): {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Deadline time (JST): {deadline_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Calculate time 3 minutes after the deadline
    deadline_plus_3min = deadline_time + datetime.timedelta(minutes=3)
    
    # If current time is past the deadline + 3 minutes, display error message
    if now > deadline_plus_3min:
        print(f"Error: Current time is past the deadline + 3 minutes. Cannot fetch odds.")
        print(f"Deadline + 3 minutes: {deadline_plus_3min.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        return pd.DataFrame(columns=columns)  # Return empty DataFrame
    
    # If current time is past the deadline, display warning
    if now > deadline_time:
        print(f"Warning: Deadline has already passed. Only fetching final odds.")
    
    # Repeat fetching until 3 minutes after the deadline (to include final odds)
    while now < deadline_plus_3min:
        # Change message before and after the deadline
        if now < deadline_time:
            print(f"\nFetching odds at {now.strftime('%Y-%m-%d %H:%M:%S')}...")
        else:
            print(f"\nFetching odds after deadline at {now.strftime('%Y-%m-%d %H:%M:%S')} (to include final odds)...")
        
        # Fetch HTML
        html = fetch_html(odds_url)
        if html:
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Get odds information
            odds_info = scrape_odds_info(soup)
            
            # Add odds information to DataFrame
            row_data = {'timestamp': now}
            for i in range(1, 7):
                boat_num = str(i)
                if boat_num in odds_info:
                    try:
                        # Convert odds to numeric (set to NaN if not numeric)
                        row_data[f'Boat {i}'] = float(odds_info[boat_num])
                    except ValueError:
                        row_data[f'Boat {i}'] = None
                else:
                    row_data[f'Boat {i}'] = None
            
            # Add to DataFrame
            odds_data = pd.concat([odds_data, pd.DataFrame([row_data])], ignore_index=True)
            
            # Display current odds
            for i in range(1, 7):
                boat_num = str(i)
                if boat_num in odds_info:
                    print(f"Boat {i}: {odds_info[boat_num]}")
        
        # Wait for 1 minute
        time.sleep(60)
        
        # Update current time (always fetch in JST)
        now = datetime.datetime.now(jst)
    
    print("\nReached 3 minutes after the deadline. Stopping odds fetching.")
    
    # If data is empty, display warning and exit
    if odds_data.empty:
        print("Warning: No odds data fetched. CSV file will not be saved.")
        return odds_data
    
    # Create directory to save CSV file (if not exists)
    odds_change_dir = '/Users/yoshidont_mind/Library/Mobile Documents/com~apple~CloudDocs/Desktop/CST term4/COMP 4949 Big Data Analytics Methods/boat_race_prediction/data/odds_change'
    os.makedirs(odds_change_dir, exist_ok=True)
    
    # Create path for CSV file
    csv_filename = f'odds_change_{date}_{venue_name}_{race_number}.csv'
    csv_path = os.path.join(odds_change_dir, csv_filename)
    
    # Convert timestamp to string and save to CSV
    odds_data_for_csv = odds_data.copy()
    
    # Check if timestamp column is datetime type and process accordingly
    try:
        # If datetime object, use .dt accessor
        if pd.api.types.is_datetime64_any_dtype(odds_data_for_csv['timestamp']):
            odds_data_for_csv['timestamp'] = odds_data_for_csv['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        else:
            # If already string, use as is; if other type, try to convert
            odds_data_for_csv['timestamp'] = odds_data_for_csv['timestamp'].apply(
                lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if isinstance(x, datetime.datetime) else str(x)
            )
    except Exception as e:
        print(f"Warning: Error occurred while converting timestamp: {e}")
        print("Timestamp will be saved in its original format.")
    
    odds_data_for_csv.to_csv(csv_path, index=False)
    print(f"Saved odds data to CSV: {csv_path}")
    
    return odds_data

# Function to plot odds change
def plot_odds_change(odds_data, date, venue_name, race_number):
    # Create figure for plotting
    plt.figure(figsize=(12, 8))
    
    # Ensure font settings are applied if needed
    set_japanese_font()
    
    # Define colors (Boat 1→Brown, Boat 2→Black, Boat 3→Red, Boat 4→Blue, Boat 5→Yellow, Boat 6→Green)
    colors = ['#8B4513', '#000000', '#FF0000', '#0000FF', '#FFD700', '#008000']
    
    # Plot odds change for each boat
    for i in range(1, 7):
        col_name = f'Boat {i}'
        if col_name in odds_data.columns:
            plt.plot(odds_data['timestamp'], odds_data[col_name], 
                    label=f'Boat {i}', color=colors[i-1], linewidth=2, marker='o')
    
    # Graph settings (explicitly specify Japanese font)
    font_prop = fm.FontProperties(family=rcParams['font.sans-serif'][0])
    plt.title(f'{date} {venue_name} Race {race_number} Odds Change', fontsize=16, fontproperties=font_prop)
    plt.xlabel('Time', fontsize=12, fontproperties=font_prop)
    plt.ylabel('Odds', fontsize=12, fontproperties=font_prop)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set Japanese font for legend
    plt.legend(fontsize=12, prop=font_prop)
    
    # Set time format for x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gcf().autofmt_xdate()  # Automatically adjust date labels for better readability
    
    # Adjust Y-axis range (add margin)
    all_odds = []
    for i in range(1, 7):
        col_name = f'Boat {i}'
        if col_name in odds_data.columns:
            # Exclude NaN
            valid_odds = odds_data[col_name].dropna()
            if not valid_odds.empty:
                all_odds.extend(valid_odds)
    
    if all_odds:
        min_odds = min(all_odds)
        max_odds = max(all_odds)
        margin = (max_odds - min_odds) * 0.1  # 10% margin
        plt.ylim(max(0, min_odds - margin), max_odds + margin)
    
    # Save file name
    filename = f'odds_change_{date}_{venue_name}_R{race_number}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved graph: {filename}")
    
    # Display graph
    plt.tight_layout()
    plt.show()

def main():
    try:
        # Parse command line arguments
        if len(sys.argv) != 4:
            print("Usage: python scripts/12_odds_change.py <date(YYYYMMDD)> <venue name> <race number>")
            print("Example: python scripts/12_odds_change.py 20250402 三国 8")
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
                print("エラー: レース番号は1から12の間で指定してください。")
                sys.exit(1)
            race_number = str(race_number)  # 文字列に戻す
        except ValueError:
            print("エラー: レース番号は数字で指定してください。")
            sys.exit(1)
        
        # 競艇場コードの取得
        venue_code = get_venue_code(venue_name)
        
        try:
            # 締切時刻の取得
            deadline_time = get_deadline_time(date, venue_name, race_number)
            
            # オッズ情報の取得と記録
            odds_data = fetch_and_record_odds(date, venue_code, race_number, deadline_time, venue_name)
            
            # オッズの推移をグラフ化
            if not odds_data.empty:
                try:
                    plot_odds_change(odds_data, date, venue_name, race_number)
                except Exception as plot_error:
                    print(f"グラフの作成中にエラーが発生しました: {plot_error}")
                    import traceback
                    traceback.print_exc()
            else:
                print("オッズデータが取得できなかったため、グラフは作成されません。")
        except Exception as data_error:
            print(f"データ処理中にエラーが発生しました: {data_error}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
