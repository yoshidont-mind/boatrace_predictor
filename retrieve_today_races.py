import pandas as pd
import os
import glob
from pathlib import Path
from datetime import datetime, date

# ====== Function to get the list of races for a specified date ======
def get_races_by_date(selected_date=None):
    """
    Retrieve the list of races for a specified date from a CSV file
    
    Parameters:
    -----------
    selected_date : datetime or str, optional
        The date to retrieve. If not specified, today's date is used.
        
    Returns:
    --------
    pandas.DataFrame or None
        DataFrame of the race list. Returns None if no appropriate file is found.
    """
    if selected_date is None:
        selected_date = datetime.today()
    elif isinstance(selected_date, str):
        selected_date = datetime.strptime(selected_date, "%Y-%m-%d")
    
    date_str = selected_date.strftime("%Y%m%d")
    
    # Search for files corresponding to the specified date in the data/race_info directory
    race_info_dir = Path("data/race_info")
    pattern = f"races_{date_str}*.csv"
    matching_files = list(race_info_dir.glob(pattern))
    
    if not matching_files:
        print(f"⚠️ No race information file found for {date_str}")
        return None
    
    # Select the file with the latest time (hhmm)
    latest_file = max(matching_files, key=lambda f: f.name)
    print(f"📂 Loading {latest_file.name}")
    
    # Load CSV file
    df = pd.read_csv(latest_file)
    
    # Remove duplicates (group by race ID, venue, and race number, and get the first row)
    df = df.drop_duplicates(subset=["レース場", "レース番号", "日付"])  # Venue, Race Number, Date
    
    # Reconstruct race ID
    def format_race_id(row):
        date_part = row["日付"].replace("-", "")  # Date
        venue_map = {
            '桐生': '01', '戸田': '02', '江戸川': '03', '平和島': '04', '多摩川': '05',
            '浜名湖': '06', '蒲郡': '07', '常滑': '08', '津': '09', '三国': '10',
            'びわこ': '11', '住之江': '12', '尼崎': '13', '鳴門': '14', '丸亀': '15',
            '児島': '16', '宮島': '17', '徳山': '18', '下関': '19', '若松': '20',
            '芦屋': '21', '福岡': '22', '唐津': '23', '大村': '24'
        }
        venue_code = venue_map.get(row["レース場"], "00")  # Venue
        race_no = int(row["レース番号"])  # Race Number
        return f"{date_part}{venue_code}{race_no:02d}"
    
    df["レースID"] = df.apply(format_race_id, axis=1)  # Race ID
    
    # Sort by scheduled closing time
    if "締切予定時刻" in df.columns:  # Scheduled Closing Time
        df = df.sort_values("締切予定時刻")  # Scheduled Closing Time
    
    # Select only the required columns
    required_columns = ["レースID", "日付", "レース場", "レース番号", "締切予定時刻", "ステータス"]  # Race ID, Date, Venue, Race Number, Scheduled Closing Time, Status
    columns_to_select = [col for col in required_columns if col in df.columns]
    
    # Add status column if it does not exist
    if "ステータス" not in df.columns:  # Status
        df["ステータス"] = "投票"  # Voting
    
    return df[columns_to_select]

# ====== Function to get the list of available dates ======
def get_available_dates():
    """
    Retrieve the list of available dates from CSV files
    
    Returns:
    --------
    list
        List of available dates in YYYY-MM-DD format
    """
    race_info_dir = Path("data/race_info")
    files = glob.glob(str(race_info_dir / "races_*.csv"))
    
    dates = set()
    for file_path in files:
        filename = os.path.basename(file_path)
        # Extract the YYYYMMDD part from the filename (races_YYYYMMDDhhmm.csv)
        date_str = filename.split('_')[1][:8]
        if len(date_str) == 8 and date_str.isdigit():
            # Convert to YYYY-MM-DD format
            formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            dates.add(formatted_date)
    
    return sorted(list(dates))

# Function for backward compatibility (maintain compatibility with existing code)
def get_today_races():
    """
    Function to get today's race list (calls get_races_by_date)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame of today's race list
    """
    return get_races_by_date(None)
