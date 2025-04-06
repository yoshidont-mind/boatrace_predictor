import pandas as pd
import os
import glob
from pathlib import Path
from datetime import datetime, date

# ====== 指定日付のレース一覧を取得する関数 ======
def get_races_by_date(selected_date=None):
    """
    指定日付のレース一覧をCSVファイルから取得する
    
    Parameters:
    -----------
    selected_date : datetime or str, optional
        取得する日付。指定しない場合は本日の日付を使用
        
    Returns:
    --------
    pandas.DataFrame or None
        レース一覧のデータフレーム。適切なファイルが見つからない場合はNone
    """
    if selected_date is None:
        selected_date = datetime.today()
    elif isinstance(selected_date, str):
        selected_date = datetime.strptime(selected_date, "%Y-%m-%d")
    
    date_str = selected_date.strftime("%Y%m%d")
    
    # data/race_info ディレクトリ内の該当日付のファイルを検索
    race_info_dir = Path("data/race_info")
    pattern = f"races_{date_str}*.csv"
    matching_files = list(race_info_dir.glob(pattern))
    
    if not matching_files:
        print(f"⚠️ {date_str} に対応するレース情報ファイルが見つかりません")
        return None
    
    # 時刻（hhmm）が最新のファイルを選択
    latest_file = max(matching_files, key=lambda f: f.name)
    print(f"📂 {latest_file.name} を読み込みます")
    
    # CSVファイル読み込み
    df = pd.read_csv(latest_file)
    
    # 重複を排除（レースID、レース場、レース番号でグループ化して最初の行を取得）
    df = df.drop_duplicates(subset=["レース場", "レース番号", "日付"])
    
    # レースIDを再構築
    def format_race_id(row):
        date_part = row["日付"].replace("-", "")
        venue_map = {
            '桐生': '01', '戸田': '02', '江戸川': '03', '平和島': '04', '多摩川': '05',
            '浜名湖': '06', '蒲郡': '07', '常滑': '08', '津': '09', '三国': '10',
            'びわこ': '11', '住之江': '12', '尼崎': '13', '鳴門': '14', '丸亀': '15',
            '児島': '16', '宮島': '17', '徳山': '18', '下関': '19', '若松': '20',
            '芦屋': '21', '福岡': '22', '唐津': '23', '大村': '24'
        }
        venue_code = venue_map.get(row["レース場"], "00")
        race_no = int(row["レース番号"])
        return f"{date_part}{venue_code}{race_no:02d}"
    
    df["レースID"] = df.apply(format_race_id, axis=1)
    
    # 締切予定時刻でソート
    if "締切予定時刻" in df.columns:
        df = df.sort_values("締切予定時刻")
    
    # 必要な列のみ選択
    required_columns = ["レースID", "日付", "レース場", "レース番号", "締切予定時刻", "ステータス"]
    columns_to_select = [col for col in required_columns if col in df.columns]
    
    # ステータス列がない場合は追加
    if "ステータス" not in df.columns:
        df["ステータス"] = "投票"
    
    return df[columns_to_select]

# ====== 利用可能な日付一覧を取得する関数 ======
def get_available_dates():
    """
    CSVファイルから利用可能な日付一覧を取得する
    
    Returns:
    --------
    list
        利用可能な日付（YYYY-MM-DD形式）のリスト
    """
    race_info_dir = Path("data/race_info")
    files = glob.glob(str(race_info_dir / "races_*.csv"))
    
    dates = set()
    for file_path in files:
        filename = os.path.basename(file_path)
        # ファイル名からYYYYMMDDの部分を抽出（races_YYYYMMDDhhmm.csv）
        date_str = filename.split('_')[1][:8]
        if len(date_str) == 8 and date_str.isdigit():
            # YYYY-MM-DD形式に変換
            formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            dates.add(formatted_date)
    
    return sorted(list(dates))

# 後方互換性のための関数（現行コードとの互換性維持）
def get_today_races():
    """
    本日のレース一覧を取得する関数（get_races_by_dateを呼び出し）
    
    Returns:
    --------
    pandas.DataFrame
        本日のレース一覧のデータフレーム
    """
    return get_races_by_date(None)
