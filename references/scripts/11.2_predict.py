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

# ボートレース場コード一覧
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

# 競艇場名からコードを取得する関数
def get_venue_code(venue_name):
    # 逆引き辞書を作成
    venue_to_code = {v: k for k, v in boatrace_venues.items()}
    
    # 競艇場名からコードを取得
    if venue_name in venue_to_code:
        return venue_to_code[venue_name]
    else:
        print(f"エラー: 競艇場「{venue_name}」は見つかりません。")
        print("有効な競艇場名: " + ", ".join(boatrace_venues.values()))
        sys.exit(1)

# 直前情報ページのURLを構築する関数
def build_beforeinfo_url(date, venue_code, race_number):
    base_url = "https://www.boatrace.jp/owpc/pc/race/beforeinfo"
    url = f"{base_url}?rno={race_number}&jcd={venue_code}&hd={date}"
    return url

# オッズ情報ページのURLを構築する関数
def build_odds_url(date, venue_code, race_number):
    base_url = "https://www.boatrace.jp/owpc/pc/race/oddstf"
    url = f"{base_url}?rno={race_number}&jcd={venue_code}&hd={date}"
    return url

# HTMLを取得する関数
def fetch_html(url, use_local_file=None):
    # 現在時刻をJSTで取得
    jst = pytz.timezone('Asia/Tokyo')
    fetch_time = datetime.now(jst)
    
    if use_local_file and os.path.exists(use_local_file):
        print(f"ローカルファイル {use_local_file} からHTMLを読み込みます")
        try:
            with open(use_local_file, 'r', encoding='utf-8') as f:
                return f.read(), fetch_time
        except Exception as e:
            print(f"ローカルファイルの読み込みに失敗しました: {e}")
            print("URLからHTMLを取得します")
    
    try:
        print(f"URLからHTMLを取得します: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # エラーがあれば例外を発生させる
        return response.text, fetch_time
    except requests.exceptions.RequestException as e:
        print(f"エラー: HTMLの取得に失敗しました: {e}")
        sys.exit(1)

# 各艇の直前情報をスクレイピングする関数
def scrape_boat_info(soup):
    boat_info = []
    
    try:
        # 各艇の情報を取得
        tbody_elements = soup.select('tbody.is-fs12')
        
        for tbody in tbody_elements:
            boat_number_element = tbody.select_one('td.is-fs14')
            if boat_number_element:
                boat_number = boat_number_element.text.strip()
                
                # 展示タイム（左から5列目）を取得
                # 各艇のtbodyの最初の行の5番目のtd要素
                first_row = tbody.select('tr')[0]
                td_elements = first_row.select('td')
                
                # 艇番、写真、選手名、体重の後に展示タイムがある
                # rowspanが使われているため、直接インデックスでアクセスできない場合がある
                exhibition_time = "N/A"
                
                # 展示タイムは通常、rowspan="4"属性を持つtd要素
                for td in tbody.select('td[rowspan="4"]'):
                    # 数値のみを含むtd要素を探す（展示タイム）
                    if re.match(r'^\d+\.\d+$', td.text.strip()):
                        exhibition_time = td.text.strip()
                        break
                
                # 進入情報を取得（2行目の2列目）
                rows = tbody.select('tr')
                entry = "N/A"
                if len(rows) > 1:
                    entry_cells = rows[1].select('td')
                    if len(entry_cells) > 1:
                        entry = entry_cells[1].text.strip()
                
                boat_info.append({
                    '艇番': boat_number,
                    '展示タイム': exhibition_time,
                    '進入': entry,
                    '単勝オッズ': "N/A"  # オッズ情報は別途取得
                })
    except Exception as e:
        print(f"ボート情報のスクレイピング中にエラーが発生しました: {e}")
        # エラーが発生しても処理を続行するため、空のリストを返さない
    
    # データが取得できなかった場合は空のデータを作成
    if not boat_info:
        for i in range(1, 7):
            boat_info.append({
                '艇番': str(i),
                '展示タイム': "N/A",
                '進入': "N/A",
                '単勝オッズ': "N/A"
            })
    
    # 艇番順にソート
    boat_info.sort(key=lambda x: int(x['艇番']))
    
    return boat_info

# 単勝オッズ情報をスクレイピングする関数
def scrape_odds_info(soup):
    odds_info = {}
    
    try:
        # 単勝オッズテーブルを探す
        # 「単勝オッズ」という見出しの後にあるテーブルを探す
        odds_tables = soup.find_all('table')
        
        for table in odds_tables:
            # テーブルヘッダーに「単勝オッズ」が含まれているか確認
            header = table.find_previous('div', string=lambda text: text and '単勝オッズ' in text if text else False)
            if header or table.find('th', string=lambda text: text and '単勝オッズ' in text if text else False):
                # 各行を処理
                rows = table.find_all('tr')
                for row in rows:
                    # 最初のセルが艇番、2番目のセルがオッズ値
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        # 艇番を取得（数字のみ）
                        boat_number_cell = cells[0]
                        boat_number = boat_number_cell.text.strip()
                        
                        # 数字のみを抽出
                        if boat_number.isdigit() and 1 <= int(boat_number) <= 6:
                            # オッズ値を取得
                            odds_value = cells[-1].text.strip()  # 最後のセルがオッズ値
                            odds_info[boat_number] = odds_value
        
        # 別の方法でも試す（テーブルの構造が異なる場合）
        if not odds_info:
            # 単勝オッズテーブルを直接探す
            for table in odds_tables:
                # テーブルの最初の行を確認
                first_row = table.find('tr')
                if first_row:
                    # 「単勝オッズ」という列見出しがあるか確認
                    headers = first_row.find_all('th')
                    for i, header in enumerate(headers):
                        if header.text.strip() == '単勝オッズ':
                            # 各行を処理
                            for row in table.find_all('tr')[1:]:  # ヘッダー行をスキップ
                                cells = row.find_all('td')
                                if len(cells) > i:
                                    # 艇番を取得
                                    boat_number = cells[0].text.strip()
                                    if boat_number.isdigit() and 1 <= int(boat_number) <= 6:
                                        # オッズ値を取得
                                        odds_value = cells[i].text.strip()
                                        odds_info[boat_number] = odds_value
        
        # さらに別の方法でも試す（クラス名を使用）
        if not odds_info:
            # 艇番のクラス名を使って特定
            for i in range(1, 7):
                # 各艇の色に対応するクラス名を持つセルを探す
                boat_cell = soup.select_one(f'td.is-boatColor{i}')
                if boat_cell:
                    # 同じ行内のオッズ値を含むセルを探す
                    row = boat_cell.parent
                    if row:
                        # 最後のセルがオッズ値の場合が多い
                        odds_cell = row.select_one('td:last-child')
                        if odds_cell:
                            odds_value = odds_cell.text.strip()
                            # 数値形式かどうか確認
                            if re.match(r'^\d+(\.\d+)?$', odds_value):
                                odds_info[str(i)] = odds_value
    
    except Exception as e:
        print(f"オッズ情報のスクレイピング中にエラーが発生しました: {e}")
    
    return odds_info

# 天候情報をスクレイピングする関数
def scrape_weather_info(soup):
    weather_info = {
        '天候': 'N/A',
        '風向': 'N/A',
        '風量': 'N/A',
        '波': 'N/A'
    }
    
    try:
        # 天候情報を取得
        weather_section = soup.select_one('.weather1')
        if weather_section:
            # 天候（雨、曇り、晴れなど）
            weather_unit = weather_section.select_one('.weather1_bodyUnit.is-weather')
            if weather_unit:
                weather_label = weather_unit.select_one('.weather1_bodyUnitLabelTitle')
                if weather_label:
                    weather_info['天候'] = weather_label.text.strip()
            
            # 風向
            wind_direction_element = weather_section.select_one('.weather1_bodyUnit.is-windDirection .weather1_bodyUnitImage')
            if wind_direction_element:
                # クラス名から風向を抽出（例: is-wind14 → 14）
                wind_direction_class = wind_direction_element.get('class', [])
                wind_direction = next((cls.replace('is-wind', '') for cls in wind_direction_class if cls.startswith('is-wind')), "N/A")
                weather_info['風向'] = wind_direction
            
            # 風量（風速）
            wind_speed_element = weather_section.select_one('.weather1_bodyUnit.is-wind .weather1_bodyUnitLabelData')
            if wind_speed_element:
                weather_info['風量'] = wind_speed_element.text.strip()
            
            # 波高
            wave_element = weather_section.select_one('.weather1_bodyUnit.is-wave .weather1_bodyUnitLabelData')
            if wave_element:
                weather_info['波'] = wave_element.text.strip()
    except Exception as e:
        print(f"天候情報のスクレイピング中にエラーが発生しました: {e}")
        # エラーが発生しても処理を続行
    
    return weather_info

# 最新のレース情報ファイルを取得する関数
def get_latest_race_info_file():
    data_dir = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/Desktop/CST term4/COMP 4949 Big Data Analytics Methods/boat_race_prediction/data/race_info")
    race_files = glob.glob(os.path.join(data_dir, "races_*.csv"))
    
    if not race_files:
        print("エラー: レース情報ファイルが見つかりません。")
        sys.exit(1)
    
    # ファイル名で日時が最新のものを選択
    latest_file = max(race_files)
    print(f"最新のレース情報ファイル: {os.path.basename(latest_file)}")
    return latest_file

# 最新のモデルファイルを取得する関数
def get_latest_model_file():
    model_dir = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/Desktop/CST term4/COMP 4949 Big Data Analytics Methods/boat_race_prediction/models")
    model_files = glob.glob(os.path.join(model_dir, "model_*.pkl"))
    
    if not model_files:
        print("エラー: モデルファイルが見つかりません。")
        sys.exit(1)
    
    # ファイル名で日付が最新のものを選択
    latest_file = max(model_files)
    print(f"最新のモデルファイル: {os.path.basename(latest_file)}")
    return latest_file

# レース基本情報を取得する関数
def get_race_base_info(date, venue_name, race_number):
    # 最新のレース情報ファイルを取得
    race_info_file = get_latest_race_info_file()
    
    # CSVファイルを読み込む
    try:
        race_info_df = pd.read_csv(race_info_file)
    except Exception as e:
        print(f"レース情報ファイルの読み込みに失敗しました: {e}")
        sys.exit(1)
    
    # 日付を統一形式に変換（2025/04/02 または 2025-04-02 → 20250402）
    if '/' in race_info_df['日付'].iloc[0]:
        race_info_df['日付'] = race_info_df['日付'].str.replace('/', '')
    elif '-' in race_info_df['日付'].iloc[0]:
        race_info_df['日付'] = race_info_df['日付'].str.replace('-', '')
    
    # 指定された条件でフィルタリング
    filtered_df = race_info_df[
        (race_info_df['日付'] == date) &
        (race_info_df['レース場'] == venue_name) &
        (race_info_df['レース番号'] == int(race_number))
    ]
    
    if filtered_df.empty:
        print(f"エラー: 指定されたレース（{date}, {venue_name}, {race_number}）の情報が見つかりません。")
        sys.exit(1)
    
    return filtered_df

# 偏差値計算関数（レース内での比較）
def calculate_deviation_score(series):
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series([50] * len(series), index=series.index)  # 全員同じ場合は50に固定
    return 10 * (series - mean) / std + 50

# データの前処理を行う関数
def preprocess_data(merged_df):
    df = merged_df.copy()
    print(f"前処理を開始します。データサイズ: {df.shape[0]}行 × {df.shape[1]}列")
    
    # 数値型変換（パーセンテージなど）
    rate_columns = ["全国勝率", "全国2連率", "当地勝率", "当地2連率", "ボート2連率", "モーター2連率"]
    for col in rate_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # 「単勝オッズ」が存在する場合、数値型に変換
    if "単勝オッズ" in df.columns:
        df["単勝オッズ"] = pd.to_numeric(df["単勝オッズ"], errors="coerce")
    
    # 「展示タイム」を数値型に変換
    if "展示タイム" in df.columns:
        df["展示タイム"] = pd.to_numeric(df["展示タイム"], errors="coerce")
        
    # 「風量」を数値型に変換（「3m」→ 3）
    if "風量" in df.columns:
        df["風量"] = df["風量"].astype(str).str.replace('m', '').str.replace('M', '')
        df["風量"] = pd.to_numeric(df["風量"], errors="coerce")
    
    # 「波」を数値型に変換（「3cm」→ 3）
    if "波" in df.columns:
        df["波"] = df["波"].astype(str).str.replace('cm', '').str.replace('CM', '')
        df["波"] = pd.to_numeric(df["波"], errors="coerce")
    
            # 「会場」をカテゴリ型から数値型に変換
    if "会場" in df.columns:
        # 既に数値型の場合はスキップ
        if not pd.api.types.is_numeric_dtype(df["会場"]):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df["会場"] = le.fit_transform(df["会場"].astype(str))
    else:
        # 「会場」が無いがレース場はある場合、レース場から会場を生成
        if "レース場" in df.columns:
            df["会場"] = df["レース場"]
            print("「レース場」から「会場」を生成しました")
            if not pd.api.types.is_numeric_dtype(df["会場"]):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df["会場"] = le.fit_transform(df["会場"].astype(str))
    
    # 日付処理
    if "日付" in df.columns:
        df["日付"] = pd.to_datetime(df["日付"], format="%Y%m%d", errors="coerce")
        df["月"] = df["日付"].dt.month
        df["曜日"] = df["日付"].dt.weekday
    
    # カテゴリ列のエンコーディング（Label EncodingでOK）
    from sklearn.preprocessing import LabelEncoder
    category_cols = ["支部", "級別", "天候", "風向"]
    for col in category_cols:
        if col in df.columns:
            # 数値型かつ整数型であれば変換不要
            if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_integer_dtype(df[col]):
                continue
            df[col] = df[col].astype(str).fillna("不明")
            df[col] = LabelEncoder().fit_transform(df[col])
    
    # 欠損値の補完（中央値を使用）
    median_values = {
        "年齢": 35,
        "体重": 53.0,
        "全国勝率": 5.50,
        "全国2連率": 30.0,
        "当地勝率": 5.50,
        "当地2連率": 30.0,
        "モーター2連率": 30.0,
        "ボート2連率": 30.0,
        "展示タイム": 6.70
    }
    
    for col, median_val in median_values.items():
        if col in df.columns:
            df[col].fillna(median_val, inplace=True)
    
    # 偏差値化する列（レース内で比較する特徴量）
    score_cols = ["全国勝率", "全国2連率", "当地勝率", "当地2連率", "モーター2連率", "ボート2連率", "展示タイム"]
    
    # レース単位で偏差値化
    for col in score_cols:
        if col in df.columns:
            new_col = f"{col}_dev"
            # グループ化する前に欠損値を0で埋める（計算がスキップされるのを防ぐ）
            df[col] = df[col].fillna(0)
            df[new_col] = df.groupby("レースID")[col].transform(calculate_deviation_score)
            
    # 明示的に存在を確認し、ない場合は作成
    required_cols = ["風量", "波"]
    for col in required_cols:
        if col not in df.columns:
            print(f"警告: 特徴量 '{col}' がデータフレームに存在しません。0で補完します。")
            df[col] = 0

    # 偏差値関連の特徴量の強制的な再計算
    score_cols = ["全国勝率", "全国2連率", "当地勝率", "当地2連率", "モーター2連率", "ボート2連率", "展示タイム"]
    for base_col in score_cols:
        dev_col = f"{base_col}_dev"
        if base_col not in df.columns:
            print(f"特徴量 '{dev_col}' の元の特徴量 '{base_col}' がないため、50で補完します")
            df[dev_col] = 50  # 偏差値の中央値
    
    print(f"前処理が完了しました。処理後のデータサイズ: {df.shape[0]}行 × {df.shape[1]}列")
    return df

# モデルを使った予測を行う関数
def predict_race_result(preprocessed_df):
    # 最新のモデルファイルを取得
    model_file = get_latest_model_file()
    
    # モデルの読み込み
    try:
        model = lgb.Booster(model_file=model_file)
    except Exception as e:
        print(f"モデルファイルの読み込みに失敗しました: {e}")
        sys.exit(1)
    
    # モデルの特徴量と同じ順序で特徴量を準備
    feature_names = model.feature_name()
    
    # 特徴量の存在確認と欠損値の補完
    for feature in feature_names:
        if feature not in preprocessed_df.columns:
            print(f"警告: 特徴量 '{feature}' がデータフレームに存在しません。0で補完します。")
            preprocessed_df[feature] = 0
    
    # 予測用の特徴量を抽出
    X_pred = preprocessed_df[feature_names]
    
    # 予測の実行
    probabilities = model.predict(X_pred)
    
    # 勝率の正規化（合計を1にする）
    normalized_probs = probabilities / probabilities.sum()
    
    # 艇番と予測結果を結合
    results = pd.DataFrame({
        '艇番': preprocessed_df['艇番'],
        '勝率予測': normalized_probs,
        '単勝オッズ': preprocessed_df['単勝オッズ']
    })
    
    # 期待値の計算（勝率予測 * オッズ）
    results['期待値'] = results['勝率予測'] * results['単勝オッズ']
    
    # 勝率予測の降順でソート
    results = results.sort_values('勝率予測', ascending=False)
    
    return results

def main():
    try:
        # コマンドライン引数の解析
        if len(sys.argv) != 4:
            print("使用方法: python 11_predict.py <日付(YYYYMMDD)> <競艇場名> <レース番号>")
            print("例: python 11_predict.py 20250402 桐生 12")
            sys.exit(1)
        
        date = sys.argv[1]
        venue_name = sys.argv[2]
        race_number = sys.argv[3]
        
        # 日付形式の検証
        if not re.match(r'^\d{8}$', date):
            print("エラー: 日付は8桁の数字（YYYYMMDD）で指定してください。")
            sys.exit(1)
        
        # レース番号の検証
        try:
            race_number = int(race_number)
            if race_number < 1 or race_number > 12:
                print("エラー: レース番号は1から12の間で指定してください。")
                sys.exit(1)
        except ValueError:
            print("エラー: レース番号は数字で指定してください。")
            sys.exit(1)
        
        # 競艇場コードの取得
        venue_code = get_venue_code(venue_name)
        
        # 直前情報ページのURLを構築
        beforeinfo_url = build_beforeinfo_url(date, venue_code, race_number)
        
        # オッズ情報ページのURLを構築
        odds_url = build_odds_url(date, venue_code, race_number)
        
        # サンプルHTMLファイルのパス（テスト用）
        beforeinfo_html_path = "/home/ubuntu/upload/beforeinfo.html"
        odds_html_path = "/home/ubuntu/upload/odds.html"
        
        # 直前情報HTMLの取得
        beforeinfo_html, _ = fetch_html(beforeinfo_url, use_local_file=beforeinfo_html_path if os.path.exists(beforeinfo_html_path) else None)
        
        # オッズ情報HTMLの取得（このタイミングの時刻を予測時刻とする）
        odds_html, prediction_time = fetch_html(odds_url, use_local_file=odds_html_path if os.path.exists(odds_html_path) else None)
        
        # BeautifulSoupでHTMLを解析
        beforeinfo_soup = BeautifulSoup(beforeinfo_html, 'html.parser')
        odds_soup = BeautifulSoup(odds_html, 'html.parser')
        
        # 各艇の直前情報を取得
        boat_info = scrape_boat_info(beforeinfo_soup)
        
        # 単勝オッズ情報を取得
        odds_info = scrape_odds_info(odds_soup)
        
        # 単勝オッズ情報をboat_infoに統合
        for boat in boat_info:
            boat_number = boat['艇番']
            if boat_number in odds_info:
                boat['単勝オッズ'] = odds_info[boat_number]
        
        # 天候情報を取得
        weather_info = scrape_weather_info(beforeinfo_soup)
        
        print("\n【直前情報を取得しました】")
        print(f"直前情報URL: {beforeinfo_url}")
        print(f"オッズ情報URL: {odds_url}")
        
        print("\n【天候情報】")
        for key, value in weather_info.items():
            print(f"{key}: {value}")
        
        print("\n【各艇の直前情報】")
        for boat in boat_info:
            print(f"艇番 {boat['艇番']}: 展示タイム {boat['展示タイム']}, 進入 {boat['進入']}, 単勝オッズ {boat['単勝オッズ']}")
        
        # レース基本情報を取得
        race_base_info = get_race_base_info(date, venue_name, race_number)
        
        print("\n【レース基本情報を取得しました】")
        print(f"レース数: {len(race_base_info)}行")
        
        # 締切予定時刻を取得
        if '締切予定時刻' in race_base_info.columns:
            deadline_time_str = race_base_info['締切予定時刻'].iloc[0]
            
            # 締切予定時刻をdatetimeオブジェクトに変換
            # 形式によって処理を変える
            jst = pytz.timezone('Asia/Tokyo')
            
            # まず日付部分を作成
            deadline_date = datetime.strptime(date, '%Y%m%d').date()
            
            try:
                # "HH:MM"形式の場合
                if ':' in deadline_time_str:
                    deadline_time = datetime.strptime(deadline_time_str, '%H:%M').time()
                # "HHMM"形式の場合
                elif len(deadline_time_str) == 4 and deadline_time_str.isdigit():
                    deadline_time = datetime.strptime(deadline_time_str, '%H%M').time()
                else:
                    raise ValueError(f"不明な時刻形式: {deadline_time_str}")
                
                # 日付と時刻を組み合わせてdatetimeオブジェクトを作成
                deadline_datetime = datetime.combine(deadline_date, deadline_time)
                deadline_datetime = jst.localize(deadline_datetime)
                
                # 締切までの残り時間（分）を計算
                time_remaining = (deadline_datetime - prediction_time).total_seconds() / 60
            except Exception as e:
                print(f"締切時刻の解析に失敗しました: {e}")
                deadline_time_str = "不明"
                time_remaining = None
        else:
            print("警告: 締切予定時刻の情報がレース情報に含まれていません")
            deadline_time_str = "不明"
            time_remaining = None
        
        # boat_infoをDataFrameに変換
        boat_info_df = pd.DataFrame(boat_info)
        
        # 艇番を数値型に変換（マージのため）
        boat_info_df['艇番'] = pd.to_numeric(boat_info_df['艇番'])
        
        # 直前情報とレース基本情報を結合
        merged_df = pd.merge(
            race_base_info,
            boat_info_df,
            on='艇番',
            how='inner'
        )
        
        print(f"\n【情報を結合しました】")
        print(f"結合後のデータ: {merged_df.shape[0]}行 × {merged_df.shape[1]}列")
        
        # 結合後のデータが6艇分あるか確認
        if len(merged_df) != 6:
            print(f"警告: 結合後のデータが6艇分ではありません（{len(merged_df)}艇分）。データを確認してください。")
        
        # 天候情報を各行に追加
        for key, value in weather_info.items():
            merged_df[key] = value
        
        # 「レース場」を「会場」としてコピー（モデルの特徴量名と一致させるため）
        if 'レース場' in merged_df.columns and '会場' not in merged_df.columns:
            merged_df['会場'] = merged_df['レース場']
        
        # レースIDを生成（日付_会場_レース番号の形式）
        merged_df['レースID'] = int(f"{date}_{venue_code}_{race_number}")
        
        # データの前処理
        preprocessed_df = preprocess_data(merged_df)
        
        # モデルを使った予測
        prediction_results = predict_race_result(preprocessed_df)
        
        # 予測時刻の文字列を作成
        prediction_time_str = prediction_time.strftime('%H:%M')
        
        # 時刻情報の表示
        print("\n【予測結果】")
        print(f"予測時刻: {prediction_time_str} JST")
        
        if deadline_time_str != "不明":
            print(f"締切予定時刻: {deadline_time_str}")
            
            if time_remaining is not None:
                if time_remaining > 0:
                    print(f"締切まで残り約 {int(time_remaining)} 分")
                else:
                    print(f"締切時刻から約 {abs(int(time_remaining))} 分経過")
        
        # 予測結果の表示
        for _, row in prediction_results.iterrows():
            print(f"艇番 {int(row['艇番'])}: 勝率予測 {row['勝率予測']:.2%}, 単勝オッズ {row['単勝オッズ']:.1f}, 期待値 {row['期待値']:.2f}")
        
        # 予想信頼度の計算（上位2艇の勝率予測の合計）
        top2_probs_sum = prediction_results.iloc[:2]['勝率予測'].sum()
        print(f"\n【予想信頼度】: {top2_probs_sum:.2%}")
        
        # 期待値が1.0を超える艇（プラス期待値の艇）を抽出
        plus_ev_boats = prediction_results[prediction_results['期待値'] > 1.0]
        if not plus_ev_boats.empty:
            print("\n【おすすめ買い目】（期待値が1.0を超える艇）")
            for _, row in plus_ev_boats.iterrows():
                print(f"艇番 {int(row['艇番'])}: 期待値 {row['期待値']:.2f}")
        else:
            print("\n【おすすめ買い目】: なし（期待値が1.0を超える艇がありません）")
        
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
