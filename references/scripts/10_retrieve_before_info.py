#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import requests
from bs4 import BeautifulSoup
import re
import os

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
    if use_local_file and os.path.exists(use_local_file):
        print(f"ローカルファイル {use_local_file} からHTMLを読み込みます")
        try:
            with open(use_local_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"ローカルファイルの読み込みに失敗しました: {e}")
            print("URLからHTMLを取得します")
    
    try:
        print(f"URLからHTMLを取得します: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # エラーがあれば例外を発生させる
        return response.text
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

# 結果を表示する関数
def display_results(boat_info, weather_info, beforeinfo_url, odds_url):
    print(f"直前情報URL: {beforeinfo_url}")
    print(f"オッズ情報URL: {odds_url}\n")
    
    print("【天候情報】")
    for key, value in weather_info.items():
        print(f"{key}: {value}")
    print()
    
    print("【各艇の直前情報】")
    for boat in boat_info:
        print(f"艇番 {boat['艇番']}: 展示タイム {boat['展示タイム']}, 進入 {boat['進入']}, 単勝オッズ {boat['単勝オッズ']}")

def main():
    try:
        # コマンドライン引数の解析
        if len(sys.argv) != 4:
            print("使用方法: python 10_retrieve_before_info.py <日付(YYYYMMDD)> <競艇場名> <レース番号>")
            print("例: python 10_retrieve_before_info.py 20250402 桐生 12")
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
        beforeinfo_html = fetch_html(beforeinfo_url, use_local_file=beforeinfo_html_path if os.path.exists(beforeinfo_html_path) else None)
        
        # オッズ情報HTMLの取得
        odds_html = fetch_html(odds_url, use_local_file=odds_html_path if os.path.exists(odds_html_path) else None)
        
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
        
        # 結果を表示
        display_results(boat_info, weather_info, beforeinfo_url, odds_url)
        
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
