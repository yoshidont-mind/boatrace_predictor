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

# 競艇場名からコードを取得する関数（逆引き辞書）
venue_to_code = {v: k for k, v in boatrace_venues.items()}

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
def fetch_html(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # エラーがあれば例外を発生させる
        return response.text, datetime.now()
    except requests.exceptions.RequestException as e:
        print(f"エラー: HTMLの取得に失敗しました: {e}")
        return None, datetime.now()

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
    
    return weather_info

# ====== 単一レースの予測関数 ======
def predict_single_race(race_id):
    try:
        date_str = race_id[:8]  # yyyymmdd
        stadium_id = int(race_id[8:10])
        race_no = int(race_id[10:])
        target_date = datetime.strptime(date_str, "%Y%m%d").date()
        stadium_name = boatrace_venues.get(f"{stadium_id:02}", "不明")

        # JSTタイムゾーンの設定
        jst = pytz.timezone('Asia/Tokyo')
        
        # ====== レース基本情報の取得 ======
        scraper = PyJPBoatrace()
        race_info = scraper.get_race_info(d=target_date, stadium=stadium_id, race=race_no)

        # ====== 直前情報とオッズの取得（スクレイピング） ======
        # 直前情報ページのURLを構築
        venue_code = f"{stadium_id:02}"
        beforeinfo_url = build_beforeinfo_url(date_str, venue_code, race_no)
        odds_url = build_odds_url(date_str, venue_code, race_no)
        
        # HTMLの取得
        beforeinfo_html, _ = fetch_html(beforeinfo_url)
        odds_html, prediction_time = fetch_html(odds_url)
        
        # HTMLが取得できなかった場合はエラー
        if not beforeinfo_html or not odds_html:
            raise Exception("直前情報またはオッズ情報の取得に失敗しました")
        
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
        
        # ====== レース基本情報をDataFrameに変換 ======
        race_entries = []
        for boat_no in range(1, 7):
            boat_key = f"boat{boat_no}"
            boat_data = race_info.get(boat_key, {})
            
            # boatinfoから対応する艇の情報を取得
            boat_info_item = next((b for b in boat_info if int(b['艇番']) == boat_no), {})
            
            race_entries.append({
                "レースID": race_id,
                "日付": target_date,
                "会場": stadium_name, # 会場名を追加
                "艇番": boat_no,
                "選手名": boat_data.get("name", ""),
                "級別": boat_data.get("class", ""),
                "支部": boat_data.get("branch", ""),
                "全国勝率": boat_data.get("global_win_pt", None),
                "全国2連率": boat_data.get("global_in2nd", None), # 全国2連率を追加
                "当地勝率": boat_data.get("local_win_pt", None),
                "当地2連率": boat_data.get("local_in2nd", None), # 当地2連率を追加
                "モーター番号": boat_data.get("motor", None),
                "モーター2連率": boat_data.get("motor_in2nd", None),
                "ボート番号": boat_data.get("boat", None),
                "ボート2連率": boat_data.get("boat_in2nd", None),
                "展示タイム": boat_info_item.get('展示タイム', None),
                "進入": boat_info_item.get('進入', None),
                "単勝オッズ": float(boat_info_item.get('単勝オッズ', 0)) if boat_info_item.get('単勝オッズ', 'N/A') != 'N/A' else None,
                "体重": boat_data.get("weight", None),
                "年齢": boat_data.get("age", None),
            })

        df_race = pd.DataFrame(race_entries)
        
        # 天候情報を各行に追加
        for key, value in weather_info.items():
            df_race[key] = value
            
        # 「風量」を数値型に変換（「3m」→ 3）
        if "風量" in df_race.columns:
            df_race["風量"] = df_race["風量"].astype(str).str.replace('m', '').str.replace('M', '')
            df_race["風量"] = pd.to_numeric(df_race["風量"], errors="coerce")
        
        # 「波」を数値型に変換（「3cm」→ 3）
        if "波" in df_race.columns:
            df_race["波"] = df_race["波"].astype(str).str.replace('cm', '').str.replace('CM', '')
            df_race["波"] = pd.to_numeric(df_race["波"], errors="coerce")

        # 重要データの欠損チェック
        for col in ["全国勝率", "全国2連率", "当地勝率", "当地2連率", "モーター2連率", "ボート2連率"]:
            if col not in df_race.columns or df_race[col].isnull().all():
                print(f"重要特徴 '{col}' のデータがありません")
                if col not in df_race.columns:
                    # 特徴量が存在しない場合は追加
                    df_race[col] = 0
                    print(f"'{col}' を追加しました（値：0）")
        
        # 日付情報が無い場合は追加
        if "日付" in df_race.columns:
            if not pd.api.types.is_datetime64_any_dtype(df_race["日付"]):
                df_race["日付"] = pd.to_datetime(df_race["日付"], errors="coerce")
        else:
            # 現在の日付を設定
            df_race["日付"] = pd.to_datetime(target_date)
        
        print("前処理を開始します...")
        # ====== 前処理 ======
        df_processed = preprocess_boatrace_dataframe(df_race.copy())
        print("前処理が完了しました")

        # ====== モデル読み込み ======
        try:
            model_path = Path("models/model_20250401.pkl")
            print(f"モデルを読み込みます: {model_path}")
            
            # pickelではなくLightGBM直接の読み込みを試みる
            import lightgbm as lgb
            model = lgb.Booster(model_file=str(model_path))
            print("モデルの読み込みが完了しました (LightGBM直接読み込み)")
        except Exception as e:
            print(f"LightGBMでのモデル読み込みに失敗しました: {e}")
            print("バックアップとしてモデルをピクルファイルとして読み込みます...")
            
            try:
                # 古いpickleファイルを読むためのprotocol指定
                with open(model_path, "rb") as f:
                    model = pickle.load(f, encoding='latin1')
                print("pickleでのモデル読み込みが完了しました")
            except Exception as e:
                print(f"モデルのロードに失敗しました: {e}")
                raise Exception(f"モデルファイルの読み込みに失敗しました: {e}")

        # ====== 予測 ======
        # モデルの特徴量と同じ順序で特徴量を準備
        feature_cols = [
            "支部", "級別", "艇番", "会場", "風量", "波", "月", "曜日",
            "全国勝率_dev", "全国2連率_dev", "当地勝率_dev", "当地2連率_dev",
            "モーター2連率_dev", "ボート2連率_dev", "展示タイム_dev"
        ]

        # 各特徴量が存在するか確認
        print("特徴量のチェックを行います...")
        for feature in feature_cols:
            if feature not in df_processed.columns:
                print(f"警告: 特徴量 '{feature}' がデータフレームに存在しません")
                # 特徴量がない場合は適当な値で埋める
                if feature.endswith('_dev'):
                    df_processed[feature] = 50  # 偏差値の中央値
                else:
                    df_processed[feature] = 0
                print(f"特徴量 '{feature}' を追加しました")

        # 予測 (LightGBMはpredict_probaではなくpredictを使用)
        print("予測を実行します...")
        try:
            # Boosterオブジェクトの場合はpredict()を使用
            df_processed["pred_proba"] = model.predict(df_processed[feature_cols])
            print("予測が完了しました")
        except Exception as e:
            print(f"予測中にエラーが発生しました: {e}")
            raise

        # 勝率を正規化（合計1になるように）
        print("勝率を正規化します...")
        sum_proba = df_processed["pred_proba"].sum()
        if sum_proba > 0:
            df_processed["pred_proba"] = df_processed["pred_proba"] / sum_proba
        print("勝率の正規化が完了しました")

        # 期待値の計算
        df_processed["期待値"] = df_processed["pred_proba"] * df_processed["単勝オッズ"]

        # 結果のデータフレームを作成
        df_result = df_processed[["艇番", "選手名", "pred_proba", "単勝オッズ", "期待値"]].sort_values("期待値", ascending=False)
        df_result.columns = ["艇番", "選手名", "勝率(予測)", "単勝オッズ", "期待値"]

        # 勝率を%表示に変換
        df_result["勝率(予測)"] = (df_result["勝率(予測)"] * 100).round(1).astype(str) + '%'
        
        # 期待値表示を見やすくする
        df_result["期待値"] = df_result["期待値"].round(2)

        # 単勝オッズは小数点以下1桁に丸める
        df_result["単勝オッズ"] = df_result["単勝オッズ"].round(1)

        # 予測時刻をJSTに変換
        prediction_time_jst = prediction_time.astimezone(jst) if prediction_time.tzinfo else jst.localize(prediction_time)

        return df_result, prediction_time_jst

    except Exception as e:
        print(f"エラー: {e}")
        traceback.print_exc()
        return None, datetime.now()
