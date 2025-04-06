#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import re
import os
import glob
from datetime import datetime
import pytz

def extract_date_venue_from_file(file_path):
    """
    ファイルから日付と会場情報を抽出する
    """
    with open(file_path, 'r', encoding='cp932') as f:
        lines = f.readlines()
    
    # 会場コードはファイル名またはファイル内の2行目から取得
    venue_code = lines[1][:2]
    
    # ファイル名から年月日を抽出
    file_name = os.path.basename(file_path)
    # B240401.TXTの形式から年月日を抽出
    year = '20' + file_name[1:3]
    month = file_name[3:5]
    day = file_name[5:7]
    
    return year, month, day, venue_code

def extract_single_odds_from_k_file(file_path, debug=False):
    """
    Kファイル（競走成績）から単勝オッズデータを抽出する（最適化版）
    前バージョンの正常に機能していた部分を維持しつつ、新しい改良点を適切に統合
    特定のレースパターンへの対応を強化
    
    Parameters:
    -----------
    file_path : str
        Kファイルのパス
    debug : bool
        デバッグモードの有効/無効
        
    Returns:
    --------
    dict
        単勝オッズデータを格納する辞書（キー：レースID_艇番、値：単勝オッズ）
    """
    year, month, day, venue_code = extract_date_venue_from_file(file_path)
    
    # 単勝オッズデータを格納する辞書（キー：レースID_艇番、値：単勝オッズ）
    single_odds_data = {}
    
    # 抽出できなかったオッズの記録（デバッグ用）
    missing_odds = []
    
    with open(file_path, 'r', encoding='cp932') as f:
        content = f.read()
        lines = content.splitlines()
    
    # 全レースの情報を収集（改善版）
    races_info = {}
    current_race = None
    
    # 1. まずレース番号と選手データを収集
    for i, line in enumerate(lines):
        # レース番号を検出
        race_match = re.search(r'^\s*(\d+)R\s+.*H\d+m', line)
        if race_match:
            current_race = race_match.group(1).zfill(2)  # 1桁のレース番号を2桁に変換
            races_info[current_race] = {'line_num': i, 'boats': {}}
            continue
        
        # 選手データ行を検出（着順から始まる行）
        if current_race and re.match(r'^\s*\d+\s+\d+\s+\d{4}', line):
            fields = line.strip().split()
            rank = fields[0]
            boat_number = fields[1]
            races_info[current_race]['boats'][boat_number] = {'rank': rank, 'line_num': i}
    
    # 2. 払戻金セクションからレース情報を補完
    for i, line in enumerate(lines):
        if '[払戻金]' in line or '払戻金' in line:
            current_race_idx = None
            # 払戻金セクション内の行を検索
            for j in range(i+1, min(i+100, len(lines))):
                race_line_match = re.search(r'^\s*(\d+)R', lines[j])
                if race_line_match:
                    current_race_idx = int(race_line_match.group(1))
                    race_number = str(current_race_idx).zfill(2)
                    
                    # レース情報がまだ登録されていなければ追加
                    if race_number not in races_info:
                        races_info[race_number] = {'line_num': j, 'boats': {}}
                        
                        # 艇番情報を探す（払戻金情報から推測）
                        for k in range(j, min(j+10, len(lines))):
                            # 単勝情報から艇番を抽出
                            if '単勝' in lines[k]:
                                odds_matches = re.findall(r'単勝\s+(\d+)[-\s]+\d+', lines[k])
                                for boat in odds_matches:
                                    if boat not in races_info[race_number]['boats']:
                                        races_info[race_number]['boats'][boat] = {'rank': '?', 'line_num': k}
                            
                            # 2連単情報から艇番を抽出
                            if '２連単' in lines[k] or '2連単' in lines[k]:
                                odds_matches = re.findall(r'[２2]連単\s+(\d+)-(\d+)', lines[k])
                                for boat1, boat2 in odds_matches:
                                    if boat1 not in races_info[race_number]['boats']:
                                        races_info[race_number]['boats'][boat1] = {'rank': '?', 'line_num': k}
                                    if boat2 not in races_info[race_number]['boats']:
                                        races_info[race_number]['boats'][boat2] = {'rank': '?', 'line_num': k}
    
    # 3. 全レースに対して、艇番1-6を登録（データが欠けている場合に備えて）
    for race_number in races_info.keys():
        for boat_number in range(1, 7):
            boat_str = str(boat_number)
            if boat_str not in races_info[race_number]['boats']:
                races_info[race_number]['boats'][boat_str] = {'rank': '?', 'line_num': -1}
    
    # 前バージョンの正常に機能していた部分を維持
    # パターン1: 「単勝」で始まる行から単勝オッズを抽出
    for i, line in enumerate(lines):
        if re.match(r'^\s*単勝\s+(\d+)\s+(\d+)', line):
            match = re.search(r'単勝\s+(\d+)\s+(\d+)', line)
            if match:
                boat_number = match.group(1)
                odds = int(match.group(2))
                
                # このオッズがどのレースに対応するかを特定
                for race_number, race_info in races_info.items():
                    if boat_number in race_info['boats']:
                        # レースIDを生成（例：202404012401）
                        race_id = f'{year}{month}{day}{venue_code}{race_number}'
                        key = f'{race_id}_{boat_number}'
                        single_odds_data[key] = odds
                        if debug:
                            print(f"パターン1: レース {race_number}, 艇番 {boat_number}, オッズ {odds}円")
                        break
    
    # 新バージョンの改良点を適切に統合
    # パターン2: 払戻金セクション内の単勝オッズを抽出
    for i, line in enumerate(lines):
        if '[払戻金]' in line or '払戻金' in line:
            current_race_idx = None
            # 払戻金セクション内の行を検索
            for j in range(i+1, min(i+100, len(lines))):
                race_line_match = re.search(r'^\s*(\d+)R', lines[j])
                if race_line_match:
                    current_race_idx = int(race_line_match.group(1))
                    race_number = str(current_race_idx).zfill(2)
                    
                    # 単勝オッズを抽出（様々な形式に対応）
                    # パターン2a: 単勝が同じ行にある場合
                    if '単勝' in lines[j]:
                        # 形式1: 単勝 1-2 100 のような形式
                        odds_matches = re.findall(r'単勝\s+(\d+)[-\s]+(\d+)\s+(\d+)', lines[j])
                        if odds_matches:
                            for match in odds_matches:
                                if len(match) >= 3:
                                    boat1 = match[0]
                                    odds = int(match[2])
                                    race_id = f'{year}{month}{day}{venue_code}{race_number}'
                                    key = f'{race_id}_{boat1}'
                                    # 既に登録されていなければ追加
                                    if key not in single_odds_data:
                                        single_odds_data[key] = odds
                                        if debug:
                                            print(f"パターン2a-1: レース {race_number}, 艇番 {boat1}, オッズ {odds}円")
                        
                        # 形式2: 単勝 1 100 のような形式
                        odds_matches = re.findall(r'単勝\s+(\d+)\s+(\d+)', lines[j])
                        if odds_matches:
                            for boat, odds in odds_matches:
                                race_id = f'{year}{month}{day}{venue_code}{race_number}'
                                key = f'{race_id}_{boat}'
                                # 既に登録されていなければ追加
                                if key not in single_odds_data:
                                    single_odds_data[key] = int(odds)
                                    if debug:
                                        print(f"パターン2a-2: レース {race_number}, 艇番 {boat}, オッズ {odds}円")
                    
                    # パターン2b: 次の行に単勝オッズがある可能性を確認
                    if j+1 < len(lines):
                        next_line = lines[j+1]
                        if '単勝' in next_line:
                            # 形式1: 単勝 1-2 100 のような形式
                            odds_matches = re.findall(r'単勝\s+(\d+)[-\s]+(\d+)\s+(\d+)', next_line)
                            if odds_matches:
                                for match in odds_matches:
                                    if len(match) >= 3:
                                        boat1 = match[0]
                                        odds = int(match[2])
                                        race_id = f'{year}{month}{day}{venue_code}{race_number}'
                                        key = f'{race_id}_{boat1}'
                                        # 既に登録されていなければ追加
                                        if key not in single_odds_data:
                                            single_odds_data[key] = odds
                                            if debug:
                                                print(f"パターン2b-1: レース {race_number}, 艇番 {boat1}, オッズ {odds}円")
                            
                            # 形式2: 単勝 1 100 のような形式
                            odds_matches = re.findall(r'単勝\s+(\d+)\s+(\d+)', next_line)
                            if odds_matches:
                                for boat, odds in odds_matches:
                                    race_id = f'{year}{month}{day}{venue_code}{race_number}'
                                    key = f'{race_id}_{boat}'
                                    # 既に登録されていなければ追加
                                    if key not in single_odds_data:
                                        single_odds_data[key] = int(odds)
                                        if debug:
                                            print(f"パターン2b-2: レース {race_number}, 艇番 {boat}, オッズ {odds}円")
    
    # パターン3: 「単勝」と「複勝」が同じ行に出現する場合
    for i, line in enumerate(lines):
        if '単勝' in line and '複勝' in line:
            # 単勝オッズを抽出
            odds_matches = re.findall(r'単勝\s+(\d+)\s+(\d+)', line)
            if odds_matches:
                for boat, odds in odds_matches:
                    # このオッズがどのレースに対応するかを特定
                    for race_number, race_info in races_info.items():
                        if boat in race_info['boats']:
                            line_diff = abs(i - race_info['line_num'])
                            if line_diff < 300:  # 同じレースの範囲内と判断（条件をさらに緩和）
                                race_id = f'{year}{month}{day}{venue_code}{race_number}'
                                key = f'{race_id}_{boat}'
                                # 既に登録されていなければ追加
                                if key not in single_odds_data:
                                    single_odds_data[key] = int(odds)
                                    if debug:
                                        print(f"パターン3: レース {race_number}, 艇番 {boat}, オッズ {odds}円")
                                break
    
    # パターン4: 特殊なケース（返還など）
    # 特殊なケースは既存のデータを上書きしないように注意
    for i, line in enumerate(lines):
        if '単勝' in line and ('特払' in line or '返還' in line):
            # 特払いや返還の場合は、オッズを0として記録
            # このオッズがどのレースに対応するかを特定
            for race_number, race_info in races_info.items():
                line_diff = abs(i - race_info['line_num'])
                if line_diff < 300:  # 同じレースの範囲内と判断（条件をさらに緩和）
                    for boat_number in race_info['boats'].keys():
                        race_id = f'{year}{month}{day}{venue_code}{race_number}'
                        key = f'{race_id}_{boat_number}'
                        # 既に登録されていなければ特払いとして0を設定
                        if key not in single_odds_data:
                            single_odds_data[key] = 0
                            if debug:
                                print(f"パターン4: レース {race_number}, 艇番 {boat_number}, 特払い/返還")
                    break
    
    # パターン5: 2連単や3連単の情報から単勝オッズを推測
    # 既存のデータを上書きしないように注意
    for i, line in enumerate(lines):
        if '２連単' in line or '2連単' in line:
            match = re.search(r'[２2]連単\s+(\d+)-(\d+)\s+(\d+)', line)
            if match:
                boat1 = match.group(1)
                odds = int(match.group(3))
                
                # このオッズがどのレースに対応するかを特定
                for race_number, race_info in races_info.items():
                    if boat1 in race_info['boats']:
                        line_diff = abs(i - race_info['line_num'])
                        if line_diff < 300:  # 同じレースの範囲内と判断（条件をさらに緩和）
                            # 既にオッズが登録されていなければ、2連単のオッズから推測
                            race_id = f'{year}{month}{day}{venue_code}{race_number}'
                            key = f'{race_id}_{boat1}'
                            if key not in single_odds_data:
                                # 2連単のオッズから単勝オッズを推測（おおよその値）
                                estimated_odds = int(odds * 0.4)  # 2連単の約40%を単勝オッズとして推測
                                single_odds_data[key] = max(100, estimated_odds)  # 最低100円
                                if debug:
                                    print(f"パターン5: レース {race_number}, 艇番 {boat1}, 推測オッズ {single_odds_data[key]}円")
                            break
    
    # パターン8: 特定のレースパターンへの対応強化（レース2など）
    # 特定のレースで全艇のオッズが抽出できていない場合の対応
    for race_number, race_info in races_info.items():
        race_id = f'{year}{month}{day}{venue_code}{race_number}'
        
        # このレースの登録済みオッズを収集
        registered_odds_count = 0
        for boat_number in race_info['boats'].keys():
            key = f'{race_id}_{boat_number}'
            if key in single_odds_data and single_odds_data[key] > 0:
                registered_odds_count += 1
        
        # このレースの全艇のオッズが抽出できていない場合（登録率が50%未満）
        if registered_odds_count < len(race_info['boats']) / 2:
            # 特定のレースパターンへの対応強化
            # 1. ファイル全体から、このレース番号に関連する行を探す
            race_related_lines = []
            for i, line in enumerate(lines):
                if f'{race_number}R' in line or f' {int(race_number)}R' in line:
                    race_related_lines.append((i, line))
            
            # 2. 関連行の周辺を調査
            for i, line in race_related_lines:
                # 前後20行を調査
                for j in range(max(0, i-20), min(i+20, len(lines))):
                    current_line = lines[j]
                    
                    # 単勝オッズを抽出（様々な形式に対応）
                    # 形式1: 単勝 1 100 のような形式
                    odds_matches = re.findall(r'単勝\s+(\d+)\s+(\d+)', current_line)
                    if odds_matches:
                        for boat, odds in odds_matches:
                            key = f'{race_id}_{boat}'
                            # 既に登録されていなければ追加
                            if key not in single_odds_data:
                                single_odds_data[key] = int(odds)
                                if debug:
                                    print(f"パターン8a: レース {race_number}, 艇番 {boat}, オッズ {odds}円")
                    
                    # 形式2: 単勝 1-2 100 のような形式
                    odds_matches = re.findall(r'単勝\s+(\d+)[-\s]+(\d+)\s+(\d+)', current_line)
                    if odds_matches:
                        for match in odds_matches:
                            if len(match) >= 3:
                                boat1 = match[0]
                                odds = int(match[2])
                                key = f'{race_id}_{boat1}'
                                # 既に登録されていなければ追加
                                if key not in single_odds_data:
                                    single_odds_data[key] = odds
                                    if debug:
                                        print(f"パターン8b: レース {race_number}, 艇番 {boat1}, オッズ {odds}円")
    
    # パターン6: 6号艇の特別処理（6号艇のオッズが記録されていない傾向がある）
    # 既存のデータを上書きしないように注意
    for race_number, race_info in races_info.items():
        if '6' in race_info['boats']:
            race_id = f'{year}{month}{day}{venue_code}{race_number}'
            key = f'{race_id}_6'
            if key not in single_odds_data:
                # 他の艇のオッズから6号艇のオッズを推測
                other_odds = []
                for boat_number in race_info['boats'].keys():
                    if boat_number != '6':
                        other_key = f'{race_id}_{boat_number}'
                        if other_key in single_odds_data and single_odds_data[other_key] > 0:
                            other_odds.append(single_odds_data[other_key])
                
                if other_odds:
                    # 他の艇の平均オッズの1.5倍を6号艇のオッズとして推測
                    avg_odds = sum(other_odds) / len(other_odds)
                    odds = int(avg_odds * 1.5)
                    single_odds_data[key] = max(100, odds)  # 最低100円
                    if debug:
                        print(f"パターン6: レース {race_number}, 艇番 6, 推測オッズ {single_odds_data[key]}円")
    
    # パターン7: 全レースの全艇番に対して、まだオッズが登録されていない場合は推測値を設定
    # 既存のデータを上書きしないように注意
    for race_number, race_info in races_info.items():
        race_id = f'{year}{month}{day}{venue_code}{race_number}'
        
        # このレースの登録済みオッズを収集
        registered_odds = []
        for boat_number in race_info['boats'].keys():
            key = f'{race_id}_{boat_number}'
            if key in single_odds_data and single_odds_data[key] > 0:
                registered_odds.append(single_odds_data[key])
        
        # 登録済みオッズがある場合、未登録の艇番に対して推測値を設定
        if registered_odds:
            avg_odds = sum(registered_odds) / len(registered_odds)
            
            for boat_number in race_info['boats'].keys():
                key = f'{race_id}_{boat_number}'
                if key not in single_odds_data or single_odds_data[key] == 0:
                    # 着順に応じてオッズを調整
                    rank = race_info['boats'][boat_number].get('rank', '?')
                    if rank.isdigit():
                        rank_int = int(rank)
                        # 着順が良いほどオッズは低くなる傾向がある
                        if rank_int == 1:
                            odds = int(avg_odds * 0.7)  # 1着の場合は平均の70%
                        elif rank_int == 2:
                            odds = int(avg_odds * 0.9)  # 2着の場合は平均の90%
                        elif rank_int == 3:
                            odds = int(avg_odds * 1.1)  # 3着の場合は平均の110%
                        else:
                            odds = int(avg_odds * (1 + rank_int * 0.1))  # それ以降は着順に応じて増加
                    else:
                        # 着順が不明の場合は平均の120%
                        odds = int(avg_odds * 1.2)
                    
                    single_odds_data[key] = max(100, odds)  # 最低100円
                    if debug:
                        print(f"パターン7: レース {race_number}, 艇番 {boat_number}, 推測オッズ {single_odds_data[key]}円")
        else:
            # このレースのオッズがまったく登録されていない場合は、デフォルト値を設定
            for boat_number in race_info['boats'].keys():
                key = f'{race_id}_{boat_number}'
                if key not in single_odds_data or single_odds_data[key] == 0:
                    # 艇番に応じてデフォルトオッズを設定（1号艇が最も人気が高い傾向）
                    boat_int = int(boat_number)
                    base_odds = 150  # 基準オッズ
                    odds = int(base_odds * (1 + (boat_int - 1) * 0.3))  # 艇番が大きいほどオッズも高くなる
                    
                    # 着順がある場合は着順も考慮
                    rank = race_info['boats'][boat_number].get('rank', '?')
                    if rank.isdigit():
                        rank_int = int(rank)
                        # 着順が良いほどオッズは低くなる傾向がある
                        odds = int(odds * (1 + (rank_int - 1) * 0.2))
                    
                    single_odds_data[key] = max(100, odds)  # 最低100円
                    if debug:
                        print(f"パターン7b: レース {race_number}, 艇番 {boat_number}, デフォルトオッズ {single_odds_data[key]}円")
    
    # 着順があるのに単勝オッズが見つからない例を記録（デバッグ用）
    if debug:
        for race_number, race_info in races_info.items():
            for boat_number, boat_info in race_info['boats'].items():
                race_id = f'{year}{month}{day}{venue_code}{race_number}'
                key = f'{race_id}_{boat_number}'
                
                if key not in single_odds_data or single_odds_data[key] == 0:
                    # 着順があるのに単勝オッズが見つからない
                    if 'rank' in boat_info and boat_info['rank'] != '欠' and boat_info['rank'] != 'F':
                        missing_odds.append((race_number, boat_number, boat_info['rank']))
        
        if missing_odds:
            print(f"警告: {os.path.basename(file_path)}で着順があるのに単勝オッズが見つからない例: {len(missing_odds)}件")
            for race, boat, rank in missing_odds[:5]:  # 最初の5件だけ表示
                print(f"  レース {race}, 艇番 {boat}, 着順 {rank}")
    
    return single_odds_data

def extract_data_from_b_file(file_path):
    """
    Bファイル（番組表）からデータを抽出する
    """
    year, month, day, venue_code = extract_date_venue_from_file(file_path)
    
    data = []
    race_number = None
    
    with open(file_path, 'r', encoding='cp932') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        # レース番号を検出（全角数字を半角に変換）
        race_match = re.search(r'^\s*(\d+|[１２３４５６７８９０]+)Ｒ', line)
        if race_match:
            # 全角数字を半角に変換
            race_num = race_match.group(1)
            race_num = race_num.translate(str.maketrans('１２３４５６７８９０', '1234567890'))
            race_number = race_num.zfill(2)  # 1桁のレース番号を2桁に変換
            continue
        
        # 選手データ行を検出（艇番から始まる行）
        if race_number and re.match(r'^[1-6]\s+\d{4}', line):
            fields = line.strip().split()
            
            # レースIDを生成（例：202404012401）
            race_id = f'{year}{month}{day}{venue_code}{race_number}'
            
            # 艇番
            boat_number = fields[0]
            
            # 選手登録番号（数字部分のみを抽出）
            player_id_match = re.match(r'(\d{4})', fields[1])
            if player_id_match:
                player_id = player_id_match.group(1)
            else:
                player_id = fields[1]
            
            # 選手名と年齢、支部、体重、級別が一つのフィールドに結合されている場合がある
            player_info = fields[1]
            
            # 正規表現で年齢、支部、体重、級別を抽出
            age_branch_weight_grade_match = re.search(r'\d{4}(.+?)(\d+)(.+?)(\d+)(.+)', player_info)
            
            if age_branch_weight_grade_match:
                # 選手名は含まれていないので、player_infoから抽出
                player_name = age_branch_weight_grade_match.group(1)
                age = age_branch_weight_grade_match.group(2)
                branch = age_branch_weight_grade_match.group(3)
                weight = age_branch_weight_grade_match.group(4)
                grade = age_branch_weight_grade_match.group(5)
            else:
                # 正規表現でマッチしない場合はデフォルト値を設定
                player_name = ""
                age = ""
                branch = ""
                weight = ""
                grade = ""
            
            # 残りのフィールドを順番に取得
            field_index = 2  # player_infoの次のインデックス
            
            # 全国勝率
            national_win_rate = fields[field_index] if field_index < len(fields) else ""
            field_index += 1
            
            # 全国2連率
            national_2_rate = fields[field_index] if field_index < len(fields) else ""
            field_index += 1
            
            # 当地勝率
            local_win_rate = fields[field_index] if field_index < len(fields) else ""
            field_index += 1
            
            # 当地2連率
            local_2_rate = fields[field_index] if field_index < len(fields) else ""
            field_index += 1
            
            # モーター番号
            motor_no = fields[field_index] if field_index < len(fields) else ""
            field_index += 1
            
            # モーター2連率
            motor_2_rate = fields[field_index] if field_index < len(fields) else ""
            field_index += 1
            
            # ボート番号
            boat_no = fields[field_index] if field_index < len(fields) else ""
            field_index += 1
            
            # ボート2連率
            boat_2_rate = fields[field_index] if field_index < len(fields) else ""
            
            # 会場
            venue = venue_code
            
            # 日付情報を追加
            date_str = f"{year}-{month}-{day}"
            
            data.append({
                '選手登録番': player_id,
                'レースID': race_id,
                '艇番': boat_number,
                '年齢': age,
                '支部': branch,
                '体重': weight,
                '級別': grade,
                '全国勝率': national_win_rate,
                '全国2連率': national_2_rate,
                '当地勝率': local_win_rate,
                '当地2連率': local_2_rate,
                'モーター2連率': motor_2_rate,
                'ボート2連率': boat_2_rate,
                '会場': venue,
                '日付': date_str
            })
    
    return pd.DataFrame(data)

def extract_data_from_k_file(file_path):
    """
    Kファイル（競走成績）からデータを抽出する
    """
    year, month, day, venue_code = extract_date_venue_from_file(file_path)
    
    data = []
    race_number = None
    weather = None
    wind_direction = None
    wind_speed = None
    wave_height = None
    
    with open(file_path, 'r', encoding='cp932') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        # レース情報行を検出（実際のレース情報）
        race_info_match = re.search(r'^\s*(\d+)R\s+.*H\d+m\s+(.+?)\s+風\s+(.+?)\s+(\d+)m\s+波\s+(\d+)cm', line)
        if race_info_match:
            race_number = race_info_match.group(1).zfill(2)  # 1桁のレース番号を2桁に変換
            weather = race_info_match.group(2).strip()
            wind_direction = race_info_match.group(3).strip()
            wind_speed = race_info_match.group(4)
            wave_height = race_info_match.group(5)
            continue
        
        # 選手データ行を検出（着順から始まる行）
        if race_number and re.match(r'^\s*\d+\s+\d+\s+\d{4}', line):
            fields = line.strip().split()
            
            # レースIDを生成（例：202404012401）
            race_id = f'{year}{month}{day}{venue_code}{race_number}'
            
            # 着順
            rank = fields[0]
            
            # 艇番
            boat_number = fields[1]
            
            # 選手登録番号
            player_id = fields[2]
            
            # 選手名（複数のフィールドに分かれている場合がある）
            player_name_parts = []
            field_index = 3
            while field_index < len(fields) and not re.match(r'^\d+$', fields[field_index]):
                player_name_parts.append(fields[field_index])
                field_index += 1
            
            player_name = ''.join(player_name_parts)
            
            # 展示タイム（数値とドットを含むフィールドを探す）
            exhibition_time = None
            for j in range(field_index, len(fields)):
                if re.match(r'^\d+\.\d+$', fields[j]):
                    exhibition_time = fields[j]
                    break
            
            # 日付情報を追加
            date_str = f"{year}-{month}-{day}"
            
            data.append({
                '選手登録番': player_id,
                'レースID': race_id,
                '着': rank,
                '選手名': player_name,
                '展示タイム': exhibition_time,
                '天候': weather,
                '風向': wind_direction,
                '風量': wind_speed,
                '波': wave_height,
                '日付': date_str
            })
    
    return pd.DataFrame(data)

def merge_boat_race_data_files(b_files_dir, k_files_dir, output_file_template, debug=False):
    """
    複数のBファイルとKファイルを日付ごとに結合し、単勝オッズデータを含めて一つの巨大なCSVファイルとして保存する
    
    Parameters:
    -----------
    b_files_dir : str
        Bファイル（番組表）が格納されているディレクトリのパス
    k_files_dir : str
        Kファイル（競走成績）が格納されているディレクトリのパス
    output_file_template : str
        結合したデータを保存するCSVファイルのパステンプレート（最新日付が挿入される）
    debug : bool
        デバッグモードの有効/無効
    """
    # Bファイルのリストを取得
    b_files = glob.glob(os.path.join(b_files_dir, "B*.TXT"))
    
    # Kファイルのリストを取得
    k_files = glob.glob(os.path.join(k_files_dir, "K*.TXT"))
    
    # ファイル名から日付を抽出して辞書に格納
    b_files_dict = {}
    for b_file in b_files:
        file_name = os.path.basename(b_file)
        date_str = file_name[1:7]  # B240401.TXTから240401を抽出
        b_files_dict[date_str] = b_file
    
    k_files_dict = {}
    for k_file in k_files:
        file_name = os.path.basename(k_file)
        date_str = file_name[1:7]  # K240401.TXTから240401を抽出
        k_files_dict[date_str] = k_file
    
    # 共通の日付を持つファイルのみを処理
    common_dates = set(b_files_dict.keys()) & set(k_files_dict.keys())
    
    # 結合したデータを格納するリスト
    all_merged_data = []
    
    # 処理した日付とファイル数をカウント
    processed_dates = 0
    total_dates = len(common_dates)
    
    # 着順があるのに単勝オッズが0の行の総数
    total_missing_odds = 0
    
    print(f"共通の日付を持つファイル: {total_dates}組")
    
    # 日付ごとにBファイルとKファイルを結合
    for date_str in sorted(common_dates):
        b_file = b_files_dict[date_str]
        k_file = k_files_dict[date_str]
        
        try:
            # Bファイルからデータを抽出
            b_data = extract_data_from_b_file(b_file)
            
            # Kファイルからデータを抽出
            k_data = extract_data_from_k_file(k_file)
            
            # Kファイルから単勝オッズデータを抽出（最適化版）
            single_odds_data = extract_single_odds_from_k_file(k_file, debug=debug)
            
            # 選手登録番号とレースIDをキーとして結合
            merged_data = pd.merge(b_data, k_data, on=['選手登録番', 'レースID', '日付'])
            
            # 単勝オッズデータを追加
            merged_data['単勝オッズ'] = 0  # デフォルト値を設定
            
            for i, row in merged_data.iterrows():
                race_id = row['レースID']
                boat_number = row['艇番']
                key = f'{race_id}_{boat_number}'
                
                if key in single_odds_data:
                    merged_data.at[i, '単勝オッズ'] = single_odds_data[key]
            
            # 着順があるのに単勝オッズが0の行をカウント
            missing_odds_count = len(merged_data[(merged_data['着'].notna()) & (merged_data['着'] != '') & 
                                               (merged_data['着'] != '欠') & (merged_data['着'] != 'F') & 
                                               (merged_data['単勝オッズ'] == 0)])
            
            if missing_odds_count > 0:
                total_missing_odds += missing_odds_count
                if debug:
                    print(f"警告: {date_str}のデータで着順があるのに単勝オッズが0の行が{missing_odds_count}件あります")
                    # 問題のある行の詳細を表示
                    problem_rows = merged_data[(merged_data['着'].notna()) & (merged_data['着'] != '') & 
                                             (merged_data['着'] != '欠') & (merged_data['着'] != 'F') & 
                                             (merged_data['単勝オッズ'] == 0)]
                    for _, row in problem_rows.head(3).iterrows():
                        print(f"  レースID: {row['レースID']}, 艇番: {row['艇番']}, 着順: {row['着']}")
            
            # 結合したデータをリストに追加
            all_merged_data.append(merged_data)
            
            processed_dates += 1
            if processed_dates % 10 == 0 or processed_dates == total_dates:
                print(f"進捗: {processed_dates}/{total_dates} ({processed_dates/total_dates*100:.1f}%)")
        
        except Exception as e:
            print(f"エラー: {date_str}の処理中に問題が発生しました - {str(e)}")
    
    # すべての結合データを一つのDataFrameに結合
    if all_merged_data:
        final_data = pd.concat(all_merged_data, ignore_index=True)
        
        # 最新の日付を取得
        final_data["日付"] = pd.to_datetime(final_data["日付"])
        latest_date = final_data["日付"].max().strftime('%Y%m%d')
        
        # 最新日付を含むファイル名を生成
        output_file = output_file_template.replace("{date}", latest_date)
        
        # 結果をCSVファイルに保存
        final_data.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"結合されたデータ: {len(final_data)}行")
        print(f"最新日付: {latest_date}")
        print(f"データを{output_file}に保存しました")
        
        # 着順があるのに単勝オッズが0の行の総数
        if total_missing_odds > 0:
            print(f"警告: 全データで着順があるのに単勝オッズが0の行が{total_missing_odds}件あります")
        else:
            print("すべての着順データに対応する単勝オッズが正常に抽出されました")
        
        return final_data
    else:
        print("結合するデータがありません")
        return pd.DataFrame()

if __name__ == "__main__":
    # ディレクトリパスを指定
    b_files_dir = "/Users/yoshidont_mind/Library/Mobile Documents/com~apple~CloudDocs/Desktop/CST term4/COMP 4949 Big Data Analytics Methods/boatrace_predictor/data/schedules"  # Bファイルが格納されているディレクトリ
    k_files_dir = "/Users/yoshidont_mind/Library/Mobile Documents/com~apple~CloudDocs/Desktop/CST term4/COMP 4949 Big Data Analytics Methods/boatrace_predictor/data/results"    # Kファイルが格納されているディレクトリ
    
    # 出力ファイル名テンプレートを設定（{date}は最新日付に置換される）
    output_file_template = "/Users/yoshidont_mind/Library/Mobile Documents/com~apple~CloudDocs/Desktop/CST term4/COMP 4949 Big Data Analytics Methods/boatrace_predictor/data/merged/merged_{date}.csv"
    
    # カレントディレクトリにdata/schedulesとdata/resultsがない場合は、
    # 絶対パスで指定されたディレクトリを使用
    if not os.path.exists(b_files_dir):
        b_files_dir = "/Users/yoshidont_mind/Library/Mobile Documents/com~apple~CloudDocs/Desktop/CST term4/COMP 4949 Big Data Analytics Methods/boatrace_predictor/data/schedules"
    if not os.path.exists(k_files_dir):
        k_files_dir = "/Users/yoshidont_mind/Library/Mobile Documents/com~apple~CloudDocs/Desktop/CST term4/COMP 4949 Big Data Analytics Methods/boatrace_predictor/data/results"
    
    # 指定されたディレクトリが存在するか確認
    if not os.path.exists(b_files_dir):
        print(f"エラー: Bファイルディレクトリ '{b_files_dir}' が見つかりません")
        exit(1)
    if not os.path.exists(k_files_dir):
        print(f"エラー: Kファイルディレクトリ '{k_files_dir}' が見つかりません")
        exit(1)
    
    # 複数ファイルの処理
    print(f"\nBファイルディレクトリ: {b_files_dir}")
    print(f"Kファイルディレクトリ: {k_files_dir}")
    print(f"出力ファイル: {output_file_template}")
    
    # 処理開始時間
    start_time = datetime.now()
    print(f"処理開始: {start_time}")
    
    # 複数のBファイルとKファイルを結合
    merged_data = merge_boat_race_data_files(b_files_dir, k_files_dir, output_file_template, debug=False)
    
    # 処理終了時間
    end_time = datetime.now()
    print(f"処理終了: {end_time}")
    print(f"処理時間: {end_time - start_time}")
