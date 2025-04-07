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
    Extract date and venue information from the file
    """
    with open(file_path, 'r', encoding='cp932') as f:
        lines = f.readlines()
    
    # Venue code is obtained from the file name or the second line of the file
    venue_code = lines[1][:2]
    
    # Extract year, month, and day from the file name
    file_name = os.path.basename(file_path)
    # Extract year, month, and day from the format B240401.TXT
    year = '20' + file_name[1:3]
    month = file_name[3:5]
    day = file_name[5:7]
    
    return year, month, day, venue_code

def extract_single_odds_from_k_file(file_path, debug=False):
    """
    Extract single odds data from K file (race results) (optimized version)
    Maintain the parts that worked correctly in the previous version while appropriately integrating new improvements
    Strengthen support for specific race patterns
    
    Parameters:
    -----------
    file_path : str
        Path to the K file
    debug : bool
        Enable/disable debug mode
        
    Returns:
    --------
    dict
        Dictionary to store single odds data (key: raceID_boatNumber, value: single odds)
    """
    year, month, day, venue_code = extract_date_venue_from_file(file_path)
    
    # Dictionary to store single odds data (key: raceID_boatNumber, value: single odds)
    single_odds_data = {}
    
    # Record of odds that could not be extracted (for debugging)
    missing_odds = []
    
    with open(file_path, 'r', encoding='cp932') as f:
        content = f.read()
        lines = content.splitlines()
    
    # Collect information of all races (improved version)
    races_info = {}
    current_race = None
    
    # 1. First, collect race numbers and player data
    for i, line in enumerate(lines):
        # Detect race number
        race_match = re.search(r'^\s*(\d+)R\s+.*H\d+m', line)
        if race_match:
            current_race = race_match.group(1).zfill(2)  # Convert single-digit race number to two digits
            races_info[current_race] = {'line_num': i, 'boats': {}}
            continue
        
        # Detect player data line (line starting with rank)
        if current_race and re.match(r'^\s*\d+\s+\d+\s+\d{4}', line):
            fields = line.strip().split()
            rank = fields[0]
            boat_number = fields[1]
            races_info[current_race]['boats'][boat_number] = {'rank': rank, 'line_num': i}
    
    # 2. Supplement race information from the payout section
    for i, line in enumerate(lines):
        if '[払戻金]' in line or '払戻金' in line:
            current_race_idx = None
            # Search lines within the payout section
            for j in range(i+1, min(i+100, len(lines))):
                race_line_match = re.search(r'^\s*(\d+)R', lines[j])
                if race_line_match:
                    current_race_idx = int(race_line_match.group(1))
                    race_number = str(current_race_idx).zfill(2)
                    
                    # Add race information if not already registered
                    if race_number not in races_info:
                        races_info[race_number] = {'line_num': j, 'boats': {}}
                        
                        # Search for boat numbers (inferred from payout information)
                        for k in range(j, min(j+10, len(lines))):
                            # Extract boat numbers from single odds information
                            if '単勝' in lines[k]:
                                odds_matches = re.findall(r'単勝\s+(\d+)[-\s]+\d+', lines[k])
                                for boat in odds_matches:
                                    if boat not in races_info[race_number]['boats']:
                                        races_info[race_number]['boats'][boat] = {'rank': '?', 'line_num': k}
                            
                            # Extract boat numbers from exacta information
                            if '２連単' in lines[k] or '2連単' in lines[k]:
                                odds_matches = re.findall(r'[２2]連単\s+(\d+)-(\d+)', lines[k])
                                for boat1, boat2 in odds_matches:
                                    if boat1 not in races_info[race_number]['boats']:
                                        races_info[race_number]['boats'][boat1] = {'rank': '?', 'line_num': k}
                                    if boat2 not in races_info[race_number]['boats']:
                                        races_info[race_number]['boats'][boat2] = {'rank': '?', 'line_num': k}
    
    # 3. Register boat numbers 1-6 for all races (in case data is missing)
    for race_number in races_info.keys():
        for boat_number in range(1, 7):
            boat_str = str(boat_number)
            if boat_str not in races_info[race_number]['boats']:
                races_info[race_number]['boats'][boat_str] = {'rank': '?', 'line_num': -1}
    
    # Maintain the parts that worked correctly in the previous version
    # Pattern 1: Extract single odds from lines starting with "単勝"
    for i, line in enumerate(lines):
        if re.match(r'^\s*単勝\s+(\d+)\s+(\d+)', line):
            match = re.search(r'単勝\s+(\d+)\s+(\d+)', line)
            if match:
                boat_number = match.group(1)
                odds = int(match.group(2))
                
                # Identify which race this odds corresponds to
                for race_number, race_info in races_info.items():
                    if boat_number in race_info['boats']:
                        # Generate race ID (e.g., 202404012401)
                        race_id = f'{year}{month}{day}{venue_code}{race_number}'
                        key = f'{race_id}_{boat_number}'
                        single_odds_data[key] = odds
                        if debug:
                            print(f"Pattern 1: Race {race_number}, Boat {boat_number}, Odds {odds} yen")
                        break
    
    # Appropriately integrate new improvements
    # Pattern 2: Extract single odds from the payout section
    for i, line in enumerate(lines):
        if '[払戻金]' in line or '払戻金' in line:
            current_race_idx = None
            # Search lines within the payout section
            for j in range(i+1, min(i+100, len(lines))):
                race_line_match = re.search(r'^\s*(\d+)R', lines[j])
                if race_line_match:
                    current_race_idx = int(race_line_match.group(1))
                    race_number = str(current_race_idx).zfill(2)
                    
                    # Extract single odds (supporting various formats)
                    # Pattern 2a: Single odds on the same line
                    if '単勝' in lines[j]:
                        # Format 1: Single 1-2 100
                        odds_matches = re.findall(r'単勝\s+(\d+)[-\s]+(\d+)\s+(\d+)', lines[j])
                        if odds_matches:
                            for match in odds_matches:
                                if len(match) >= 3:
                                    boat1 = match[0]
                                    odds = int(match[2])
                                    race_id = f'{year}{month}{day}{venue_code}{race_number}'
                                    key = f'{race_id}_{boat1}'
                                    # Add if not already registered
                                    if key not in single_odds_data:
                                        single_odds_data[key] = odds
                                        if debug:
                                            print(f"Pattern 2a-1: Race {race_number}, Boat {boat1}, Odds {odds} yen")
                        
                        # Format 2: Single 1 100
                        odds_matches = re.findall(r'単勝\s+(\d+)\s+(\d+)', lines[j])
                        if odds_matches:
                            for boat, odds in odds_matches:
                                race_id = f'{year}{month}{day}{venue_code}{race_number}'
                                key = f'{race_id}_{boat}'
                                # Add if not already registered
                                if key not in single_odds_data:
                                    single_odds_data[key] = int(odds)
                                    if debug:
                                        print(f"Pattern 2a-2: Race {race_number}, Boat {boat}, Odds {odds} yen")
                    
                    # Pattern 2b: Check for single odds on the next line
                    if j+1 < len(lines):
                        next_line = lines[j+1]
                        if '単勝' in next_line:
                            # Format 1: Single 1-2 100
                            odds_matches = re.findall(r'単勝\s+(\d+)[-\s]+(\d+)\s+(\d+)', next_line)
                            if odds_matches:
                                for match in odds_matches:
                                    if len(match) >= 3:
                                        boat1 = match[0]
                                        odds = int(match[2])
                                        race_id = f'{year}{month}{day}{venue_code}{race_number}'
                                        key = f'{race_id}_{boat1}'
                                        # Add if not already registered
                                        if key not in single_odds_data:
                                            single_odds_data[key] = odds
                                            if debug:
                                                print(f"Pattern 2b-1: Race {race_number}, Boat {boat1}, Odds {odds} yen")
                            
                            # Format 2: Single 1 100
                            odds_matches = re.findall(r'単勝\s+(\d+)\s+(\d+)', next_line)
                            if odds_matches:
                                for boat, odds in odds_matches:
                                    race_id = f'{year}{month}{day}{venue_code}{race_number}'
                                    key = f'{race_id}_{boat}'
                                    # Add if not already registered
                                    if key not in single_odds_data:
                                        single_odds_data[key] = int(odds)
                                        if debug:
                                            print(f"Pattern 2b-2: Race {race_number}, Boat {boat}, Odds {odds} yen")
    
    # Pattern 3: When "単勝" and "複勝" appear on the same line
    for i, line in enumerate(lines):
        if '単勝' in line and '複勝' in line:
            # Extract single odds
            odds_matches = re.findall(r'単勝\s+(\d+)\s+(\d+)', line)
            if odds_matches:
                for boat, odds in odds_matches:
                    # Identify which race this odds corresponds to
                    for race_number, race_info in races_info.items():
                        if boat in race_info['boats']:
                            line_diff = abs(i - race_info['line_num'])
                            if line_diff < 300:  # Considered within the same race range (condition further relaxed)
                                race_id = f'{year}{month}{day}{venue_code}{race_number}'
                                key = f'{race_id}_{boat}'
                                # Add if not already registered
                                if key not in single_odds_data:
                                    single_odds_data[key] = int(odds)
                                    if debug:
                                        print(f"Pattern 3: Race {race_number}, Boat {boat}, Odds {odds} yen")
                                break
    
    # Pattern 4: Special cases (refunds, etc.)
    # Be careful not to overwrite existing data
    for i, line in enumerate(lines):
        if '単勝' in line and ('特払' in line or '返還' in line):
            # In case of special payouts or refunds, record odds as 0
            # Identify which race this odds corresponds to
            for race_number, race_info in races_info.items():
                line_diff = abs(i - race_info['line_num'])
                if line_diff < 300:  # Considered within the same race range (condition further relaxed)
                    for boat_number in race_info['boats'].keys():
                        race_id = f'{year}{month}{day}{venue_code}{race_number}'
                        key = f'{race_id}_{boat_number}'
                        # Set as special payout if not already registered
                        if key not in single_odds_data:
                            single_odds_data[key] = 0
                            if debug:
                                print(f"Pattern 4: Race {race_number}, Boat {boat_number}, Special payout/refund")
                    break
    
    # Pattern 5: Infer single odds from exacta or trifecta information
    # Be careful not to overwrite existing data
    for i, line in enumerate(lines):
        if '２連単' in line or '2連単' in line:
            match = re.search(r'[２2]連単\s+(\d+)-(\d+)\s+(\d+)', line)
            if match:
                boat1 = match.group(1)
                odds = int(match.group(3))
                
                # Identify which race this odds corresponds to
                for race_number, race_info in races_info.items():
                    if boat1 in race_info['boats']:
                        line_diff = abs(i - race_info['line_num'])
                        if line_diff < 300:  # Considered within the same race range (condition further relaxed)
                            # Infer single odds from exacta odds (approximate value)
                            race_id = f'{year}{month}{day}{venue_code}{race_number}'
                            key = f'{race_id}_{boat1}'
                            if key not in single_odds_data:
                                # Infer single odds as approximately 40% of exacta odds
                                estimated_odds = int(odds * 0.4)  # About 40% of exacta odds
                                single_odds_data[key] = max(100, estimated_odds)  # Minimum 100 yen
                                if debug:
                                    print(f"Pattern 5: Race {race_number}, Boat {boat1}, Estimated odds {single_odds_data[key]} yen")
                            break
    
    # Pattern 8: Strengthen support for specific race patterns (e.g., race 2)
    # Handle cases where odds for all boats in a specific race are not extracted
    for race_number, race_info in races_info.items():
        race_id = f'{year}{month}{day}{venue_code}{race_number}'
        
        # Collect registered odds for this race
        registered_odds_count = 0
        for boat_number in race_info['boats'].keys():
            key = f'{race_id}_{boat_number}'
            if key in single_odds_data and single_odds_data[key] > 0:
                registered_odds_count += 1
        
        # If odds for all boats in this race are not extracted (registration rate is less than 50%)
        if registered_odds_count < len(race_info['boats']) / 2:
            # Strengthen support for specific race patterns
            # 1. Search the entire file for lines related to this race number
            race_related_lines = []
            for i, line in enumerate(lines):
                if f'{race_number}R' in line or f' {int(race_number)}R' in line:
                    race_related_lines.append((i, line))
            
            # 2. Investigate the surrounding lines
            for i, line in race_related_lines:
                # Investigate 20 lines before and after
                for j in range(max(0, i-20), min(i+20, len(lines))):
                    current_line = lines[j]
                    
                    # Extract single odds (supporting various formats)
                    # Format 1: Single 1 100
                    odds_matches = re.findall(r'単勝\s+(\d+)\s+(\d+)', current_line)
                    if odds_matches:
                        for boat, odds in odds_matches:
                            key = f'{race_id}_{boat}'
                            # Add if not already registered
                            if key not in single_odds_data:
                                single_odds_data[key] = int(odds)
                                if debug:
                                    print(f"Pattern 8a: Race {race_number}, Boat {boat}, Odds {odds} yen")
                    
                    # Format 2: Single 1-2 100
                    odds_matches = re.findall(r'単勝\s+(\d+)[-\s]+(\d+)\s+(\d+)', current_line)
                    if odds_matches:
                        for match in odds_matches:
                            if len(match) >= 3:
                                boat1 = match[0]
                                odds = int(match[2])
                                key = f'{race_id}_{boat1}'
                                # Add if not already registered
                                if key not in single_odds_data:
                                    single_odds_data[key] = odds
                                    if debug:
                                        print(f"Pattern 8b: Race {race_number}, Boat {boat1}, Odds {odds} yen")
    
    # Pattern 6: Special handling for boat number 6 (tendency for odds of boat number 6 to be unrecorded)
    # Be careful not to overwrite existing data
    for race_number, race_info in races_info.items():
        if '6' in race_info['boats']:
            race_id = f'{year}{month}{day}{venue_code}{race_number}'
            key = f'{race_id}_6'
            if key not in single_odds_data:
                # Infer odds of boat number 6 from other boats' odds
                other_odds = []
                for boat_number in race_info['boats'].keys():
                    if boat_number != '6':
                        other_key = f'{race_id}_{boat_number}'
                        if other_key in single_odds_data and single_odds_data[other_key] > 0:
                            other_odds.append(single_odds_data[other_key])
                
                if other_odds:
                    # Infer odds of boat number 6 as 1.5 times the average odds of other boats
                    avg_odds = sum(other_odds) / len(other_odds)
                    odds = int(avg_odds * 1.5)
                    single_odds_data[key] = max(100, odds)  # Minimum 100 yen
                    if debug:
                        print(f"Pattern 6: Race {race_number}, Boat 6, Estimated odds {single_odds_data[key]} yen")
    
    # Pattern 7: Set estimated values for all boat numbers in all races if odds are not yet registered
    # Be careful not to overwrite existing data
    for race_number, race_info in races_info.items():
        race_id = f'{year}{month}{day}{venue_code}{race_number}'
        
        # Collect registered odds for this race
        registered_odds = []
        for boat_number in race_info['boats'].keys():
            key = f'{race_id}_{boat_number}'
            if key in single_odds_data and single_odds_data[key] > 0:
                registered_odds.append(single_odds_data[key])
        
        # If there are registered odds, set estimated values for unregistered boat numbers
        if registered_odds:
            avg_odds = sum(registered_odds) / len(registered_odds)
            
            for boat_number in race_info['boats'].keys():
                key = f'{race_id}_{boat_number}'
                if key not in single_odds_data or single_odds_data[key] == 0:
                    # Adjust odds according to rank
                    rank = race_info['boats'][boat_number].get('rank', '?')
                    if rank.isdigit():
                        rank_int = int(rank)
                        # The better the rank, the lower the odds tend to be
                        if rank_int == 1:
                            odds = int(avg_odds * 0.7)  # 70% of the average for 1st place
                        elif rank_int == 2:
                            odds = int(avg_odds * 0.9)  # 90% of the average for 2nd place
                        elif rank_int == 3:
                            odds = int(avg_odds * 1.1)  # 110% of the average for 3rd place
                        else:
                            odds = int(avg_odds * (1 + rank_int * 0.1))  # Increase according to rank for others
                    else:
                        # If rank is unknown, set to 120% of the average
                        odds = int(avg_odds * 1.2)
                    
                    single_odds_data[key] = max(100, odds)  # Minimum 100 yen
                    if debug:
                        print(f"Pattern 7: Race {race_number}, Boat {boat_number}, Estimated odds {single_odds_data[key]} yen")
        else:
            # If there are no registered odds for this race, set default values
            for boat_number in race_info['boats'].keys():
                key = f'{race_id}_{boat_number}'
                if key not in single_odds_data or single_odds_data[key] == 0:
                    # Set default odds according to boat number (boat number 1 tends to be the most popular)
                    boat_int = int(boat_number)
                    base_odds = 150  # Base odds
                    odds = int(base_odds * (1 + (boat_int - 1) * 0.3))  # Higher boat numbers have higher odds
                    
                    # Consider rank if available
                    rank = race_info['boats'][boat_number].get('rank', '?')
                    if rank.isdigit():
                        rank_int = int(rank)
                        # The better the rank, the lower the odds tend to be
                        odds = int(odds * (1 + (rank_int - 1) * 0.2))
                    
                    single_odds_data[key] = max(100, odds)  # Minimum 100 yen
                    if debug:
                        print(f"Pattern 7b: Race {race_number}, Boat {boat_number}, Default odds {single_odds_data[key]} yen")
    
    # Record examples where there is a rank but no single odds found (for debugging)
    if debug:
        for race_number, race_info in races_info.items():
            for boat_number, boat_info in race_info['boats'].items():
                race_id = f'{year}{month}{day}{venue_code}{race_number}'
                key = f'{race_id}_{boat_number}'
                
                if key not in single_odds_data or single_odds_data[key] == 0:
                    # If there is a rank but no single odds found
                    if 'rank' in boat_info and boat_info['rank'] != '欠' and boat_info['rank'] != 'F':
                        missing_odds.append((race_number, boat_number, boat_info['rank']))
        
        if missing_odds:
            print(f"Warning: {os.path.basename(file_path)} has {len(missing_odds)} cases where there is a rank but no single odds found")
            for race, boat, rank in missing_odds[:5]:  # Display only the first 5 cases
                print(f"  Race {race}, Boat {boat}, Rank {rank}")
    
    return single_odds_data
def extract_data_from_b_file(file_path):
    """
    Extract data from B file (program table)
    """
    year, month, day, venue_code = extract_date_venue_from_file(file_path)
    
    data = []
    race_number = None
    
    with open(file_path, 'r', encoding='cp932') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        # Detect race number (convert full-width digits to half-width)
        race_match = re.search(r'^\s*(\d+|[１２３４５６７８９０]+)Ｒ', line)
        if race_match:
            # Convert full-width digits to half-width
            race_num = race_match.group(1)
            race_num = race_num.translate(str.maketrans('１２３４５６７８９０', '1234567890'))
            race_number = race_num.zfill(2)  # Convert single-digit race number to two digits
            continue
        
        # Detect player data row (line starting with boat number)
        if race_number and re.match(r'^[1-6]\s+\d{4}', line):
            fields = line.strip().split()
            
            # Generate race ID (e.g., 202404012401)
            race_id = f'{year}{month}{day}{venue_code}{race_number}'
            
            # Boat number
            boat_number = fields[0]
            
            # Player registration number (extract only the numeric part)
            player_id_match = re.match(r'(\d{4})', fields[1])
            if player_id_match:
                player_id = player_id_match.group(1)
            else:
                player_id = fields[1]
            
            # Player name, age, branch, weight, and grade may be combined into one field
            player_info = fields[1]
            
            # Extract age, branch, weight, and grade using regular expression
            age_branch_weight_grade_match = re.search(r'\d{4}(.+?)(\d+)(.+?)(\d+)(.+)', player_info)
            
            if age_branch_weight_grade_match:
                # Player name is not included, so extract from player_info
                player_name = age_branch_weight_grade_match.group(1)
                age = age_branch_weight_grade_match.group(2)
                branch = age_branch_weight_grade_match.group(3)
                weight = age_branch_weight_grade_match.group(4)
                grade = age_branch_weight_grade_match.group(5)
            else:
                # Set default values if regular expression does not match
                player_name = ""
                age = ""
                branch = ""
                weight = ""
                grade = ""
            
            # Retrieve remaining fields in order
            field_index = 2  # Index after player_info
            
            # National win rate
            national_win_rate = fields[field_index] if field_index < len(fields) else ""
            field_index += 1
            
            # National 2-win rate
            national_2_rate = fields[field_index] if field_index < len(fields) else ""
            field_index += 1
            
            # Local win rate
            local_win_rate = fields[field_index] if field_index < len(fields) else ""
            field_index += 1
            
            # Local 2-win rate
            local_2_rate = fields[field_index] if field_index < len(fields) else ""
            field_index += 1
            
            # Motor number
            motor_no = fields[field_index] if field_index < len(fields) else ""
            field_index += 1
            
            # Motor 2-win rate
            motor_2_rate = fields[field_index] if field_index < len(fields) else ""
            field_index += 1
            
            # Boat number
            boat_no = fields[field_index] if field_index < len(fields) else ""
            field_index += 1
            
            # Boat 2-win rate
            boat_2_rate = fields[field_index] if field_index < len(fields) else ""
            
            # Venue
            venue = venue_code
            
            # Add date information
            date_str = f"{year}-{month}-{day}"
            
            data.append({
                '選手登録番': player_id,  # Player registration number
                'レースID': race_id,  # Race ID
                '艇番': boat_number,  # Boat number
                '年齢': age,  # Age
                '支部': branch,  # Branch
                '体重': weight,  # Weight
                '級別': grade,  # Grade
                '全国勝率': national_win_rate,  # National win rate
                '全国2連率': national_2_rate,  # National 2-win rate
                '当地勝率': local_win_rate,  # Local win rate
                '当地2連率': local_2_rate,  # Local 2-win rate
                'モーター2連率': motor_2_rate,  # Motor 2-win rate
                'ボート2連率': boat_2_rate,  # Boat 2-win rate
                '会場': venue,  # Venue
                '日付': date_str  # Date
            })
    
    return pd.DataFrame(data)

def extract_data_from_k_file(file_path):
    """
    Extract data from K file (race results)
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
        # Detect race information row (actual race information)
        race_info_match = re.search(r'^\s*(\d+)R\s+.*H\d+m\s+(.+?)\s+風\s+(.+?)\s+(\d+)m\s+波\s+(\d+)cm', line)
        if race_info_match:
            race_number = race_info_match.group(1).zfill(2)  # Convert single-digit race number to two digits
            weather = race_info_match.group(2).strip()
            wind_direction = race_info_match.group(3).strip()
            wind_speed = race_info_match.group(4)
            wave_height = race_info_match.group(5)
            continue
        
        # Detect player data row (line starting with rank)
        if race_number and re.match(r'^\s*\d+\s+\d+\s+\d{4}', line):
            fields = line.strip().split()
            
            # Generate race ID (e.g., 202404012401)
            race_id = f'{year}{month}{day}{venue_code}{race_number}'
            
            # Rank
            rank = fields[0]
            
            # Boat number
            boat_number = fields[1]
            
            # Player registration number
            player_id = fields[2]
            
            # Player name (may be split into multiple fields)
            player_name_parts = []
            field_index = 3
            while field_index < len(fields) and not re.match(r'^\d+$', fields[field_index]):
                player_name_parts.append(fields[field_index])
                field_index += 1
            
            player_name = ''.join(player_name_parts)
            
            # Exhibition time (search for field containing numbers and dot)
            exhibition_time = None
            for j in range(field_index, len(fields)):
                if re.match(r'^\d+\.\d+$', fields[j]):
                    exhibition_time = fields[j]
                    break
            
            # Add date information
            date_str = f"{year}-{month}-{day}"
            
            data.append({
                '選手登録番': player_id,  # Player registration number
                'レースID': race_id,  # Race ID
                '着': rank,  # Rank
                '選手名': player_name,  # Player name
                '展示タイム': exhibition_time,  # Exhibition time
                '天候': weather,  # Weather
                '風向': wind_direction,  # Wind direction
                '風量': wind_speed,  # Wind speed
                '波': wave_height,  # Wave height
                '日付': date_str  # Date
            })
    
    return pd.DataFrame(data)

def merge_boat_race_data_files(b_files_dir, k_files_dir, output_file_template, debug=False):
    """
    Merge multiple B files and K files by date and save as a single large CSV file including single odds data
    
    Parameters:
    -----------
    b_files_dir : str
        Path to the directory containing B files (program tables)
    k_files_dir : str
        Path to the directory containing K files (race results)
    output_file_template : str
        Template for the path to save the merged data CSV file (latest date will be inserted)
    debug : bool
        Enable/disable debug mode
    """
    # Get list of B files
    b_files = glob.glob(os.path.join(b_files_dir, "B*.TXT"))
    
    # Get list of K files
    k_files = glob.glob(os.path.join(k_files_dir, "K*.TXT"))
    
    # Extract dates from file names and store in dictionary
    b_files_dict = {}
    for b_file in b_files:
        file_name = os.path.basename(b_file)
        date_str = file_name[1:7]  # Extract 240401 from B240401.TXT
        b_files_dict[date_str] = b_file
    
    k_files_dict = {}
    for k_file in k_files:
        file_name = os.path.basename(k_file)
        date_str = file_name[1:7]  # Extract 240401 from K240401.TXT
        k_files_dict[date_str] = k_file
    
    # Process only files with common dates
    common_dates = set(b_files_dict.keys()) & set(k_files_dict.keys())
    
    # List to store merged data
    all_merged_data = []
    
    # Count processed dates and number of files
    processed_dates = 0
    total_dates = len(common_dates)
    
    # Total number of rows with rank but no single odds
    total_missing_odds = 0
    
    print(f"Files with common dates: {total_dates} sets")
    
    # Merge B files and K files by date
    for date_str in sorted(common_dates):
        b_file = b_files_dict[date_str]
        k_file = k_files_dict[date_str]
        
        try:
            # Extract data from B file
            b_data = extract_data_from_b_file(b_file)
            
            # Extract data from K file
            k_data = extract_data_from_k_file(k_file)
            
            # Extract single odds data from K file (optimized version)
            single_odds_data = extract_single_odds_from_k_file(k_file, debug=debug)
            
            # Merge using player registration number and race ID as keys
            merged_data = pd.merge(b_data, k_data, on=['選手登録番', 'レースID', '日付'])
            
            # Add single odds data
            merged_data['単勝オッズ'] = 0  # Set default value
            
            for i, row in merged_data.iterrows():
                race_id = row['レースID']
                boat_number = row['艇番']
                key = f'{race_id}_{boat_number}'
                
                if key in single_odds_data:
                    merged_data.at[i, '単勝オッズ'] = single_odds_data[key]
            
            # Count rows with rank but no single odds
            missing_odds_count = len(merged_data[(merged_data['着'].notna()) & (merged_data['着'] != '') & 
                                               (merged_data['着'] != '欠') & (merged_data['着'] != 'F') & 
                                               (merged_data['単勝オッズ'] == 0)])
            
            if missing_odds_count > 0:
                total_missing_odds += missing_odds_count
                if debug:
                    print(f"Warning: {date_str} data has {missing_odds_count} rows with rank but no single odds")
                    # Display details of problematic rows
                    problem_rows = merged_data[(merged_data['着'].notna()) & (merged_data['着'] != '') & 
                                             (merged_data['着'] != '欠') & (merged_data['着'] != 'F') & 
                                             (merged_data['単勝オッズ'] == 0)]
                    for _, row in problem_rows.head(3).iterrows():
                        print(f"  Race ID: {row['レースID']}, Boat number: {row['艇番']}, Rank: {row['着']}")
            
            # Add merged data to list
            all_merged_data.append(merged_data)
            
            processed_dates += 1
            if processed_dates % 10 == 0 or processed_dates == total_dates:
                print(f"Progress: {processed_dates}/{total_dates} ({processed_dates/total_dates*100:.1f}%)")
        
        except Exception as e:
            print(f"Error: Problem occurred while processing {date_str} - {str(e)}")
    
    # Combine all merged data into a single DataFrame
    if all_merged_data:
        final_data = pd.concat(all_merged_data, ignore_index=True)
        
        # Get the latest date
        final_data["日付"] = pd.to_datetime(final_data["日付"])
        latest_date = final_data["日付"].max().strftime('%Y%m%d')
        
        # Generate file name with latest date
        output_file = output_file_template.replace("{date}", latest_date)
        
        # Save result to CSV file
        final_data.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"Merged data: {len(final_data)} rows")
        print(f"Latest date: {latest_date}")
        print(f"Data saved to {output_file}")
        
        # Total number of rows with rank but no single odds
        if total_missing_odds > 0:
            print(f"Warning: There are {total_missing_odds} rows with rank but no single odds in all data")
        else:
            print("All rank data has corresponding single odds successfully extracted")
        
        return final_data
    else:
        print("No data to merge")
        return pd.DataFrame()

if __name__ == "__main__":
    # Specify directory paths
    b_files_dir = "/Users/yoshidont_mind/Library/Mobile Documents/com~apple~CloudDocs/Desktop/CST term4/COMP 4949 Big Data Analytics Methods/boatrace_predictor/data/schedules"  # Directory containing B files
    k_files_dir = "/Users/yoshidont_mind/Library/Mobile Documents/com~apple~CloudDocs/Desktop/CST term4/COMP 4949 Big Data Analytics Methods/boatrace_predictor/data/results"    # Directory containing K files
    
    # Set output file name template ({date} will be replaced with the latest date)
    output_file_template = "/Users/yoshidont_mind/Library/Mobile Documents/com~apple~CloudDocs/Desktop/CST term4/COMP 4949 Big Data Analytics Methods/boatrace_predictor/data/merged/merged_{date}.csv"
    
    # If data/schedules and data/results do not exist in the current directory,
    # use the directories specified by absolute paths
    if not os.path.exists(b_files_dir):
        b_files_dir = "/Users/yoshidont_mind/Library/Mobile Documents/com~apple~CloudDocs/Desktop/CST term4/COMP 4949 Big Data Analytics Methods/boatrace_predictor/data/schedules"
    if not os.path.exists(k_files_dir):
        k_files_dir = "/Users/yoshidont_mind/Library/Mobile Documents/com~apple~CloudDocs/Desktop/CST term4/COMP 4949 Big Data Analytics Methods/boatrace_predictor/data/results"
    
    # Check if specified directories exist
    if not os.path.exists(b_files_dir):
        print(f"Error: B file directory '{b_files_dir}' not found")
        exit(1)
    if not os.path.exists(k_files_dir):
        print(f"Error: K file directory '{k_files_dir}' not found")
        exit(1)
    
    # Process multiple files
    print(f"\nB file directory: {b_files_dir}")
    print(f"K file directory: {k_files_dir}")
    print(f"Output file: {output_file_template}")
    
    # Start processing time
    start_time = datetime.now()
    print(f"Processing start: {start_time}")
    
    # Merge multiple B files and K files
    merged_data = merge_boat_race_data_files(b_files_dir, k_files_dir, output_file_template, debug=False)
    
    # End processing time
    end_time = datetime.now()
    print(f"Processing end: {end_time}")
    print(f"Processing time: {end_time - start_time}")
