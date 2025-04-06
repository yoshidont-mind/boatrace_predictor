import streamlit as st
import pandas as pd
import pickle
import time
import pytz
from datetime import datetime, timedelta
from pathlib import Path

from predict_race import predict_single_race
from retrieve_today_races import get_races_by_date, get_available_dates

st.set_page_config(page_title="Boat Race Prediction App", layout="wide")
st.title("🎉 Boat Race Prediction App")

# ====== State Management for Global Session ======
if "races_df" not in st.session_state:
    st.session_state.races_df = None
if "last_updated" not in st.session_state:
    st.session_state.last_updated = None
if "prediction_results" not in st.session_state:
    st.session_state.prediction_results = {}  # race_id -> [result_df, predict_time]
if "selected_date" not in st.session_state:
    st.session_state.selected_date = datetime.now().strftime("%Y-%m-%d")

# ====== Date Selection and List Retrieval ======
col1, col2 = st.columns([3, 1])

with col1:
    # Retrieve the list of available dates
    available_dates = get_available_dates()
    
    if not available_dates:
        st.warning("Race information files not found. Please check the data/race_info directory.")
    else:
        # Select today's date by default
        today = datetime.now().strftime("%Y-%m-%d")
        default_index = 0
        
        # If today's date is available, select it
        if today in available_dates:
            default_index = available_dates.index(today)
        
        # Date selection widget
        selected_date = st.selectbox(
            "Select a date:",
            available_dates,
            index=default_index
        )
        
        # Update the race list if the date changes
        if selected_date != st.session_state.selected_date:
            st.session_state.selected_date = selected_date
            with st.spinner(f"Retrieving race list for {selected_date}..."):
                try:
                    st.session_state.races_df = get_races_by_date(selected_date)
                    st.session_state.last_updated = datetime.now()
                except Exception as e:
                    st.error(f"Failed to retrieve race information: {e}")

# ====== Race List Display ======
if st.session_state.races_df is not None:
    races_df = st.session_state.races_df.copy()
    
    # Remove unnecessary columns
    if 'レースID' in races_df.columns:
        races_df = races_df.drop(columns=['レースID'])
    if 'ステータス' in races_df.columns:
        races_df = races_df.drop(columns=['ステータス'])
    
    # Process deadline time (add date if only time data)
    if '締切予定時刻' in races_df.columns:
        def format_time(time_str):
            if pd.isna(time_str) or time_str == "不明":
                return "unknown"
            # Check if the date is already included
            if ':' in time_str and len(time_str) <= 5:  # e.g. "15:30"
                today_str = datetime.now().strftime("%Y-%m-%d")
                return f"{today_str} {time_str}"
            return time_str
        
    # Sort the dataframe by deadline time
    races_df = races_df.sort_values("締切予定時刻")

    # Set JST timezone
    jst = pytz.timezone('Asia/Tokyo')
    now_jst = datetime.now(jst)
    
    # Reorder columns to match the requested order: "日付", "締切予定時刻", "レース場", "レース番号"
    if all(col in races_df.columns for col in ['日付', '締切予定時刻', 'レース場', 'レース番号']):
        column_order = ['日付', '締切予定時刻', 'レース場', 'レース番号']
        # Add any other columns that exist in the dataframe but not in our ordered list
        column_order.extend([col for col in races_df.columns if col not in column_order])
        # Reorder the dataframe columns
        races_df = races_df[column_order]
    
    # Convert レース番号 to string to ensure it displays left-aligned
    if 'レース番号' in races_df.columns:
        races_df['レース番号'] = races_df['レース番号'].astype(str)
    
    # Remove the last updated timestamp
    st.dataframe(races_df, use_container_width=True)

    # ===== Race Table Display =====
    st.write(f"### ⛵ Race List for {selected_date}")
    
    # Create a container to display the race list
    races_container = st.container()
    
    # Display each race in a card style
    for i, row in races_df.iterrows():
        race_id = st.session_state.races_df.loc[i, "レースID"]
        
        with races_container.expander(f"【{row['レース場']} {row['レース番号']}R】Deadline: {row['締切予定時刻']}"):
            # Create a container to display the prediction results for each race
            result_container = st.container()
            
            # Display a link to the race result page
            date_part = race_id[:8]
            venue_code = race_id[8:10]
            race_num = race_id[10:]
            result_url = f"https://www.boatrace.jp/owpc/pc/race/raceresult?rno={race_num}&jcd={venue_code}&hd={date_part}"
            st.markdown(f"[🏁 Check the results of this race]({result_url})")
            
            # Prediction button
            if st.button(f"🔮 Predict this race", key=f"predict_{race_id}"):
                with st.spinner("Getting last-minute information → Collecting odds → Predicting with model..."):
                    try:
                        result_df, predict_time = predict_single_race(race_id)
                        if result_df is not None:
                            # Remove the player name column
                            if '選手名' in result_df.columns:
                                result_df = result_df.drop(columns=['選手名'])
                            
                            # Sort by the rank column (first column)
                            if '順位' in result_df.columns:
                                result_df = result_df.sort_values('順位')
                            else:
                                # Consider the first column as the rank if there is no column name
                                result_df = result_df.sort_values(result_df.columns[0])
                            
                            # Save the prediction results in the session
                            st.session_state.prediction_results[race_id] = [result_df, predict_time]
                            
                            # Display only the prediction completion message (do not display the table)
                            st.success(f"Prediction completed!")
                        else:
                            st.error("Prediction failed. Insufficient data or an error occurred during data retrieval.")
                    except Exception as e:
                        st.error(f"Error occurred during prediction processing: {e}")
            
            # Display past prediction results if available
            if race_id in st.session_state.prediction_results:
                saved_result, saved_time = st.session_state.prediction_results[race_id]
                
                # Remove the player name column (if not already removed)
                if '選手名' in saved_result.columns:
                    saved_result = saved_result.drop(columns=['選手名'])
                
                # Sort by the rank column (first column)
                if '順位' in saved_result.columns:
                    saved_result = saved_result.sort_values('順位')
                else:
                    # Consider the first column as the rank if there is no column name
                    saved_result = saved_result.sort_values(saved_result.columns[0])
                
                st.write(f"#### Prediction Results")
                st.dataframe(saved_result, use_container_width=True)
                
                # Extract boats with an expected value greater than 1.0 (positive expected value boats)
                plus_ev_boats = saved_result[saved_result['期待値'] > 1.0]
                if not plus_ev_boats.empty:
                    st.write("#### 💰 Recommended Bets (Boats with an expected value greater than 1.0)")
                    for _, boat_row in plus_ev_boats.iterrows():
                        st.write(f"Boat Number **{int(boat_row['艇番'])}**: Expected Value **{boat_row['期待値']}**")
                else:
                    st.info("※ No boats with an expected value greater than 1.0")
else:
    st.info("Press the 'Get today's race list' button")
