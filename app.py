import streamlit as st
import pandas as pd
import pickle
import time
import pytz
from datetime import datetime, timedelta
from pathlib import Path

from predict_race import predict_single_race
from retrieve_today_races import get_today_races

st.set_page_config(page_title="競艇予測アプリ", layout="wide")
st.title("🎉 競艇予測アプリ")

# ====== サイドバー設定 ======
st.sidebar.write("### メニュー")
st.sidebar.markdown("- [Streamlit公式](https://streamlit.io)")
st.sidebar.markdown("- 現在時刻: " + datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

# ====== グローバルセッション用のステート管理 ======
if "races_df" not in st.session_state:
    st.session_state.races_df = None
if "last_updated" not in st.session_state:
    st.session_state.last_updated = None
if "prediction_results" not in st.session_state:
    st.session_state.prediction_results = {}  # race_id -> [result_df, predict_time]

# ====== レース一覧取得ボタンと更新ボタン ======
col1, col2 = st.columns([3, 1])

with col1:
    if st.button("📋 本日のレース一覧を取得"):
        with st.spinner("取得中..."):
            try:
                st.session_state.races_df = get_today_races()
                st.session_state.last_updated = datetime.now()
            except Exception as e:
                st.error(f"レース情報の取得に失敗しました: {e}")

with col2:
    if st.button("↻ 一覧を更新"):
        with st.spinner("再取得中..."):
            try:
                st.session_state.races_df = get_today_races()
                st.session_state.last_updated = datetime.now()
            except Exception as e:
                st.error(f"レース情報の更新に失敗しました: {e}")

# ====== レース一覧表示 ======
if st.session_state.races_df is not None:
    races_df = st.session_state.races_df.copy()
    
    # 締切予定時刻の処理（時刻データのみなら日付を追加）
    if '締切予定時刻' in races_df.columns:
        def format_time(time_str):
            if pd.isna(time_str) or time_str == "不明":
                return "不明"
            # すでに日付が含まれているか確認
            if ':' in time_str and len(time_str) <= 5:  # e.g. "15:30"
                today_str = datetime.now().strftime("%Y-%m-%d")
                return f"{today_str} {time_str}"
            return time_str
        
        races_df['締切予定時刻'] = races_df['締切予定時刻'].apply(format_time)
        
    # データフレームを締切時刻でソート
    races_df = races_df.sort_values("締切予定時刻")

    # JSTタイムゾーンの設定
    jst = pytz.timezone('Asia/Tokyo')
    now_jst = datetime.now(jst)
    
    st.write(f"### 🕰️ 最終更新: {st.session_state.last_updated.strftime('%Y/%m/%d %H:%M:%S')} 現在")
    st.dataframe(races_df, use_container_width=True)

    # ===== レーステーブル表示 =====
    st.write("### ⛵ 今日のレース一覧")
    
    # レース一覧を表示するコンテナを作成
    races_container = st.container()
    
    # 各レースごとにカードスタイルで表示
    for i, row in races_df.iterrows():
        race_id = row["レースID"]
        
        with races_container.expander(f"【{row['場']} {row['R']}R】締切: {row['締切予定時刻']} / ステータス: {row['ステータス']}"):
            # レースごとの予測結果を表示するコンテナ
            result_container = st.container()
            
            # 予測ボタン
            if st.button(f"🔮 このレースを予測する", key=f"predict_{race_id}"):
                with st.spinner("直前情報の取得 → オッズ収集 → モデル予測中..."):
                    try:
                        result_df, predict_time = predict_single_race(race_id)
                        if result_df is not None:
                            # 予測結果をセッションに保存
                            st.session_state.prediction_results[race_id] = [result_df, predict_time]
                            
                            # 締切までの残り時間計算
                            try:
                                # 締切予定時刻が時:分 形式の場合は当日の日付を追加
                                deadline_str = row['締切予定時刻']
                                if ':' in deadline_str and len(deadline_str) <= 5:  # e.g. "15:17"
                                    today_str = predict_time.strftime("%Y-%m-%d")
                                    deadline_str = f"{today_str} {deadline_str}"
                                
                                deadline_time = pd.to_datetime(deadline_str)
                                # もし締切時刻のタイムゾーン情報がなければ、予測時刻と同じタイムゾーンを設定
                                if predict_time.tzinfo and not deadline_time.tzinfo:
                                    deadline_time = pytz.timezone('Asia/Tokyo').localize(deadline_time)
                                
                                time_to_deadline = int((deadline_time - predict_time).total_seconds() / 60)
                                time_info = f"締切{time_to_deadline}分前"
                            except Exception as e:
                                print(f"締切時間計算エラー: {e}")
                                time_info = "時刻不明"
                            
                            # 結果を表示
                            st.success(f"予測完了!（{predict_time.strftime('%H:%M:%S')} 時点 / {time_info}）")
                            st.dataframe(result_df, use_container_width=True)
                            
                            # 期待値が1.0を超える艇（プラス期待値の艇）を抽出
                            plus_ev_boats = result_df[result_df['期待値'] > 1.0]
                            if not plus_ev_boats.empty:
                                st.write("#### 💰 おすすめ買い目（期待値が1.0を超える艇）")
                                for _, boat_row in plus_ev_boats.iterrows():
                                    st.write(f"艇番 **{int(boat_row['艇番'])}**: 期待値 **{boat_row['期待値']}**")
                            else:
                                st.info("※ 期待値が1.0を超える艇はありません")
                        else:
                            st.error("予測に失敗しました。締切済みのレースか、データ不足の可能性があります。")
                    except Exception as e:
                        st.error(f"予測処理中にエラーが発生しました: {e}")
            
            # 過去の予測結果があれば表示
            if race_id in st.session_state.prediction_results:
                saved_result, saved_time = st.session_state.prediction_results[race_id]
                
                # 締切までの残り時間計算
                try:
                    # 締切予定時刻が時:分 形式の場合は当日の日付を追加
                    deadline_str = row['締切予定時刻']
                    if ':' in deadline_str and len(deadline_str) <= 5:  # e.g. "15:17"
                        today_str = saved_time.strftime("%Y-%m-%d")
                        deadline_str = f"{today_str} {deadline_str}"
                    
                    deadline_time = pd.to_datetime(deadline_str)
                    # もし締切時刻のタイムゾーン情報がなければ、保存時刻と同じタイムゾーンを設定
                    if saved_time.tzinfo and not deadline_time.tzinfo:
                        deadline_time = pytz.timezone('Asia/Tokyo').localize(deadline_time)
                    
                    time_to_deadline = int((deadline_time - saved_time).total_seconds() / 60)
                    time_info = f"締切{time_to_deadline}分前"
                except Exception as e:
                    print(f"締切時間計算エラー: {e}")
                    time_info = "時刻不明"
                
                st.write(f"#### 最新予測結果（{saved_time.strftime('%H:%M:%S')} 時点 / {time_info}）")
                st.dataframe(saved_result, use_container_width=True)
                
                # 期待値が1.0を超える艇（プラス期待値の艇）を抽出
                plus_ev_boats = saved_result[saved_result['期待値'] > 1.0]
                if not plus_ev_boats.empty:
                    st.write("#### 💰 おすすめ買い目（期待値が1.0を超える艇）")
                    for _, boat_row in plus_ev_boats.iterrows():
                        st.write(f"艇番 **{int(boat_row['艇番'])}**: 期待値 **{boat_row['期待値']}**")
                else:
                    st.info("※ 期待値が1.0を超える艇はありません")
else:
    st.info("'本日のレース一覧を取得'ボタンを押してください")
