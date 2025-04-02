import streamlit as st
import pandas as pd
import pickle
import time
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

# ====== レース一覧取得ボタンと更新ボタン ======
col1, col2 = st.columns([3, 1])

with col1:
    if st.button("📋 本日のレース一覧を取得"):
        with st.spinner("取得中..."):
            st.session_state.races_df = get_today_races()
            st.session_state.last_updated = datetime.now()

with col2:
    if st.button("↻ 一覧を更新"):
        with st.spinner("再取得..."):
            st.session_state.races_df = get_today_races()
            st.session_state.last_updated = datetime.now()

# ====== レース一覧表示 ======
if st.session_state.races_df is not None:
    races_df = st.session_state.races_df.copy()
    races_df = races_df.sort_values("締切予定時刻")

    st.write(f"### 🕰️ 最終更新: {st.session_state.last_updated.strftime('%Y/%m/%d %H:%M:%S')} 現在")
    st.dataframe(races_df, use_container_width=True)

    # ===== 各レースごとに「予測」ボタン表示 =====
    for i, row in races_df.iterrows():
        race_id = row["レースID"]
        st.write("---")
        st.markdown(f"#### ⛵ {row['場']} {row['R']}R / 締切: {row['締切予定時刻']}")
        if st.button(f"🤖 このレース({row['場']} {row['R']}R)を予測", key=f"predict_{race_id}"):
            with st.spinner("データ取得 → 前処理 → 予測..."):
                time.sleep(1.0)  # テスト用ウェイト
                result_df, predict_time = predict_single_race(race_id)
                if result_df is not None:
                    st.success(f"予測完了!（{predict_time.strftime('%H:%M:%S')} 時点 / 締切 {row['締切予定時刻']} の {int((pd.to_datetime(row['締切予定時刻']) - predict_time).total_seconds() / 60)}分前）")
                    st.dataframe(result_df, use_container_width=True)
                else:
                    st.error("予測に失敗しました...")
else:
    st.info("'本日のレース一覧を取得'ボタンを押してください")
