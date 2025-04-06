import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from pyunpack import Archive
import pytz

# 今日の日付をJST（日本標準時）で取得
jst = pytz.timezone('Asia/Tokyo')
today = datetime.now(jst)

# 1年前の日付を計算
start_date = today - timedelta(days=365)

# 保存先ディレクトリの設定
b_file_dir = "/Users/yoshidont_mind/Library/Mobile Documents/com~apple~CloudDocs/Desktop/CST term4/COMP 4949 Big Data Analytics Methods/boatrace_predictor/data/schedules"
k_file_dir = "/Users/yoshidont_mind/Library/Mobile Documents/com~apple~CloudDocs/Desktop/CST term4/COMP 4949 Big Data Analytics Methods/boatrace_predictor/data/results"

# ディレクトリが存在しない場合は作成
os.makedirs(b_file_dir, exist_ok=True)
os.makedirs(k_file_dir, exist_ok=True)

# ユーザーエージェントの設定
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

# 指定された日付範囲でループ
current_date = start_date
while current_date < today:
    # 日付のフォーマット設定
    yyyy = current_date.strftime('%Y')
    yy = current_date.strftime('%y')
    mm = current_date.strftime('%m')
    dd = current_date.strftime('%d')

    # BファイルとKファイルのURL設定
    b_file_url = f"http://www1.mbrace.or.jp/od2/B/{yyyy}{mm}/b{yy}{mm}{dd}.lzh"
    k_file_url = f"http://www1.mbrace.or.jp/od2/K/{yyyy}{mm}/k{yy}{mm}{dd}.lzh"

    # ダウンロードと解凍の関数
    def download_and_extract(url, save_dir, file_prefix):
        local_filename = os.path.join(save_dir, f"{file_prefix}{yy}{mm}{dd}.lzh")
        extracted_filename = os.path.join(save_dir, f"{file_prefix}{yy}{mm}{dd}.txt")
        # ファイルが既に存在する場合はスキップ
        if os.path.exists(extracted_filename):
            print(f"File {extracted_filename} already exists. Skipping download.")
            return
        try:
            # ファイルのダウンロード
            with requests.get(url, headers=headers, stream=True) as r:
                r.raise_for_status()
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"Downloaded {local_filename}")
            # ファイルの解凍
            Archive(local_filename).extractall(save_dir)
            print(f"Extracted {extracted_filename}")
            # ダウンロードしたLZHファイルの削除
            os.remove(local_filename)
            print(f"Deleted {local_filename}")
        except requests.HTTPError as e:
            print(f"HTTP Error for {url}: {e}")
        except Exception as e:
            print(f"Error processing {url}: {e}")

    # Bファイルのダウンロードと解凍
    download_and_extract(b_file_url, b_file_dir, 'b')

    # Kファイルのダウンロードと解凍
    download_and_extract(k_file_url, k_file_dir, 'k')

    # 次の日へ
    current_date += timedelta(days=1)
