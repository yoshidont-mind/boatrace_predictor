# Boat Race Predictor

A machine learning-based system to predict Japanese boat race (kyotei) results, analyze race data, and provide betting recommendations.

## Root Directory Files

- **app.py**: Streamlit web application for race prediction and visualization. Allows users to select dates, view race lists, and see prediction results with win probability, odds, and expected value for each boat.

- **predict_race.py**: Core module that handles race prediction. Scrapes real-time race information (boat performance, exhibition time, weather, etc.), processes the data, and applies the trained model to generate predictions.

- **retrieve_today_races.py**: Retrieves the list of today's races including venue, race number, and scheduled deadline time. Provides functions to get races for a specific date and list all available dates.

- **utils.py**: Utility functions used across the project, including data preprocessing, deviation score calculation, and column name translations between Japanese and English.

## Script Files in `scripts/` Directory

- **1_retrieve_past_data.py**: Downloads historical boat race data (schedules and results) from the official website for the past year, storing them as text files in the data directory.

- **2_merge_data.py**: Processes the downloaded text files (B files for race schedules and K files for race results) and merges them into CSV files organized by date. Extracts race outcomes, odds information, and boat performance data.

- **3_explore_data.py**: Performs exploratory data analysis on the merged data. Generates visualizations including correlation heatmaps, win rate by boat number, win rate by racer class, and relationship between various features and race outcomes.

- **4_preprocessing.py**: Preprocesses the merged data for machine learning, including numeric conversion, handling missing values, feature engineering, and calculating deviation scores for relative comparison within races.

- **5_lightGBM.py**: Implements the LightGBM model for race prediction. Includes model training using historical data with time-series split, evaluation, feature importance analysis, and various betting strategy simulations.

- **6_retrieve_race_info.py**: Scrapes current race information from the official website using the PyJPBoatrace library. Collects race schedules, racer information, and boat data for today's races.

- **7_observe_odds_change.py**: Tracks the changes in race odds leading up to a race, allowing analysis of market movements. Visualizes odds changes over time to identify betting patterns and market sentiment.

- **8_predict.py**: Command-line interface for race prediction. Takes date, venue, and race number as input, scrapes real-time data, applies the model, and outputs predictions with confidence levels and betting recommendations.

## Data Files

### `data/merged/` Directory

Contains merged data files combining race schedules and results, with filenames in the format `merged_YYYYMMDD.csv`. Each file includes:

- Race identification information (race ID, date, venue, boat number)
- Racer information (name, class, branch, age, weight)
- Performance metrics (national/local win rates, boat/motor performance)
- Race conditions (weather, wind, wave height)
- Race outcomes (finishing position) and odds

Example files:
- `merged_20250401.csv`, `merged_20250402.csv`, etc.

### `data/preprocessed/` Directory

Contains preprocessed data ready for model training, with filenames in the format `preprocessed_YYYYMMDD.csv`. These files include all features from merged data plus:

- Feature engineering results (month, day of week)
- Categorical encoding (venue, class, branch)
- Deviation scores for comparison within races
- Target variable (is_win) indicating whether the boat won

Example files:
- `preprocessed_20250401.csv`, `preprocessed_20250402.csv`, etc.

### `data/race_info/` Directory

Contains race schedule information scraped from the website, with filenames in the format `races_YYYYMMDDhhmm.csv`. Each file includes:

- Race schedules (date, venue, race number)
- Deadline times for each race
- Race status information
- Racer and boat details for each race
- Performance statistics for racers, boats, and motors

Example files:
- `races_202504011557.csv`, `races_202504020902.csv`, etc.

The timestamp in the filename (hhmm) represents when the data was scraped, allowing for tracking of information updates throughout the day.
