#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script performs exploratory data analysis on boat race data.
It outputs the data types of each column, the number of missing values, and statistical information in tabular form.
It generates heatmaps and other graphs useful for analyzing the target variable (whether the rank is 1st or not) and saves them in the graphs directory.

Usage:
    python scripts/4_explore_data.py [input file path]

Example:
    python scripts/4_explore_data.py data/merged/boat_race_extracted.csv
    python scripts/4_explore_data.py data/merged/merged_20250406.csv
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import matplotlib as mpl
from sklearn.preprocessing import LabelEncoder


# Japanese font settings
mpl.rcParams['font.family'] = ['Hiragino Sans GB', 'AppleGothic', 'Yu Gothic', 'Meiryo', 'sans-serif']
# Set font size larger
plt.rcParams['font.size'] = 12
# Check the directory for saving graphs
GRAPHS_DIR = "graphs"
if not os.path.exists(GRAPHS_DIR):
    os.makedirs(GRAPHS_DIR)
    print(f"Created directory for saving graphs: {GRAPHS_DIR}")


def analyze_data(input_file):
    """
    Reads a CSV file and performs exploratory data analysis.
    Graphs are saved in the graphs directory.
    
    Args:
        input_file (str): Path to the CSV file to analyze
        
    Returns:
        bool: True if the process is successful, otherwise False
    """
    try:
        # Check if the file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' does not exist.")
            return False
            
        # Read the CSV file
        print(f"Reading file '{input_file}'...")
        df = pd.read_csv(input_file)
        
        # Display basic information
        print("\n===== Basic Data Information =====")
        print(f"Number of rows: {df.shape[0]}")
        print(f"Number of columns: {df.shape[1]}")
        print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
        
        # Display list of column names
        print("\n===== List of Column Names =====")
        for i, col in enumerate(df.columns, 1):
            print(f"{i:2d}. {col}")
        
        # Analyze data types
        print("\n===== Data Types of Each Column =====")
        dtype_df = pd.DataFrame({
            'Column Name': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Memory Usage (KB)': df.memory_usage(deep=True)[1:] / 1024
        })
        print(tabulate(dtype_df, headers='keys', tablefmt='grid', showindex=False))
        
        # Analyze missing values
        print("\n===== Analysis of Missing Values =====")
        null_df = pd.DataFrame({
            'Column Name': df.columns,
            'Number of Missing Values': df.isnull().sum(),
            'Percentage of Missing Values (%)': df.isnull().sum() / len(df) * 100
        }).sort_values('Number of Missing Values', ascending=False)
        print(tabulate(null_df, headers='keys', tablefmt='grid', showindex=False))
        
        # Statistical information of numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            print("\n===== Statistical Information of Numeric Columns =====")
            stats_df = pd.DataFrame({
                'Column Name': numeric_cols,
                'Min': [df[col].min() for col in numeric_cols],
                '1st Quartile': [df[col].quantile(0.25) for col in numeric_cols],
                'Median': [df[col].median() for col in numeric_cols],
                'Mean': [df[col].mean() for col in numeric_cols],
                '3rd Quartile': [df[col].quantile(0.75) for col in numeric_cols],
                'Max': [df[col].max() for col in numeric_cols],
                'Standard Deviation': [df[col].std() for col in numeric_cols],
                'Variance': [df[col].var() for col in numeric_cols]
            })
            print(tabulate(stats_df, headers='keys', tablefmt='grid', showindex=False, floatfmt='.3f'))
            
            # Analyze outliers
            print("\n===== Analysis of Outliers (Numeric Columns Only) =====")
            outliers_df = pd.DataFrame(columns=['Column Name', 'Number of Outliers', 'Percentage of Outliers (%)', 'Outlier Criteria'])
            
            for col in numeric_cols:
                if df[col].nunique() > 1:  # Exclude columns with only one unique value
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                    outlier_count = len(outliers)
                    outlier_percent = outlier_count / len(df) * 100
                    
                    outliers_df = pd.concat([outliers_df, pd.DataFrame({
                        'Column Name': [col],
                        'Number of Outliers': [outlier_count],
                        'Percentage of Outliers (%)': [outlier_percent],
                        'Outlier Criteria': [f"< {lower_bound:.3f} or > {upper_bound:.3f}"]
                    })], ignore_index=True)
            
            print(tabulate(outliers_df, headers='keys', tablefmt='grid', showindex=False))
        
        # Analyze categorical columns (string columns)
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        if cat_cols:
            print("\n===== Analysis of Categorical Columns =====")
            cat_df = pd.DataFrame({
                'Column Name': cat_cols,
                'Number of Unique Values': [df[col].nunique() for col in cat_cols],
                'Mode': [df[col].mode()[0] if not df[col].mode().empty else 'None' for col in cat_cols],
                'Frequency of Mode': [df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0 for col in cat_cols],
                'Percentage of Mode (%)': [df[col].value_counts().iloc[0] / len(df) * 100 if not df[col].value_counts().empty else 0 for col in cat_cols]
            })
            print(tabulate(cat_df, headers='keys', tablefmt='grid', showindex=False))
            
            # Display top values for each categorical column
            for col in cat_cols:
                if df[col].nunique() <= 20:  # Display only if unique values are 20 or less
                    print(f"\nValue distribution of column '{col}':")
                    value_counts = df[col].value_counts()
                    value_percent = value_counts / len(df) * 100
                    value_df = pd.DataFrame({
                        'Value': value_counts.index,
                        'Frequency': value_counts.values,
                        'Percentage (%)': value_percent.values
                    })
                    print(tabulate(value_df.head(10), headers='keys', tablefmt='grid', showindex=False))
                else:
                    print(f"\nTop 10 values of column '{col}':")
                    value_counts = df[col].value_counts()
                    value_percent = value_counts / len(df) * 100
                    value_df = pd.DataFrame({
                        'Value': value_counts.index,
                        'Frequency': value_counts.values,
                        'Percentage (%)': value_percent.values
                    })
                    print(tabulate(value_df.head(10), headers='keys', tablefmt='grid', showindex=False))
        
        # Analyze date columns
        try:
            if 'date' in df.columns:
                print("\n===== Analysis of Date Column =====")
                df['date'] = pd.to_datetime(df['date'])
                date_range = df['date'].max() - df['date'].min()
                print(f"Date range: {df['date'].min()} to {df['date'].max()} ({date_range.days} days)")
                print(f"Number of unique dates: {df['date'].nunique()}")
                
                # Data count by month
                monthly_counts = df['date'].dt.to_period('M').value_counts().sort_index()
                print("\nData count by month:")
                for period, count in monthly_counts.items():
                    print(f"{period}: {count} entries")
        except:
            print("Error occurred while analyzing the date column.")
        
        # ===== Add analysis of target variable (is_win) =====
        print("\n===== Analysis of Target Variable (Rank 1 or Not) =====")
        
        # Create is_win column indicating whether the rank is 1st or not (if not exists)
        if "is_win" not in df.columns and "着" in df.columns:  # "着" means "Rank"
            df["is_win"] = df["着"].apply(lambda x: 1 if x == 1 else 0)
            print("Created is_win column from rank")
        elif "is_win" not in df.columns:
            print("Warning: Skipping target variable analysis due to lack of rank data")
            return True
        
        # Check class distribution
        win_count = df["is_win"].sum()
        total_count = len(df)
        win_percent = win_count / total_count * 100
        
        print(f"Total number of races: {total_count}")
        print(f"Number of 1st places: {win_count} ({win_percent:.2f}%)")
        print(f"Number of 2nd places or lower: {total_count - win_count} ({100 - win_percent:.2f}%)")
        
        # Visualize class distribution (pie chart)
        plt.figure(figsize=(10, 6))
        win_counts = [win_count, total_count - win_count]
        labels = ["1st Place", "2nd Place or Lower"]
        plt.pie(win_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
        plt.title("Distribution of Ranks")
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f"{GRAPHS_DIR}/class_distribution.png", dpi=300, bbox_inches='tight')
        print(f"Saved class distribution graph: {GRAPHS_DIR}/class_distribution.png")
        plt.close()
        
        # Extract numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Analyze the relationship between is_win column and each numeric column
        useful_numeric_cols = [col for col in numeric_cols 
                               if col != "is_win" and col != "着"  # "着" means "Rank"
                               and df[col].nunique() > 5]  # Exclude columns with too few unique values
        
        if useful_numeric_cols:
            print("\nRelationship between main numeric features and rank:")
            
            # Correlation analysis
            corr_with_win = pd.DataFrame({
                'Feature': useful_numeric_cols,
                'Correlation with is_win': [df[col].corr(df["is_win"]) for col in useful_numeric_cols]
            }).sort_values('Correlation with is_win', ascending=False)
            
            print(tabulate(corr_with_win, headers='keys', tablefmt='grid', showindex=False, floatfmt='.4f'))
            
            # Display correlation coefficients as a heatmap (all features)
            plt.figure(figsize=(12, 10))
            
            # Create correlation dataframe (correlation between is_win and each feature)
            corr_data = {}
            for col in useful_numeric_cols:
                corr_data[col] = [df[col].corr(df["is_win"])]
            
            corr_df = pd.DataFrame(corr_data, index=['is_win']).T.sort_values('is_win', ascending=False)
            
            # Generate heatmap
            sns.heatmap(corr_df, annot=True, fmt=".4f", cmap="coolwarm", center=0,
                       vmin=-0.4, vmax=0.4, cbar_kws={"shrink": .8}, linewidths=0.5)
            
            plt.title("Correlation Coefficients between Features and Target Variable (is_win)")
            
            plt.tight_layout()
            plt.savefig(f"{GRAPHS_DIR}/target_correlation.png", dpi=300, bbox_inches='tight')
            print(f"Saved correlation coefficient graph: {GRAPHS_DIR}/target_correlation.png")
            plt.close()
            
            # Heatmap (overall correlation)
            plt.figure(figsize=(14, 12))
            # Extract columns with absolute correlation coefficient of 0.1 or higher
            significant_cols = [col for col in useful_numeric_cols 
                              if abs(df[col].corr(df["is_win"])) >= 0.1]
            
            if significant_cols:
                # Add is_win to the end
                significant_cols.append("is_win")
                corr_matrix = df[significant_cols].corr()
                
                # Create mask to display only upper triangle
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                
                # Generate heatmap
                sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", 
                           cmap="coolwarm", center=0, vmin=-1, vmax=1, 
                           linewidths=0.5, cbar_kws={"shrink": .8})
                
                plt.title("Correlation Coefficient Heatmap between Features")
                plt.tight_layout()
                plt.savefig(f"{GRAPHS_DIR}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
                print(f"Saved correlation heatmap: {GRAPHS_DIR}/correlation_heatmap.png")
            else:
                print("No columns with strong correlation with is_win found")
            plt.close()
            
            # Create box plots for the top 5 numeric features with high correlation
            top_corr_cols = corr_with_win.head(5)['Feature'].tolist()
            if top_corr_cols:
                plt.figure(figsize=(15, 10))
                for i, col in enumerate(top_corr_cols, 1):
                    plt.subplot(2, 3, i)
                    sns.boxplot(x='is_win', y=col, data=df, palette=['#ff9999','#66b3ff'])
                    plt.title(f"Relationship between {col} and 1st Place")
                    plt.xlabel("1st Place (1) vs Others (0)")
                    plt.ylabel(col)
                
                plt.tight_layout()
                plt.savefig(f"{GRAPHS_DIR}/boxplots_by_win.png", dpi=300, bbox_inches='tight')
                print(f"Saved box plots: {GRAPHS_DIR}/boxplots_by_win.png")
                plt.close()
                
                # Violin plots
                plt.figure(figsize=(15, 10))
                for i, col in enumerate(top_corr_cols, 1):
                    plt.subplot(2, 3, i)
                    sns.violinplot(x='is_win', y=col, data=df, palette=['#ff9999','#66b3ff'], inner='box')
                    plt.title(f"Distribution of {col} and 1st Place")
                    plt.xlabel("1st Place (1) vs Others (0)")
                    plt.ylabel(col)
                
                plt.tight_layout()
                plt.savefig(f"{GRAPHS_DIR}/violinplots_by_win.png", dpi=300, bbox_inches='tight')
                print(f"Saved violin plots: {GRAPHS_DIR}/violinplots_by_win.png")
                plt.close()
        
        # Analyze categorical columns (especially in relation to rank)
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        if cat_cols:
            print("\n===== Relationship between Categorical Columns and Rank =====")
            
            # Select up to 5 categorical columns and plot their relationship with the target variable
            selected_cats = cat_cols[:5]  # Use only the first 5
            
            for col in selected_cats:
                # Calculate win rate by category
                if df[col].nunique() <= 15:  # Only if the number of categories is not too large
                    win_rate_by_cat = df.groupby(col)["is_win"].mean().sort_values(ascending=False)
                    counts_by_cat = df.groupby(col).size()
                    
                    # Create dataframe to display
                    cat_df = pd.DataFrame({
                        'Category': win_rate_by_cat.index,
                        'Win Rate': win_rate_by_cat.values * 100,
                        'Sample Size': counts_by_cat.values
                    })
                    
                    print(f"\nWin rate of feature '{col}':")
                    print(tabulate(cat_df, headers='keys', tablefmt='grid', showindex=False, floatfmt='.2f'))
                    
                    # Plot
                    plt.figure(figsize=(12, 6))
                    ax = sns.barplot(x='Category', y='Win Rate', data=cat_df, color='skyblue')
                    
                    # Display win rate on top of bars
                    for i, p in enumerate(ax.patches):
                        ax.annotate(f'{p.get_height():.1f}%\n(n={counts_by_cat.values[i]})', 
                                  (p.get_x() + p.get_width() / 2., p.get_height()), 
                                  ha = 'center', va = 'bottom', 
                                  xytext = (0, 5), textcoords = 'offset points')
                    
                    plt.title(f"Win Rate by {col}")
                    plt.xlabel(col)
                    plt.ylabel("Win Rate (%)")
                    plt.xticks(rotation=45)
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    
                    # Save graph
                    graph_path = f"{GRAPHS_DIR}/win_rate_by_{col.replace(' ', '_')}.png"
                    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
                    print(f"Saved win rate graph: {graph_path}")
                    plt.close()
        
        # Relationship between target variable and boat number
        if "艇番" in df.columns:  # "艇番" means "Boat Number"
            plt.figure(figsize=(10, 6))
            win_by_lane = df.groupby("艇番")["is_win"].mean().sort_index() * 100
            counts_by_lane = df.groupby("艇番").size()
            
            # Bar plot
            ax = sns.barplot(x=win_by_lane.index, y=win_by_lane.values, palette="Blues_d")
            
            # Display values and sample size on top of bars
            for i, p in enumerate(ax.patches):
                lane_idx = win_by_lane.index[i]
                ax.annotate(f'{p.get_height():.1f}%\n(n={counts_by_lane[lane_idx]})', 
                          (p.get_x() + p.get_width() / 2., p.get_height()), 
                          ha = 'center', va = 'bottom', 
                          xytext = (0, 5), textcoords = 'offset points')
            
            plt.title("Win Rate by Boat Number")
            plt.xlabel("Boat Number")
            plt.ylabel("Win Rate (%)")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f"{GRAPHS_DIR}/win_rate_by_lane.png", dpi=300, bbox_inches='tight')
            print(f"Saved win rate by boat number graph: {GRAPHS_DIR}/win_rate_by_lane.png")
            plt.close()
        
        print("\nAnalysis complete. Graphs are saved in the graphs directory.")
        return True
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main function. Processes command line arguments and executes data analysis.
    """
    # Process command line arguments
    if len(sys.argv) < 2:
        print("Usage: python scripts/4_explore_data.py [input file path]")
        print("Example: python scripts/4_explore_data.py data/merged/boat_race_extracted.csv")
        return False
    
    input_file = sys.argv[1]
    
    # Execute data analysis
    success = analyze_data(input_file)
    return success


if __name__ == "__main__":
    main()
