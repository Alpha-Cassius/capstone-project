import logging
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

from data_cleaning import (
    load_data, 
    analyze_nulls, 
    fill_low_null_numeric_columns, 
    remove_duplicates, 
    correct_dtypes
)
from eda_analysis import (
    get_descriptive_stats, 
    calculate_skewness, 
    detect_outliers_iqr, 
    compute_correlations, 
    get_grouped_aggregation
)
from visualizations import (
    plot_line_chart, 
    plot_bar_chart, 
    plot_histogram, 
    plot_scatter, 
    plot_boxplot, 
    plot_correlation_heatmap
)

def main():
    logging.info("Starting Capstone Project Pipeline...")
    
    # 1. Load Data
    raw_data_path = 'data/raw/raw_client_data.csv'
    df = load_data(raw_data_path)
    df_original = df.copy() # Store original for Pre-Imputation Task 9a
    
    # 2. Data Cleaning: Nulls
    null_df = analyze_nulls(df)
    df = fill_low_null_numeric_columns(df, null_df)
    
    # 3. Data Cleaning: Duplicates
    df = remove_duplicates(df)
    
    # 4. Data Cleaning: Types
    df = correct_dtypes(df)
    
    # 5. EDA: Descriptive Stats & Skewness
    desc_stats = get_descriptive_stats(df)
    skewness, highest_skew = calculate_skewness(df)
    
    # 6. EDA: Outlier Detection
    outliers_info = detect_outliers_iqr(df, ['Age', 'Fare'])
    
    # 7. Visualizations
    logging.info("Generating visualizations...")
    plot_line_chart(df, 'Age')
    plot_bar_chart(df, 'Pclass', 'Fare')
    plot_histogram(df, highest_skew)
    plot_scatter(df, 'Age', 'Fare')
    plot_boxplot(df, 'Survived', 'Age')
    
    # 8. Correlations
    pearson_corr, spearman_corr = compute_correlations(df)
    plot_correlation_heatmap(pearson_corr)
    
    # 9a. Imputation Strategy Comparison
    logging.info("Executing Task 9a: Pre-imputation comparison...")
    top_two_skewed = skewness.abs().nlargest(2).index.tolist()
    
    df_orig_calc = df_original.copy()
    if not pd.api.types.is_numeric_dtype(df_orig_calc['Fare']):
        df_orig_calc['Fare'] = df_orig_calc['Fare'].astype(str).str.replace('$', '', regex=False)
        df_orig_calc['Fare'] = pd.to_numeric(df_orig_calc['Fare'], errors='coerce')

    for col in top_two_skewed:
        orig_mean = df_orig_calc[col].mean()
        orig_median = df_orig_calc[col].median()
        logging.info(f"{col} BEFORE imputation: Mean = {orig_mean:.4f}, Median = {orig_median:.4f}")
        df[col] = df[col].fillna(orig_median)
        
    remaining_nulls = df[top_two_skewed].isnull().sum().sum()
    logging.info(f"Nulls remaining in top two skewed columns after median imputation: {remaining_nulls}")
    
    # 9b. Spearman Rank Difference
    logging.info("Executing Task 9b: Spearman vs Pearson difference...")
    diff_matrix = (spearman_corr - pearson_corr).abs()
    diff_unstacked = diff_matrix.unstack()
    diff_unstacked = diff_unstacked[diff_unstacked.index.get_level_values(0) < diff_unstacked.index.get_level_values(1)]
    top_3_diffs = diff_unstacked.nlargest(3)
    logging.info(f"Top 3 Correlation Differences:\n{top_3_diffs}")

    # 9c. Grouped Aggregation
    grouped = get_grouped_aggregation(df, 'Pclass', 'Fare')
    logging.info(f"Grouped Aggregation of Fare by Pclass:\n{grouped}")
    
    # Finalize
    processed_data_path = 'data/processed/cleaned_data.csv'
    df.to_csv(processed_data_path, index=False)
    logging.info(f"Pipeline complete! Cleaned dataset saved to {processed_data_path}")

if __name__ == "__main__":
    main()
