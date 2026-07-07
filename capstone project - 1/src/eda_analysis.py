import pandas as pd
import numpy as np
import logging

def get_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df[numeric_cols].describe()

def calculate_skewness(df: pd.DataFrame):
    """Calculate and log skewness for numeric columns, identifying the most skewed."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    skewness = df[numeric_cols].skew()
    highest_skew_col = skewness.abs().idxmax()
    logging.info(f"Skewness calculated. Highest absolute skewness: {highest_skew_col} ({skewness[highest_skew_col]:.4f})")
    return skewness, highest_skew_col

def detect_outliers_iqr(df: pd.DataFrame, columns: list) -> dict:
    """Detect outliers using the Interquartile Range (IQR) method."""
    outliers_info = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outliers_info[col] = {
            'bounds': (lower_bound, upper_bound),
            'count': len(outliers)
        }
        logging.info(f"Outliers in '{col}': {len(outliers)} (bounds: [{lower_bound:.2f}, {upper_bound:.2f}])")
    return outliers_info

def compute_correlations(df: pd.DataFrame):
    """Compute Pearson and Spearman correlation matrices."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    pearson_corr = df[numeric_cols].corr(method='pearson')
    spearman_corr = df[numeric_cols].corr(method='spearman')
    
    # Identify pair with highest absolute Pearson correlation
    corr_unstacked = pearson_corr.abs().unstack()
    corr_unstacked = corr_unstacked[corr_unstacked < 1.0] # Remove self correlation
    highest_corr_pair = corr_unstacked.idxmax()
    highest_corr_val = pearson_corr.loc[highest_corr_pair[0], highest_corr_pair[1]]
    logging.info(f"Highest absolute Pearson correlation: {highest_corr_pair} (r = {highest_corr_val:.4f})")
    
    return pearson_corr, spearman_corr

def get_grouped_aggregation(df: pd.DataFrame, cat_col: str, num_col: str) -> pd.DataFrame:
    """Compute grouped aggregations."""
    grouped = df.groupby(cat_col)[num_col].agg(['mean', 'std', 'count'])
    return grouped
