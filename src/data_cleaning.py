import pandas as pd
import numpy as np
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from the given filepath."""
    logging.info(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    logging.info(f"Data loaded successfully. Shape: {df.shape}")
    return df

def analyze_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Perform null value analysis."""
    logging.info("Performing null value analysis...")
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df)) * 100
    null_df = pd.DataFrame({'Null_Count': null_counts, 'Null_Percentage': null_pct})
    
    high_null_cols = null_df[null_df['Null_Percentage'] > 20].index.tolist()
    if high_null_cols:
        logging.warning(f"Columns exceeding 20% null rate: {high_null_cols}")
    
    return null_df

def fill_low_null_numeric_columns(df: pd.DataFrame, null_df: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric columns with < 20% missing values using their median."""
    df_cleaned = df.copy()
    low_null_cols = null_df[(null_df['Null_Percentage'] <= 20) & (null_df['Null_Percentage'] > 0)].index.tolist()
    
    for col in low_null_cols:
        if pd.api.types.is_numeric_dtype(df_cleaned[col]):
            median_val = df_cleaned[col].median()
            df_cleaned[col] = df_cleaned[col].fillna(median_val)
            logging.info(f"Filled nulls in '{col}' with median: {median_val:.2f}")
            
    return df_cleaned

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Identify and remove duplicate rows."""
    duplicates_count = df.duplicated().sum()
    logging.info(f"Number of duplicate rows found: {duplicates_count}")
    
    df_deduped = df.drop_duplicates().copy()
    logging.info(f"Rows removed: {len(df) - len(df_deduped)}. New shape: {df_deduped.shape}")
    
    return df_deduped

def correct_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Correct column data types specific to this dataset."""
    df_corrected = df.copy()
    
    # Correct 'Fare' if it's stored as object with '$'
    if not pd.api.types.is_numeric_dtype(df_corrected['Fare']):
        df_corrected['Fare'] = df_corrected['Fare'].astype(str).str.replace('$', '', regex=False)
        df_corrected['Fare'] = pd.to_numeric(df_corrected['Fare'], errors='coerce')
        logging.info("Corrected 'Fare' dtype to numeric.")
        
    # Convert repetitive string columns to category
    mem_before = df_corrected.memory_usage(deep=True).sum()
    df_corrected['Embarked'] = df_corrected['Embarked'].astype('category')
    mem_after = df_corrected.memory_usage(deep=True).sum()
    
    logging.info(f"Converted 'Embarked' to category. Memory reduced from {mem_before} to {mem_after} bytes.")
    
    return df_corrected
