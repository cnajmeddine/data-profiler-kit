import numpy as np
import pandas as pd

def detect_outliers_zscore(series: pd.Series, threshold: float = 2.0) -> pd.Series:
    """
    Detect outliers using the Z-score method.
    
    Args:
        series: Pandas Series containing numeric values
        threshold: Z-score threshold (default: 2.0)
    
    Returns:
        Boolean Series where True indicates an outlier
    """
    # Ensure we're working with numeric data
    if not pd.api.types.is_numeric_dtype(series):
        return pd.Series(False, index=series.index)
    
    # Remove NA values for calculation
    clean_series = series.dropna()
    
    # Handle empty series or series with all same values
    if len(clean_series) < 3 or clean_series.std() == 0:
        return pd.Series(False, index=series.index)
    
    mean = clean_series.mean()
    std = clean_series.std()
    
    # Calculate z-scores and create boolean mask
    z_scores = np.abs((clean_series - mean) / std)
    outlier_mask = pd.Series(False, index=series.index)
    outlier_mask[clean_series.index] = z_scores > threshold
    
    return outlier_mask

def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    Detect outliers using the Interquartile Range (IQR) method.
    
    Args:
        series: Pandas Series containing numeric values
        multiplier: IQR multiplier (default: 1.5)
    
    Returns:
        Boolean Series where True indicates an outlier
    """
    # Ensure we're working with numeric data
    if not pd.api.types.is_numeric_dtype(series):
        return pd.Series(False, index=series.index)
    
    # Remove NA values for calculation
    clean_series = series.dropna()
    
    # Handle empty series or insufficient data
    if len(clean_series) < 3:
        return pd.Series(False, index=series.index)
    
    q1 = clean_series.quantile(0.25)
    q3 = clean_series.quantile(0.75)
    iqr = q3 - q1
    
    # Handle zero IQR case
    if iqr == 0:
        return pd.Series(False, index=series.index)
    
    lower_bound = q1 - (multiplier * iqr)
    upper_bound = q3 + (multiplier * iqr)
    
    # Create outlier mask
    outlier_mask = pd.Series(False, index=series.index)
    outlier_mask[clean_series.index] = (clean_series < lower_bound) | (clean_series > upper_bound)
    
    return outlier_mask