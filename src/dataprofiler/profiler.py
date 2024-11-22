import pandas as pd
import numpy as np
from typing import Dict, Any
from .utils import detect_outliers_zscore, detect_outliers_iqr

class DataProfiler:
    """Main class for generating data profiles from pandas DataFrames."""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with a pandas DataFrame."""
        self.df = df.copy()
    
    def generate_profile(self) -> Dict[str, pd.DataFrame]:
        """Generate a complete profile of the DataFrame."""
        return {
            'basic_info': self._get_basic_info(),
            'missing_values': self._analyze_missing_values(),
            'column_stats': self._analyze_columns(),
            'duplicates': self._analyze_duplicates(),
            'outliers': self._analyze_outliers()
        }
    
    def _get_basic_info(self) -> pd.DataFrame:
        """Get basic information about the DataFrame."""
        memory_usage_bytes = self.df.memory_usage(deep=True).sum()
        memory_usage_mb = memory_usage_bytes / (1024 ** 2)  # Convert bytes to MB

        # Create structured DataFrame
        data = {
            'Metric': ['Rows', 'Columns', 'Total Cells', 'Memory Usage (MB)', 'Datatypes'],
            'Value': [
                len(self.df),
                len(self.df.columns),
                self.df.size,
                round(memory_usage_mb, 2),
                str(self.df.dtypes.value_counts().to_dict())
            ]
        }
        return pd.DataFrame(data)
    
    def _analyze_missing_values(self) -> pd.DataFrame:
        """Analyze missing values in the DataFrame."""
        missing = self.df.isnull().sum()
        percentage = (missing / len(self.df)) * 100

        # Return as a DataFrame
        return pd.DataFrame({
            'Column': self.df.columns,
            'Missing Count': missing.values,
            'Missing Percentage': percentage.round(2).values
        })
    
    def _analyze_columns(self) -> pd.DataFrame:
        """Analyze statistics for each column based on its data type."""
        stats = []
        
        for column in self.df.columns:
            dtype = str(self.df[column].dtype)
            if pd.api.types.is_numeric_dtype(self.df[column]):
                col_stats = self._analyze_numeric_column(column)
            elif pd.api.types.is_datetime64_any_dtype(self.df[column]):
                col_stats = self._analyze_datetime_column(column)
            else:
                col_stats = self._analyze_categorical_column(column)
            
            col_stats['Column'] = column
            col_stats['Data Type'] = dtype
            stats.append(col_stats)
        
        return pd.DataFrame(stats)
    
    def _analyze_numeric_column(self, column: str) -> Dict[str, Any]:
        """Analyze a numeric column."""
        stats = self.df[column].describe().to_dict()
        stats['Skewness'] = float(self.df[column].skew())
        stats['Kurtosis'] = float(self.df[column].kurtosis())
        return stats
    
    def _analyze_categorical_column(self, column: str) -> Dict[str, Any]:
        """Analyze a categorical column."""
        value_counts = self.df[column].value_counts()
        return {
            'Unique Values': len(value_counts),
            'Top Values': str(value_counts.head(5).to_dict()),
            'Top Value Percentages': str((value_counts / len(self.df) * 100).head(5).to_dict())
        }
    
    def _analyze_datetime_column(self, column: str) -> Dict[str, Any]:
        """Analyze a datetime column."""
        return {
            'Min': self.df[column].min(),
            'Max': self.df[column].max(),
            'Range (Days)': (self.df[column].max() - self.df[column].min()).days
        }
    
    def _analyze_duplicates(self) -> pd.DataFrame:
        """Analyze duplicate rows and columns."""
        duplicate_rows = self.df.duplicated().sum()
        duplicate_cols = self.df.T.duplicated().sum()

        # Return as DataFrame
        data = {
            'Metric': ['Duplicate Rows', 'Duplicate Columns'],
            'Count': [duplicate_rows, duplicate_cols],
            'Percentage': [
                round((duplicate_rows / len(self.df)) * 100, 2),
                round((duplicate_cols / len(self.df.columns)) * 100, 2)
            ]
        }
        return pd.DataFrame(data)
    
    def _analyze_outliers(self) -> pd.DataFrame:
        """Analyze outliers in numeric columns using both Z-score and IQR methods."""
        results = []
        
        for column in self.df.select_dtypes(include=[np.number]).columns:
            clean_data = self.df[column].dropna()
            
            if len(clean_data) >= 3:
                zscore_outliers = detect_outliers_zscore(clean_data)
                iqr_outliers = detect_outliers_iqr(clean_data)
                
                results.append({
                    'Column': column,
                    'Z-Score Outliers (Count)': zscore_outliers.sum(),
                    'Z-Score Outliers (Indices)': str(zscore_outliers[zscore_outliers].index.tolist()),
                    'IQR Outliers (Count)': iqr_outliers.sum(),
                    'IQR Outliers (Indices)': str(iqr_outliers[iqr_outliers].index.tolist())
                })
        
        return pd.DataFrame(results)
