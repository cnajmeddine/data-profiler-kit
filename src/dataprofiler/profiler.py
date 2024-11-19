import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from .utils import detect_outliers_zscore, detect_outliers_iqr

class DataProfiler:
    """Main class for generating data profiles from pandas DataFrames."""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with a pandas DataFrame."""
        self.df = df.copy()
        self.profile: Dict[str, Any] = {}
    
    def generate_profile(self) -> Dict[str, Any]:
        """Generate a complete profile of the DataFrame."""
        self.profile = {
            'basic_info': self._get_basic_info(),
            'missing_values': self._analyze_missing_values(),
            'column_stats': self._analyze_columns(),
            'duplicates': self._analyze_duplicates(),
            'outliers': self._analyze_outliers()
        }
        return self.profile
    
    def _get_basic_info(self) -> Dict[str, Any]:
        """Get basic information about the DataFrame."""
        return {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'total_cells': self.df.size,
            'memory_usage': self.df.memory_usage(deep=True).sum(),
            'datatypes': self.df.dtypes.value_counts().to_dict()
        }
    
    def _analyze_missing_values(self) -> Dict[str, Any]:
        """Analyze missing values in the DataFrame."""
        missing = self.df.isnull().sum()
        return {
            'total_missing': missing.sum(),
            'missing_by_column': missing.to_dict(),
            'missing_percentages': (missing / len(self.df) * 100).to_dict()
        }
    
    def _analyze_columns(self) -> Dict[str, Dict[str, Any]]:
        """Analyze statistics for each column based on its data type."""
        column_stats = {}
        
        for column in self.df.columns:
            stats = {'dtype': str(self.df[column].dtype)}
            
            if pd.api.types.is_numeric_dtype(self.df[column]):
                stats.update(self._analyze_numeric_column(column))
            elif pd.api.types.is_datetime64_any_dtype(self.df[column]):
                stats.update(self._analyze_datetime_column(column))
            else:
                stats.update(self._analyze_categorical_column(column))
                
            column_stats[column] = stats
            
        return column_stats
    
    def _analyze_numeric_column(self, column: str) -> Dict[str, Any]:
        """Analyze a numeric column."""
        stats = self.df[column].describe().to_dict()
        stats['skewness'] = float(self.df[column].skew())
        stats['kurtosis'] = float(self.df[column].kurtosis())
        return stats
    
    def _analyze_categorical_column(self, column: str) -> Dict[str, Any]:
        """Analyze a categorical column."""
        value_counts = self.df[column].value_counts()
        return {
            'unique_values': len(value_counts),
            'top_values': value_counts.head(5).to_dict(),
            'value_percentages': (value_counts / len(self.df) * 100).head(5).to_dict()
        }
    
    def _analyze_datetime_column(self, column: str) -> Dict[str, Any]:
        """Analyze a datetime column."""
        return {
            'min': self.df[column].min(),
            'max': self.df[column].max(),
            'range_days': (self.df[column].max() - self.df[column].min()).days
        }
    
    def _analyze_duplicates(self) -> Dict[str, Any]:
        """Analyze duplicate rows and columns."""
        duplicate_rows = self.df.duplicated()
        duplicate_cols = self.df.T.duplicated()
        
        return {
            'duplicate_rows': {
                'count': int(duplicate_rows.sum()),
                'percentage': float(duplicate_rows.sum() / len(self.df) * 100)
            },
            'duplicate_columns': {
                'count': int(duplicate_cols.sum()),
                'columns': self.df.columns[duplicate_cols].tolist()
            }
        }
    
    def _analyze_outliers(self) -> Dict[str, Dict[str, Any]]:
        """Analyze outliers in numeric columns using both Z-score and IQR methods."""
        outliers = {}
        
        for column in self.df.select_dtypes(include=[np.number]).columns:
            # Clean the data by removing NA values for outlier detection
            clean_data = self.df[column].dropna()
            
            # Only perform outlier detection if we have enough data points
            if len(clean_data) >= 3:  # Need at least 3 points for meaningful detection
                zscore_outliers = detect_outliers_zscore(clean_data, threshold=2.0)  # Lowered threshold
                iqr_outliers = detect_outliers_iqr(clean_data)
                
                outliers[column] = {
                    'zscore': {
                        'count': int(zscore_outliers.sum()),
                        'percentage': float(zscore_outliers.sum() / len(clean_data) * 100),
                        'indices': zscore_outliers[zscore_outliers].index.tolist()
                    },
                    'iqr': {
                        'count': int(iqr_outliers.sum()),
                        'percentage': float(iqr_outliers.sum() / len(clean_data) * 100),
                        'indices': iqr_outliers[iqr_outliers].index.tolist()
                    }
                }
            else:
                outliers[column] = {
                    'zscore': {'count': 0, 'percentage': 0.0, 'indices': []},
                    'iqr': {'count': 0, 'percentage': 0.0, 'indices': []}
                }
        
        return outliers