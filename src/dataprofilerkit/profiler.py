import pandas as pd
import numpy as np
from typing import Dict, Any
from IPython.display import display

# Set pandas display options globally
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

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

        # Create structured DataFrame matching test expectations
        data = {
            'Metric': [
                'Number of Rows',
                'Number of Columns',
                'Memory Usage',
                'Datatypes'
            ],
            'Value': [
                len(self.df),
                len(self.df.columns),
                int(memory_usage_mb),  # Convert to integer as expected by test
                str(self.df.dtypes.value_counts().to_dict())
            ]
        }
        result_df = pd.DataFrame(data)

        # Align the 'Value' column to the left for better readability
        result_df['Value'] = result_df['Value'].astype(str)

        # Use display to output the DataFrame
        display(result_df)

        return pd.DataFrame(data)
    
    def _analyze_missing_values(self) -> pd.DataFrame:
        """Analyze missing values in the DataFrame."""
        missing = self.df.isnull().sum()
        percentage = (missing / len(self.df)) * 100

        # Return as a DataFrame with columns matching test expectations
        return pd.DataFrame({
            'Column': missing.index,
            'Total Missing': missing.values,
            'Percentage Missing': percentage.round(2).values
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

        # Return as DataFrame with structure matching test expectations
        data = {
            'Metric': ['Duplicate Rows', 'Duplicate Columns'],
            'Value': [duplicate_rows, duplicate_cols],  # Changed from 'Count' to 'Value'
            'Percentage': [
                round((duplicate_rows / len(self.df)) * 100, 2),
                round((duplicate_cols / len(self.df.columns)) * 100, 2)
            ]
        }
        return pd.DataFrame(data)
    
    def _analyze_outliers(self) -> pd.DataFrame:
        """
        Analyze outliers in the DataFrame using Z-Score and IQR methods.

        Returns:
            pd.DataFrame: A DataFrame containing outlier analysis for each numeric column.
        """
        outlier_data = []
        zscore_threshold = 2  # Threshold for Z-Score
        iqr_multiplier = 1.5  # Multiplier for IQR

        for column in self.df.select_dtypes(include=['number']).columns:
            column_data = self.df[column].dropna()

            # Z-Score Method
            mean = column_data.mean()
            std_dev = column_data.std()
            zscore_outliers = column_data[abs((column_data - mean) / std_dev) > zscore_threshold].index.tolist()
            zscore_count = len(zscore_outliers)

            # IQR Method
            q1 = column_data.quantile(0.25)
            q3 = column_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr
            iqr_outliers = column_data[(column_data < lower_bound) | (column_data > upper_bound)].index.tolist()
            iqr_count = len(iqr_outliers)

            # Append results to outlier_data
            if zscore_count > 0:
                outlier_data.append({
                    'Column': column,
                    'Method': 'Z-Score',
                    'Count': zscore_count,
                    'Percentage': round((zscore_count / len(self.df)) * 100, 2),
                    'Indices': zscore_outliers
                })

            if iqr_count > 0:
                outlier_data.append({
                    'Column': column,
                    'Method': 'IQR',
                    'Count': iqr_count,
                    'Percentage': round((iqr_count / len(self.df)) * 100, 2),
                    'Indices': iqr_outliers
                })

        return pd.DataFrame(outlier_data, columns=['Column', 'Method', 'Count', 'Percentage', 'Indices'])