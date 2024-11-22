import pytest
import pandas as pd
import numpy as np
from dataprofilerkit import DataProfiler

@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'numeric': [2, 2, 3, 3, 4, 100],  # Made the outlier less extreme for Z-Score detection
        'categorical': ['A', 'B', 'A', 'C', 'B', 'D'],
        'datetime': pd.date_range('2023-01-01', periods=6),
        'missing_vals': [1, None, 3, None, 5, 6]
    })


def test_basic_info(sample_df):
    """Test basic information generation."""
    profiler = DataProfiler(sample_df)
    profile = profiler.generate_profile()
    basic_info_df = profile['basic_info']

    # Validate the structure and values of basic information
    assert basic_info_df.loc[basic_info_df['Metric'] == 'Number of Rows', 'Value'].iloc[0] == 6
    assert basic_info_df.loc[basic_info_df['Metric'] == 'Number of Columns', 'Value'].iloc[0] == 4
    assert isinstance(basic_info_df.loc[basic_info_df['Metric'] == 'Memory Usage', 'Value'].iloc[0], (int, np.integer))

def test_missing_values(sample_df):
    """Test missing values analysis."""
    profiler = DataProfiler(sample_df)
    profile = profiler.generate_profile()
    missing_values_df = profile['missing_values']

    # Validate missing value counts and percentages
    missing_vals_row = missing_values_df[missing_values_df['Column'] == 'missing_vals']
    assert missing_vals_row['Total Missing'].iloc[0] == 2
    assert missing_vals_row['Percentage Missing'].iloc[0] == pytest.approx(33.33, rel=1e-2)

def test_outlier_detection(sample_df):
    """Test outlier detection."""
    profiler = DataProfiler(sample_df)
    profile = profiler.generate_profile()
    outliers_df = profile['outliers']

    # Validate Z-Score outliers
    zscore_outliers = outliers_df[(outliers_df['Column'] == 'numeric') & (outliers_df['Method'] == 'Z-Score')]
    assert not zscore_outliers.empty, "No Z-Score outliers detected, check detection logic or thresholds."
    assert zscore_outliers['Count'].iloc[0] >= 1
    assert 5 in zscore_outliers['Indices'].iloc[0]  # Directly check if index 5 is in the list

    # Validate IQR outliers
    iqr_outliers = outliers_df[(outliers_df['Column'] == 'numeric') & (outliers_df['Method'] == 'IQR')]
    assert not iqr_outliers.empty, "No IQR outliers detected, check detection logic or thresholds."
    assert iqr_outliers['Count'].iloc[0] >= 1
    assert 5 in iqr_outliers['Indices'].iloc[0]  # Directly check if index 5 is in the list

def test_duplicates(sample_df):
    """Test duplicate detection."""
    profiler = DataProfiler(sample_df)
    profile = profiler.generate_profile()
    duplicates_df = profile['duplicates']

    # Validate duplicates analysis structure
    duplicate_rows = duplicates_df.loc[duplicates_df['Metric'] == 'Duplicate Rows', 'Value'].iloc[0]
    assert isinstance(duplicate_rows, (int, np.integer))
    assert duplicate_rows == 0  # No duplicate rows in this sample

    duplicate_columns = duplicates_df.loc[duplicates_df['Metric'] == 'Duplicate Columns', 'Value'].iloc[0]
    assert isinstance(duplicate_columns, (int, np.integer))
    assert duplicate_columns == 0  # No duplicate columns in this sample
