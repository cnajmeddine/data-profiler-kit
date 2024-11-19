import pytest
import pandas as pd
import numpy as np
from dataprofilerkit import DataProfiler

@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'numeric': [2, 2, 3, 3, 4, 1000],  # Made the outlier more extreme and added more consistent values
        'categorical': ['A', 'B', 'A', 'C', 'B', 'D'],
        'datetime': pd.date_range('2023-01-01', periods=6),
        'missing_vals': [1, None, 3, None, 5, 6]
    })

def test_basic_info(sample_df):
    """Test basic information generation."""
    profiler = DataProfiler(sample_df)
    profile = profiler.generate_profile()
    
    assert profile['basic_info']['rows'] == 6
    assert profile['basic_info']['columns'] == 4

def test_missing_values(sample_df):
    """Test missing values analysis."""
    profiler = DataProfiler(sample_df)
    profile = profiler.generate_profile()
    
    assert profile['missing_values']['total_missing'] == 2
    assert profile['missing_values']['missing_by_column']['missing_vals'] == 2

def test_outlier_detection(sample_df):
    """Test outlier detection."""
    profiler = DataProfiler(sample_df)
    profile = profiler.generate_profile()
    
    # Print debug information
    print("\nOutlier detection results:")
    print(f"Z-score outliers: {profile['outliers']['numeric']['zscore']}")
    print(f"IQR outliers: {profile['outliers']['numeric']['iqr']}")
    
    # Test for outliers
    assert profile['outliers']['numeric']['zscore']['count'] >= 1, "No z-score outliers detected"
    assert profile['outliers']['numeric']['iqr']['count'] >= 1, "No IQR outliers detected"
    
    # Verify the outlier value is included in the indices
    assert 5 in profile['outliers']['numeric']['zscore']['indices'], "Expected outlier index not found"

def test_duplicates(sample_df):
    """Test duplicate detection."""
    profiler = DataProfiler(sample_df)
    profile = profiler.generate_profile()
    
    assert isinstance(profile['duplicates']['duplicate_rows']['count'], (int, np.integer))
    assert isinstance(profile['duplicates']['duplicate_columns']['count'], (int, np.integer))