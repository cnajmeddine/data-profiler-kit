# DataProfilerKit

A Python library that provides quick and insightful data profiling for pandas DataFrames. It generates detailed reports including missing values analysis, data type information, correlations, outliers, and column statistics in a clear, organized format.

## Installation

```bash
pip install data-profiler-kit
```

## Usage

```python
from dataprofiler import DataProfiler
import pandas as pd

# Create or load your DataFrame
df = pd.read_csv('your_data.csv')

# Create a DataProfiler instance
profiler = DataProfiler(df)

# Generate the profile
profile = profiler.generate_profile()

# Access different aspects of the profile
print("Basic Information:")
print(profile['basic_info'])

print("\nMissing Values Analysis:")
print(profile['missing_values'])

print("\nColumn Statistics:")
print(profile['column_stats'])

print("\nDuplicates Analysis:")
print(profile['duplicates'])

print("\nOutliers Analysis:")
print(profile['outliers'])
```

## Core Functionality

- ### Basic DataFrame Information:
    - Number of rows, columns, and total cells.
    - Memory usage of the DataFrame.
    - Data types and their counts.

- ### Missing Value Analysis:
    - Total missing values across the DataFrame.
    - Missing values by column.
    - Percentage of missing values for each column.

- ### Column-wise Analysis:

    - #### Numeric Columns:
        - Descriptive statistics (mean, median, standard deviation, etc.).
        - Skewness and kurtosis.

    - #### Categorical Columns:
        - Count of unique values.
        - Top 5 most frequent values with their percentages.

    - #### Datetime Columns:
        - Minimum and maximum values.
        - Range in days.

- ### Duplicate Detection:
    - Duplicate rows (count and percentage).
    - Duplicate columns (count and list of column names).

- ### Outlier Detection:
    - For numeric columns, detects outliers using:
        - Z-score method (with indices and percentages).
        - Interquartile Range (IQR) method (with indices and percentages).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.