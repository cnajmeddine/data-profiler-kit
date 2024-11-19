# DataProfilerKit

A Python library that provides quick and insightful data profiling for pandas DataFrames. It generates detailed reports including missing values analysis, data type information, correlations, outliers, and column statistics in a clear, organized format.

## Installation

```bash
pip install data-profiler-kit
```

## Usage

```python
from dataprofilerkit import DataProfiler
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

## Features

- Comprehensive DataFrame analysis
- Missing values detection and summary
- Column-wise statistics based on data types
- Duplicate rows and columns detection
- Outlier detection using Z-score and IQR methods
- Support for numeric, categorical, and datetime columns

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.