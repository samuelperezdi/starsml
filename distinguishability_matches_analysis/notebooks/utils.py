"""
utils.py

Author: Joshua D. Ingram

This file contains utility functions for the distinguishability of first and second best
matches analysis in the StarsML project.

Functions:
- create_bins_min_count: Find bin bounds with a minimum count of data in each range.
- ks_test_filter_ranges: Perform two-sample KS test on specified variable with filters.
"""

import pandas as pd
import numpy as np
import scipy.stats as stats

# Function to create bins with at least `min_count` values in each
def create_bins_min_count(dataframe, column, min_count=1000):
    """
    Find bin bounds with a minimum count of data in each range.

    Parameters
    ---------
    df : pd.DataFrame
        Dataframe of data.

    column : str
        Column name for column to create bins for.

    min_count : int
        Minimum count for each bin.

    Returns
    -------
    list
        List of bin bounds.
    """

    sorted_values = dataframe[column].sort_values().values
    bins = []
    current_count = 0

    # Iterate through sorted values and create bins
    for value in sorted_values:
        current_count += 1
        if current_count >= min_count:
            bins.append(value)
            current_count = 0

    # Ensure the last bin captures the max value
    if bins[-1] != sorted_values[-1]:
        bins.append(sorted_values[-1])

    # Adding the minimum value (0) to the start of the bins
    bins.insert(0, 0)

    return bins



def ks_test_filter_ranges(data1, data2, variable, filter_column1, filter_column2, 
                          filter1, filter2, filter3, filter4,
                          unique_id, filter_count = "first"):
    """
    This function takes in two dataframes and performs a KS test on the variable
    specified. The KS test is performed on the variable for the dataframes
    filtered by the specified filters. The function returns a dataframe with the
    KS test results for each filter.
    """

    """
    Perform two-sample KS test on specified variable with filters.

    This function takes in two dataframes and performs a KS test on the variable
    specified. The KS test is performed on the variable for the dataframes
    filtered by the specified filters. The function returns a dataframe with the
    KS test results for each filter.

    Parameters
    ---------
    data1 : pd.DataFrame
        Dataframe of first dataset.

    data1 : pd.DataFrame
        Dataframe of second dataset.

    variable : string
        Variable of interest for filtering and analysis.

    filter_column1 : string
        Column one for first filter ranges.

    filter_column2 : string
        Column two for second filter ranges.

    filter1 : float
        Lower range for filter on filter_column1

    filter2 : float
        Upper range for filter on filter_column1

    filter3 : float
        Lower range for filter on filter_column2

    filter4 : float
        Upper range for filter on filter_column2

    filter_count : string
        Specifies which dataset to filter. Options are "first", "second", or "both".

    unique_id : string
        Unique identifier for each row in the dataframes.

    Returns
    -------
    list
        List of KS test results: [KS statistic, p-value, len(data1), len(data2)]

    """

    if filter_count == "first":
        data1 = data1[(data1[filter_column1] > filter1) & (data1[filter_column1] <= filter2) & (data1[filter_column2] > filter3) & (data1[filter_column2] <= filter4)]
        data2 = data2[data2[unique_id].isin(data1[unique_id])]
    elif filter_count == "second":
        data2 = data2[data2[filter_column1] > filter1 & data2[filter_column1] < filter2 & data2[filter_column2] > filter3 & data2[filter_column2] < filter4]
        data1 = data1[data1[unique_id].isin(data2[unique_id])]
    elif filter_count == "both":
        data1 = data1[data1[filter_column1] > filter1 & data1[filter_column1] < filter2 & data1[filter_column2] > filter3 & data1[filter_column2] < filter4]
        data2 = data2[data2[filter_column1] > filter1 & data2[filter_column1] < filter2 & data2[filter_column2] > filter3 & data2[filter_column2] < filter4]
    else:
        print("Invalid filter_count. Please enter 'first', 'second', or 'both'.")

    if len(data1) == 0 or len(data2) == 0:
        # print("No data found for the specified filters.")
        return [None, None, 0, 0]
    
    sample1 = data1[variable]
    sample2 = data2[variable]

    ks_test_results = stats.ks_2samp(sample1, sample2)

    return [ks_test_results.statistic, ks_test_results.pvalue, len(data1), len(data2)]
    