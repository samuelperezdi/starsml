"""
utils.py

Author: Joshua D. Ingram

This file contains utility functions for the distinguishability of firrst and second best
matches analysis in the StarsML project.

Functions:
- function1: Description of function1.
- function2: Description of function2.
- ...
"""

def min_count_bins(df, column, min_count = 1000):
    """
    Define bins with a minimum count of data in each range.

    Parameters
    ---------
    df : pd.DataFrame
        Dataframe of data.

    column : str
        Column name for column to create bins for.

    min_count : int
        Minimum count for each bin.
    """

    