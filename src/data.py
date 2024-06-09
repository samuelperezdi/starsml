import numpy as np
import pandas as pd
from astropy.table import Table
from src.utils import votable_to_pandas

def read_votable_to_dataframe(filepath, columns=None):
    """
    Read a VOTable file and convert it to a Pandas DataFrame.
    Optionally select specific columns.

    Parameters:
    - filepath: str, path to the VOTable file.
    - columns: list of str, specific columns to select (default is None).

    Returns:
    - pd.DataFrame
    """
    df = votable_to_pandas(filepath)
    if columns:
        df = df[columns]
    return df

def convert_byte_columns_to_str(table):
    """
    Convert byte-string columns in an Astropy Table to string columns.

    Parameters:
    - table: Astropy Table

    Returns:
    - Astropy Table with byte-string columns converted to strings
    """
    for col in table.columns:
        if table[col].dtype.kind == 'S':
            table[col] = table[col].astype(str)
    return table

def filter_multiple_matches(df, group_col):
    """
    Filter DataFrame rows that have multiple entries in a specified group.

    Parameters:
    - df: pd.DataFrame
    - group_col: str, column name to group by

    Returns:
    - pd.DataFrame with filtered rows
    """
    multiple_entries = df.groupby(group_col).filter(lambda x: len(x) > 1)[group_col].unique()
    return df[df[group_col].isin(multiple_entries)]

def get_most_probable_matches(df, group_col, prob_col):
    """
    Get the most probable matches from a DataFrame.

    Parameters:
    - df: pd.DataFrame
    - group_col: str, column name to group by
    - prob_col: str, column name of the probability metric

    Returns:
    - pd.DataFrame with the most probable matches
    """
    return df.loc[df.groupby(group_col)[prob_col].idxmax()]

def get_second_most_probable_matches(df, group_col, prob_col):
    """
    Get the second most probable matches from a DataFrame.

    Parameters:
    - df: pd.DataFrame
    - group_col: str, column name to group by
    - prob_col: str, column name of the probability metric

    Returns:
    - pd.DataFrame with the second most probable matches
    """
    sorted_df = df.sort_values([group_col, prob_col], ascending=[True, False])
    return sorted_df.groupby(group_col).nth(1).reset_index()

def get_last_probable_matches(df, group_col, prob_col):
    """
    Get the least probable matches from a DataFrame.

    Parameters:
    - df: pd.DataFrame
    - group_col: str, column name to group by
    - prob_col: str, column name of the probability metric

    Returns:
    - pd.DataFrame with the least probable matches
    """
    sorted_df = df.sort_values([group_col, prob_col], ascending=[True, False])
    return sorted_df.groupby(group_col).nth(-1).reset_index()

def prepare_final_dataset(df_nway, df_csc):
    """
    Merge df_nway and df_csc, and select columns in the desired order with new names.

    Parameters:
    - df_nway: pd.DataFrame, the first dataset
    - df_csc: pd.DataFrame, the second dataset

    Returns:
    - pd.DataFrame, the merged dataset with columns in the specified order
    """
    df_nway['CSC21_CSCID'] = df_nway['CSC21_CSCID'].str.replace('_', ' ')
    df_nway['CSC21_CSCID'] = df_nway['CSC21_CSCID'].str.strip()
    df_csc['name'] = df_csc['name'].str.strip()
    merged_df = df_nway.merge(df_csc, left_on='CSC21_CSCID', right_on='name', how='left')

    columns_mapping = {
        'CSC21_CSCID': 'csc21_name',
        'CSC21_RA': 'csc21_ra',
        'CSC21_Dec': 'csc21_dec',
        'GAIA3_source_id': 'gaia3_source_id',
        'GAIA3_ra': 'gaia3_ra',
        'GAIA3_dec': 'gaia3_dec',
        'GAIA3_phot_g_mean_mag': 'phot_g_mean_mag',
        'GAIA3_phot_bp_mean_mag': 'phot_bp_mean_mag',
        'GAIA3_phot_rp_mean_mag': 'phot_rp_mean_mag',
        'GAIA3_bp_rp': 'bp_rp',
        'GAIA3_bp_g': 'bp_g',
        'GAIA3_g_rp': 'g_rp',
        'GAIA3_parallax': 'parallax',
        'GAIA3_parallax_over_error': 'parallax_over_error',
        'hard_hs': 'hard_hs',
        'hard_hm': 'hard_hm',
        'hard_ms': 'hard_ms',
        'var_intra_prob_b': 'var_intra_prob_b',
        'var_inter_prob_b': 'var_inter_prob_b',
        'Separation_GAIA3_CSC21': 'separation',
        'dist_bayesfactor': 'dist_bayesfactor',
        'dist_post': 'dist_post',
        'p_single': 'p_single',
        'p_any': 'p_any',
        'p_i': 'p_i',
        'match_flag': 'match_flag'
    }

    # rename columns
    merged_df.rename(columns=columns_mapping, inplace=True)

    # final columns
    final_features = [
        'csc21_name', 'csc21_ra', 'csc21_dec', 'gaia3_source_id', 'gaia3_ra', 'gaia3_dec', 
        'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'bp_rp', 'bp_g', 'g_rp', 
        'parallax', 'parallax_over_error', 'hard_hs', 'hard_hm', 'hard_ms', 'var_intra_prob_b', 
        'var_inter_prob_b', 'separation', 'dist_bayesfactor', 'dist_post', 'p_single', 'p_any', 
        'p_i', 'match_flag'
    ]

    # Return the final dataframe with the selected columns
    return merged_df[final_features]

def include_classifications(df_base, df_additional, base_col='csc21_name', additional_col=None, additional_columns=None, rename_columns=None):
    """
    Include classifications from another dataset based on a common column with an option to rename the included columns.

    Parameters:
    - df_base: pd.DataFrame, the base dataset to which classifications will be added
    - df_additional: pd.DataFrame, the additional dataset containing classifications
    - base_col: str, the column name in the base dataset used for matching (default is 'csc21_name')
    - additional_col: str, the column name in the additional dataset used for matching
    - additional_columns: list of str, additional columns from the additional dataset to include
    - rename_columns: dict, dictionary mapping original column names to new names

    Returns:
    - pd.DataFrame, the base dataset with additional classifications and columns included
    """
    if additional_columns is None:
        additional_columns = []
        
    if rename_columns is None:
        rename_columns = {}

    # rename the matching column in the additional dataset to match the base dataset
    df_additional_renamed = df_additional.rename(columns={additional_col: base_col})
    
    # columns to merge
    columns_to_merge = [base_col] + additional_columns
    
    # select the columns
    df_additional_selected = df_additional_renamed[columns_to_merge]

    # rename additional columns
    df_additional_selected.rename(columns=rename_columns, inplace=True)
    
    # merge
    df_merged = df_base.merge(df_additional_selected, on=base_col, how='left')
    
    return df_merged

def read_gzipped_votable_to_dataframe(filepath):
    """
    Read a gzipped VOTable file and convert it to a Pandas DataFrame.

    Parameters:
    - filepath: str, path to the gzipped VOTable file

    Returns:
    - pd.DataFrame
    """
    import gzip
    with gzip.open(filepath, 'rb') as f:
        votable = Table.read(f, format='votable')
    return votable.to_pandas()

def get_complement(df, group_col, prob_col):
    """
    Get the complement of the most probable matches from a DataFrame.

    Parameters:
    - df: pd.DataFrame, the input DataFrame
    - group_col: str, column name to group by
    - prob_col: str, column name of the probability metric

    Returns:
    - pd.DataFrame with the complement of the most probable matches
    """
    indices_most_probable = df.groupby(group_col)[prob_col].idxmax()
    df_complement = df.loc[~df.index.isin(indices_most_probable)]
    
    return df_complement
