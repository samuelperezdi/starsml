import numpy as np
import pandas as pd
import os
import sys
import logging

from src.utils import votable_to_pandas
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u

from astropy.coordinates import Angle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)

def setup_paths():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        'out_data': os.path.join(base_path, 'out_data'),
        'data': os.path.join(base_path, 'data')
    }

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
    df_nway['csc21_name'] = df_nway['csc21_name'].str.replace('_', ' ')
    df_nway['csc21_name'] = df_nway['csc21_name'].str.strip()
    df_csc['name'] = df_csc['name'].str.strip()
    merged_df = df_nway.merge(df_csc, left_on='csc21_name', right_on='name', how='left')

    columns_mapping = {
        'CSC21_CSCID': 'csc21_name',
        'CSC21_RA': 'csc21_ra',
        'CSC21_Dec': 'csc21_dec',
        'CSC21_Errmaj' : 'csc21_errmaj',
        'CSC21_Errmin' : 'csc21_errmin',
        'CSC21_ErrPA' : 'csc21_errpa',
        'GAIA3_source_id': 'gaia3_source_id',
        'GAIA3_ra': 'gaia3_ra',
        'GAIA3_dec': 'gaia3_dec',
        'GAIA3_era' : 'gaia3_era',
        'GAIA3_edec' : 'gaia3_edec',
        'GAIA3_epa' : 'gaia3_epa',
        'GAIA3_phot_g_mean_mag': 'phot_g_mean_mag',
        'GAIA3_phot_bp_mean_mag': 'phot_bp_mean_mag',
        'GAIA3_phot_rp_mean_mag': 'phot_rp_mean_mag',
        'GAIA3_bp_rp': 'bp_rp',
        'GAIA3_bp_g': 'bp_g',
        'GAIA3_g_rp': 'g_rp',
        'GAIA3_parallax': 'parallax',
        'GAIA3_parallax_over_error': 'parallax_over_error',
        'Separation_GAIA3_CSC21': 'separation',
    }

    # rename columns
    merged_df.rename(columns=columns_mapping, inplace=True)

    # final columns
    final_features = [
        'csc21_name', 'csc21_ra', 'csc21_dec',
        'csc21_errmaj',
        'csc21_errmin',
        'csc21_errpa',
        'hard_hs',
        'hard_hm',
        'hard_hm_lolim',
        'hard_hm_hilim',
        'hard_ms',
        'hard_ms_lolim',
        'hard_ms_hilim',
        'var_intra_prob_b',
        'var_intra_index_b',
        'var_inter_prob_b',
        'var_inter_index_b',
        'var_inter_sigma_b',
        'extent_flag',
        'pileup_flag',
        'var_flag',
        'src_area_b',
        'photflux_aper_b',
        'photflux_aper_hilim_b',
        'photflux_aper_lolim_b',
        'acis_time', 
        'min_theta_mean',
        'yangetal_gcs_class',
        'yangetal_training_class',
        'perezdiazetal_class',
        #
        'gaia3_source_id', 'gaia3_ra', 'gaia3_dec', 
        'gaia3_era', 'gaia3_edec', 'gaia3_epa',
        'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'bp_rp', 'bp_g', 'g_rp', 
        'parallax', 'parallax_error', 'parallax_over_error', 
        'pmra', 'pmra_error', 'pmdec', 'pmdec_error',
        'phot_g_mean_flux', 'phot_g_mean_flux_error', 'phot_bp_mean_flux',
        'phot_bp_mean_flux_error', 'phot_rp_mean_flux', 'phot_rp_mean_flux_error',
        'radial_velocity', 'radial_velocity_error', 'vbroad', 'vbroad_error',
        'phot_variable_flag', 'classprob_dsc_combmod_quasar', 'classprob_dsc_combmod_galaxy',
        'classprob_dsc_combmod_star', 'distance_gspphot', 'distance_gspphot_lower',
        'distance_gspphot_upper',
        'separation', 
        'dist_bayesfactor', 
        'dist_post', 
        'p_single', 
        'p_any', 
        'p_i', 
        'match_flag',
        'count'
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

def create_additional_negatives():
    """
    find the nearest match in the second probable dataset for each missing id, combine optical properties from the match 
    with x-ray properties from the first_prob_df, and save to 'additional_negatives.csv'.

    parameters:
    - def_first_prob_df: pd.DataFrame, the first probability dataframe
    - def_second_prob_df: pd.DataFrame, the second probability dataframe
    - missing_ids: set, the chandra_source_id values missing in the last probable dataframe

    returns:
    - None
    """
    paths = setup_paths()

    # read input CSV files
    def_last_prob_df = pd.read_csv(os.path.join(paths['out_data'], 'last_prob_class_gaia_props.csv'))
    def_second_prob_df = pd.read_csv(os.path.join(paths['out_data'], 'second_most_prob_class_gaia_props.csv'))
    def_first_prob_df = pd.read_csv(os.path.join(paths['out_data'], 'most_prob_class_gaia_props.csv'))

    additional_negatives = []

    # check which chandra sources are missing a negative
    chandra_ids_in_pos = def_first_prob_df['csc21_name'].unique()
    missing_ids = set(chandra_ids_in_pos) - set(def_last_prob_df['csc21_name'])

    for missing_id in missing_ids:
        pos_row = def_first_prob_df[def_first_prob_df['csc21_name'] == missing_id].iloc[0]
        ra_pos = pos_row['csc21_ra']
        dec_pos = pos_row['csc21_dec']

        second_prob_matches = def_second_prob_df

        # calculate angular distances using astropy
        pos_coord = SkyCoord(ra=ra_pos * u.deg, dec=dec_pos * u.deg)
        match_coords = SkyCoord(ra=second_prob_matches['gaia3_ra'].values * u.deg, dec=second_prob_matches['gaia3_dec'].values * u.deg)

        distances = pos_coord.separation(match_coords)
        min_distance_index = distances.argmin()
        nearest_match = second_prob_matches.iloc[min_distance_index]

        # combine x-ray properties from first_prob_df with optical properties from nearest match
        combined_row = pos_row.copy()
        optical_columns = ['gaia3_source_id', 'gaia3_ra', 'gaia3_dec', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag',
       'bp_rp', 'bp_g', 'g_rp', 'parallax', 'parallax_over_error', 'parallax_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error',
       'phot_g_mean_flux', 'phot_g_mean_flux_error', 'phot_bp_mean_flux',
       'phot_bp_mean_flux_error', 'phot_rp_mean_flux',
       'phot_rp_mean_flux_error', 'radial_velocity', 'radial_velocity_error',
       'vbroad', 'vbroad_error', 'phot_variable_flag',
       'classprob_dsc_combmod_quasar', 'classprob_dsc_combmod_galaxy',
       'classprob_dsc_combmod_star', 'distance_gspphot',
       'distance_gspphot_lower', 'distance_gspphot_upper']
        nway_columns = ['dist_bayesfactor',
       'dist_post', 'p_single', 'p_any', 'p_i', 'match_flag']
        
        for col in optical_columns:
            combined_row[col] = nearest_match[col]

        for col in nway_columns:
            combined_row[col] = np.nan

        combined_row['separation'] = distances[min_distance_index]

        additional_negatives.append(combined_row)

    additional_negatives_df = pd.DataFrame(additional_negatives)
    additional_negatives_df.to_csv(os.path.join(paths['out_data'], 'additional_negatives.csv'), index=False)

def get_data(separation_thresholds):
    """
    process and filter data, generate df_pos, df_neg for each threshold range.
    Parameters:
    - separation_thresholds: dict, thresholds for different off-axis ranges
    Returns:
    - results: dict, containing positive and negative datasets for each threshold range
    """
    paths = setup_paths()
   
    # read input CSV files
    def_last_prob_df = pd.read_csv(os.path.join(paths['out_data'], 'last_prob_class_gaia_props.csv'))
    def_first_prob_df = pd.read_csv(os.path.join(paths['out_data'], 'most_prob_class_gaia_props.csv'))
   
    # extract unique csc21_name values from the first probability dataframe
    chandra_ids_in_first = def_first_prob_df['csc21_name'].unique()
    # filter last probability dataframe based on csc21_name values
    filtered_last_prob_df = def_last_prob_df[def_last_prob_df['csc21_name'].isin(chandra_ids_in_first)]
    
    # apply separation thresholds based on min_theta_mean
    def_first_prob_df['threshold_label'] = pd.cut(
        def_first_prob_df['min_theta_mean'],
        bins=[0, 3, 6, float('inf')],
        labels=['0-3', '3-6', '6+']
    )
    
    # initialize results dictionary
    results = {
        '0-3': {'df_pos': None, 'df_neg': None},
        '3-6': {'df_pos': None, 'df_neg': None},
        '6+': {'df_pos': None, 'df_neg': None}
    }
    
    # check if additional negatives file exists
    additional_negatives_file = os.path.join(paths['out_data'], 'additional_negatives.csv')
    if not os.path.exists(additional_negatives_file):
        logging.info("Generating additional negatives...")
        create_additional_negatives()
    else:
        logging.info("Additional negatives file found. Loading existing data...")
    
    # read the additional negatives
    additional_negatives_df = pd.read_csv(additional_negatives_file)
    
    # concatenate last_prob with additional negatives
    combined_last_prob_df = pd.concat([filtered_last_prob_df, additional_negatives_df], ignore_index=True)
    
    # process each threshold range
    for label in results.keys():
        threshold = separation_thresholds[label]
       
        # filter positive cases
        df_pos = def_first_prob_df.query(
            f'threshold_label == "{label}" and separation <= {threshold}'
        )
        chandra_ids_in_pos = df_pos['csc21_name'].unique()
        
        # filter negative cases
        df_neg = combined_last_prob_df[combined_last_prob_df['csc21_name'].isin(chandra_ids_in_pos)]
        df_pos.sort_values('csc21_name', inplace=True)
        df_neg.sort_values('csc21_name', inplace=True)
        
        # store positive and negative sets
        results[label]['df_pos'] = df_pos
        results[label]['df_neg'] = df_neg
    
    return results

    # replace zero flux values with NaN
    #df_pos['flux_aper_b'].replace(0, np.nan, inplace=True)
    #df_neg['flux_aper_b'].replace(0, np.nan, inplace=True)

    # calculate gmag_logflux
    #df_pos['gmag_logflux'] = df_pos['phot_g_mean_mag'] + np.log10(df_pos['flux_aper_b'] / 1e-13) * 2.5
    #df_neg['gmag_logflux'] = df_neg['phot_g_mean_mag'] + np.log10(df_neg['flux_aper_b'] / 1e-13) * 2.5

def read_data(separation_thresholds, folder='', suffix=''):
    """
    read saved positive and negative datasets for each threshold range from csv files.

    parameters:
    - separation_thresholds: dict, thresholds for different off-axis ranges

    returns:
    - results: dict, containing positive and negative datasets for each threshold range
    """
    paths = setup_paths()
    os.makedirs(paths['out_data'], exist_ok=True)
    output_path = paths['out_data']
    if folder:
        output_path = os.path.join(paths['out_data'], f'{folder}')

    results = {key: {'df_pos': None, 'df_neg': None} for key in separation_thresholds.keys()}

    # read the data from csv files
    for key in separation_thresholds.keys():
        results[key]['df_pos'] = pd.read_parquet(os.path.join(output_path, f'df_pos_{key}{suffix}.parquet'))
        results[key]['df_neg'] = pd.read_parquet(os.path.join(output_path, f'df_neg_{key}{suffix}.parquet'))

    return results


def create_hashes(df, features_to_hash, num_bins, method='bin_hash'):
    if method == 'all':
        return pd.Series(['all'] * len(df), index=df.index)
    
    def create_bin_edges(series, num_bins):
        min_val, max_val = series.min(), series.max()
        return np.linspace(min_val, max_val, num_bins + 1)

    def assign_to_bin(value, bin_edges):
        return np.digitize(value, bin_edges) - 1

    feature_bins = {}
    categorical_mappings = {}
    for feature in features_to_hash:
        if pd.api.types.is_numeric_dtype(df[feature]):
            feature_bins[feature] = create_bin_edges(df[feature], num_bins)
        else:
            unique_values = df[feature].unique()
            categorical_mappings[feature] = {val: i % num_bins for i, val in enumerate(unique_values)}

    def create_bin_id(row):
        bin_ids = []
        for feature in features_to_hash:
            if pd.api.types.is_numeric_dtype(df[feature]):
                bin_id = assign_to_bin(row[feature], feature_bins[feature])
            else:
                bin_id = categorical_mappings[feature].get(row[feature], num_bins - 1)
            bin_ids.append(str(bin_id).zfill(2))
        return ''.join(bin_ids)

    return df.apply(create_bin_id, axis=1)

def get_data_full_negatives(separation_thresholds, num_random_negatives=1000, num_bins=100, doppelganger_method=''):
    paths = setup_paths()
    
    # read full parquet file
    df_full = pd.read_parquet(os.path.join(paths['out_data'], 'nway_csc21_gaia3_full.parquet'))
    
    features_to_hash = ['phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag',
                        'bp_rp', 'bp_g', 'g_rp', 'parallax', 'pmra', 'pmdec',
                        'phot_g_mean_flux', 'phot_bp_mean_flux',
                        'phot_rp_mean_flux', 'radial_velocity', 'vbroad', 'phot_variable_flag',
                        'classprob_dsc_combmod_quasar', 'classprob_dsc_combmod_galaxy',
                        'classprob_dsc_combmod_star', 'distance_gspphot']

    if doppelganger_method:
        df_full = process_hashes(df_full, features_to_hash, num_bins, doppelganger_method, paths)
        print('finished hashing.')
        
    # identify most probable matches (positives)
    df_pos = df_full.loc[df_full.groupby('csc21_name')['p_i'].idxmax()]
    
    # identify least probable matches (clear negatives)
    df_pos_comp = df_full[~df_full.index.isin(df_pos.index)].copy()
    df_neg = df_pos_comp.loc[df_pos_comp.groupby('csc21_name')['p_i'].idxmin()]
    df_neg['negative_type'] = 'clear_negative'
    
    # prepare column lists
    x_ray_columns = df_full.columns[:df_full.columns.get_loc('gaia3_source_id')].tolist()
    gaia_columns = df_full.columns[df_full.columns.get_loc('gaia3_source_id'):].tolist()

    # delete negatives that are dopplegangers of positives
    if doppelganger_method:
        positive_hashes = set(df_pos['hash'])
        df_pos_comp_filtered = df_pos_comp[~df_pos_comp['hash'].isin(positive_hashes)]
    else:
        df_pos_comp_filtered = df_pos_comp
    
    # add intermediate cases
    df_intermediate = df_pos_comp[df_pos_comp['separation'] > 5].copy()
    df_intermediate['negative_type'] = 'intermediate'
    
    # positives by off-axis angle
    df_pos['threshold_label'] = pd.cut(
        df_pos['min_theta_mean'],
        bins=[0, 3, 6, float('inf')],
        labels=['0-3', '3-6', '6+']
    )
    
    results = {
        '0-3': {'df_pos': None, 'df_neg': None},
        '3-6': {'df_pos': None, 'df_neg': None},
        '6+': {'df_pos': None, 'df_neg': None}
    }
    
    print('Before going to each threshold.')

    # process each threshold range
    for label in results.keys():
        threshold = separation_thresholds[label]
        
        pos_range = df_pos.query(f'threshold_label == "{label}" and separation <= {threshold} and p_any >= 0.5')
        chandra_ids_in_pos = pos_range['csc21_name'].unique()
        
        # generate random negatives
        df_random_neg = fast_random_match_pandas(pos_range[x_ray_columns], 
                                                 df_pos_comp_filtered[gaia_columns], 
                                                 num_random_negatives)
        
        df_random_neg['negative_type'] = 'random'

        df_random_neg.loc[:, 'separation':'count'] = -1

        # combine all negatives
        neg_range = pd.concat([
            df_neg[df_neg['csc21_name'].isin(chandra_ids_in_pos)],
            df_random_neg,
            df_intermediate[df_intermediate['csc21_name'].isin(chandra_ids_in_pos)]
        ], ignore_index=True)
        
        results[label]['df_pos'] = pos_range
        results[label]['df_neg'] = neg_range
        
        print(f"Range {label}: {len(pos_range)} positives, {len(neg_range)} negatives")
    
    return results

def fast_random_match_pandas(x_ray_df, gaia_df, num_random_negatives):
    # repeat each x-ray row num_random_negatives
    x_ray_repeated = x_ray_df.loc[x_ray_df.index.repeat(num_random_negatives)].reset_index(drop=True)
    
    # sample random rows from gaia_df
    total_samples = len(x_ray_df) * num_random_negatives

    random_gaia = gaia_df.sample(n=total_samples, replace=True).reset_index(drop=True)
        # rename and map Gaia columns
    column_mapping = {
        'SOURCE_ID': 'gaia3_source_id',
        'ra': 'gaia3_ra',
        'dec': 'gaia3_dec'
    }
    
    df_negs_processed = random_gaia.rename(columns=column_mapping)
    
    # add missing columns with defaults
    default_columns = {
        'gaia3_epa' : -1,
        'gaia3_era': -1,
        'gaia3_edec': -1,
        'separation': -1,
        'dist_bayesfactor': -1,
        'dist_post': -1,
        'p_single': -1,
        'p_any': -1,
        'p_i': -1,
        'match_flag': 0,
        'count': -1,
        'benchmark_label': 0
    }
    
    for col, default_val in default_columns.items():
        df_negs_processed[col] = default_val

    # combine the repeated x-ray data with the random gaia data
    result = pd.concat([x_ray_repeated, random_gaia], axis=1)
    
    return result

def fast_random_match_duckdb(x_ray_df, negative_gaia_sources_path, num_random_negatives):
    import duckdb
    con = duckdb.connect()
    total_samples = len(x_ray_df) * num_random_negatives
    
    query = f"""
    SELECT 
        SOURCE_ID as gaia3_source_id,
        ra as gaia3_ra,
        dec as gaia3_dec,
        *  EXCLUDE (SOURCE_ID, ra, dec)
    FROM '{negative_gaia_sources_path}'
    USING SAMPLE {total_samples} ROWS
    """
    
    random_gaia = con.execute(query).df()
    
    # repeat x-ray rows and combine with random gaia
    x_ray_repeated = x_ray_df.loc[x_ray_df.index.repeat(num_random_negatives)].reset_index(drop=True)
    result = pd.concat([x_ray_repeated, random_gaia], axis=1)
    
    # add default columns 
    default_columns = {
        'gaia3_epa': -1, 'gaia3_era': -1, 'gaia3_edec': -1,
        'separation': -1, 'dist_bayesfactor': -1, 'dist_post': -1,
        'p_single': -1, 'p_any': -1, 'p_i': -1,
        'match_flag': -1, 'count': -1, 'benchmark_label': 0
    }
    
    for col, val in default_columns.items():
        result[col] = val
        
    con.close()
    return result

def process_hashes(df_full, features_to_hash, num_bins, doppelganger_method, paths):
    hash_file = os.path.join(paths['out_data'], f'hashes_{doppelganger_method}_{num_bins}_bins.parquet')
    
    if os.path.exists(hash_file):
        df_full['hash'] = pd.read_parquet(hash_file)['hash']
    else:
        df_transform = df_full.copy()
        
        log_transform_feats = [
            'parallax', 'parallax_error', 'photflux_aper_b', 'phot_g_mean_flux', 'phot_bp_mean_flux',
            'phot_rp_mean_flux', 'radial_velocity', 'vbroad', 'distance_gspphot'
        ]
        
        for feat in log_transform_feats:
            if feat in df_transform.columns:
                if feat == 'parallax':
                    df_transform[feat] = np.log10(df_transform[feat].clip(5e-5, 1))
                else:
                    df_transform[feat] = np.log10(df_transform[feat].clip(lower=1e-300))
        
        df_full['hash'] = create_hashes(df_transform, features_to_hash, num_bins, method=doppelganger_method)
        df_full[['hash']].to_parquet(hash_file)
    
    print('finished hashing.')
    return df_full

def get_data_basic_matches(df_full, off_axis_range, separation_threshold):
    """
    identify and filter probable matches from a dataframe
    parameters:
    df_full (pd.DataFrame): input dataframe with match data
    off_axis_range (str): range label for off-axis angle
    separation_threshold (float): maximum separation threshold
    returns:
    tuple: filtered positive and negative matches dataframes
    """
    
    # identify most probable matches (positives)
    df_pos = df_full.loc[df_full.groupby('csc21_name')['p_i'].idxmax()]
    
    # identify least probable matches (negatives)
    df_pos_comp = df_full[~df_full.index.isin(df_pos.index)]
    df_neg = df_pos_comp.loc[df_pos_comp.groupby('csc21_name')['separation'].idxmax()]
    df_neg = df_neg[df_neg['separation'] >= 5].copy()
    df_neg['negative_type'] = 'clear_negative'


    # add intermediate cases
    df_intermediate = df_pos_comp[df_pos_comp['separation'] >= 5].copy()
    df_intermediate['negative_type'] = 'intermediate'

    # label positives by off-axis angle
    df_pos['threshold_label'] = pd.cut(
        df_pos['min_theta_mean'],
        bins=[0, 3, 6, float('inf')],
        labels=['0-3', '3-6', '6+']
    )
    
    # filter for specific range and separation threshold
    pos_range = df_pos.query(f'threshold_label == "{off_axis_range}" and separation <= {separation_threshold} and p_any >= 0.9')
    chandra_ids_in_pos = pos_range['csc21_name'].unique()
    #neg_range = df_neg[df_neg['csc21_name'].isin(chandra_ids_in_pos)]

    neg_range = pd.concat([
        df_neg[df_neg['csc21_name'].isin(chandra_ids_in_pos)],
        df_intermediate[df_intermediate['csc21_name'].isin(chandra_ids_in_pos)]
    ], ignore_index=True)
    
    print(f"Range {off_axis_range}: {len(pos_range)} positives, {len(neg_range)} negatives")
    
    return pos_range, neg_range