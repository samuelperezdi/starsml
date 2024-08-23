import sys
import os
import argparse

# add parent directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from typing import List

import pandas as pd
from astropy.table import Table

from src.data import (
    read_votable_to_dataframe,
    convert_byte_columns_to_str,
    filter_multiple_matches,
    get_most_probable_matches,
    get_second_most_probable_matches,
    get_last_probable_matches,
    prepare_final_dataset,
    include_classifications,
    read_gzipped_votable_to_dataframe
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_paths():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        'data': os.path.join(base_path, 'data'),
        'out_data': os.path.join(base_path, 'out_data')
    }

def load_nway_data(file_path: str) -> pd.DataFrame:
    t_nway = Table.read(file_path, format='fits')
    t_nway = convert_byte_columns_to_str(t_nway)
    df_nway_all = t_nway.to_pandas()
    # correct chandra ids
    df_nway_all['CSC21_CSCID'] = df_nway_all['CSC21_CSCID'].str.replace('_', ' ').str.strip()

    # generate dataframe of possible chandra matches
    nway_csc21_possible_matches_count = pd.DataFrame(df_nway_all[['CSC21_CSCID']].value_counts(), columns=['count'])
    
    # include them in df_nway_all 
    df_nway_all = df_nway_all.merge(nway_csc21_possible_matches_count, left_on='CSC21_CSCID', right_on='CSC21_CSCID', how='left')
    
    return df_nway_all

def process_matches(df_nway_all: pd.DataFrame, df_csc_all: pd.DataFrame) -> tuple:
    df_filtered = filter_multiple_matches(df_nway_all, 'CSC21_CSCID')
    df_most_probable = get_most_probable_matches(df_nway_all, 'CSC21_CSCID', 'p_i')
    df_second_most_probable = get_second_most_probable_matches(df_filtered, 'CSC21_CSCID', 'p_i')
    df_last_probable = get_last_probable_matches(df_filtered, 'CSC21_CSCID', 'p_i')

    return (
        prepare_final_dataset(df_most_probable, df_csc_all),
        prepare_final_dataset(df_second_most_probable, df_csc_all),
        prepare_final_dataset(df_last_probable, df_csc_all)
    )

def load_classifications(data_path: str) -> tuple:
    return (
        pd.read_csv(os.path.join(data_path, 'yangetal_gcs.csv')),
        pd.read_csv(os.path.join(data_path, 'yangetal_training.csv')),
        pd.read_csv(os.path.join(data_path, 'uniquely_classified.csv'))
    )

def add_classifications(df: pd.DataFrame, classifications: List[pd.DataFrame]) -> pd.DataFrame:
    yangetal_gcs, yangetal_training, perezdiazetal_class = classifications
    df = include_classifications(df, yangetal_gcs, 'csc21_name', 'CSCv2_name', ['Class'], {'Class': 'yangetal_gcs_class'})
    df = include_classifications(df, yangetal_training, 'csc21_name', 'name', ['Class'], {'Class': 'yangetal_training_class'})
    df = include_classifications(df, perezdiazetal_class, 'csc21_name', 'name', ['agg_master_class'], {'agg_master_class': 'perezdiazetal_class'})
    return df

def process_gaia_data(df: pd.DataFrame, gaia_props: pd.DataFrame) -> pd.DataFrame:
    df['gaia3_source_id'] = df['gaia3_source_id'].astype(str).str.strip()
    gaia_props['gaia3_source_id'] = gaia_props['gaia3_source_id'].astype(str).str.strip()
    return pd.merge(df, gaia_props, on='gaia3_source_id', how='left')

def main(test_mode=False):
    paths = setup_paths()
    os.makedirs(paths['out_data'], exist_ok=True)
    
    try:
        df_nway_all = load_nway_data(os.path.join(paths['data'], 'nway_CSC21_GAIA3.fits'))
        if test_mode:
            df_nway_all = df_nway_all.head(1000)  # Limit to first 1000 rows for testing
        
        df_csc_all = read_votable_to_dataframe(os.path.join(paths['data'], 'csc_all_1.vot'))
        if test_mode:
            df_csc_all = df_csc_all.head(1000)  # Limit to first 1000 rows for testing
        
        prepared_most_probable, prepared_second_prob, prepared_last_prob = process_matches(df_nway_all, df_csc_all)
        
        classifications = load_classifications(paths['data'])
        
        # for df, name in zip([prepared_most_probable, prepared_second_prob, prepared_last_prob], 
        #                     ['most_prob', 'second_most_prob', 'last_prob']):
        #     df_with_class = add_classifications(df, classifications)
        #     df_with_class.to_csv(os.path.join(paths['out_data'], f'{name}_class.csv'), index=False)
        
        # load and process gaia data
        gaia_props = read_gzipped_votable_to_dataframe(os.path.join(paths['data'], 'additional_gaia_properties-result.vot.gz'))
        gaia_selected_props = gaia_props[['gaia3_source_id', 'parallax_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error',
                                          'phot_g_mean_flux', 'phot_g_mean_flux_error', 'phot_bp_mean_flux',
                                          'phot_bp_mean_flux_error', 'phot_rp_mean_flux', 'phot_rp_mean_flux_error',
                                          'radial_velocity', 'radial_velocity_error', 'vbroad', 'vbroad_error',
                                          'phot_variable_flag', 'classprob_dsc_combmod_quasar', 'classprob_dsc_combmod_galaxy',
                                          'classprob_dsc_combmod_star', 'distance_gspphot', 'distance_gspphot_lower',
                                          'distance_gspphot_upper']]
        
        for df, name in zip([prepared_most_probable, prepared_second_prob, prepared_last_prob], 
                            ['most_prob', 'second_most_prob', 'last_prob']):
            df_with_class = add_classifications(df, classifications)
            df_with_gaia = process_gaia_data(df_with_class, gaia_selected_props)
            df_with_gaia.to_csv(os.path.join(paths['out_data'], f'{name}_class_gaia_props.csv'), index=False)
        
        logging.info(f"data processing completed successfully in {'test' if test_mode else 'full'} mode.")
    except Exception as e:
        logging.error(f"an error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creates tables with most probable and negative matches.')
    parser.add_argument('--test', action='store_true', help='Run in test mode with limited data')
    args = parser.parse_args()
    
    main(test_mode=args.test)