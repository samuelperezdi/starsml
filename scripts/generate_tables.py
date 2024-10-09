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
    read_gzipped_votable_to_dataframe,
    read_data
)

from src.utils import preprocess

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_paths():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        'data': os.path.join(base_path, 'data'),
        'out_data': os.path.join(base_path, 'out_data')
    }

def load_nway_data(folder_path: str) -> pd.DataFrame:
    file_name = 'nway_CSC21_GAIA3'
    parquet_path = os.path.join(folder_path, f'{file_name}.parquet')
    fits_path = os.path.join(folder_path, f'{file_name}.fits')
    
    if os.path.exists(parquet_path):
        # read from parquet
        df_nway_all = pd.read_parquet(parquet_path, engine='fastparquet')
    else:
        # read from fits
        t_nway = Table.read(fits_path, format='fits')
        t_nway = convert_byte_columns_to_str(t_nway)
        df_nway_all = t_nway.to_pandas()
        # save as parquet for future use
        df_nway_all.to_parquet(parquet_path, index=False, engine='fastparquet')
    
    print('nway all finished reading!')
    # correct chandra ids
    df_nway_all['CSC21_CSCID'] = df_nway_all['CSC21_CSCID'].str.replace('_', ' ').str.strip()
    
    # generate dataframe of possible chandra matches
    nway_csc21_possible_matches_count = df_nway_all['CSC21_CSCID'].value_counts().rename('count').reset_index()
    
    # include them in df_nway_all
    df_nway_all = df_nway_all.merge(nway_csc21_possible_matches_count, left_on='CSC21_CSCID', right_on='CSC21_CSCID', how='left')
    
    return df_nway_all


def load_csc_data(folder_path: str) -> pd.DataFrame:
    file_name = 'csc_all'
    parquet_path = os.path.join(folder_path, f'{file_name}.parquet')
    vot_path = os.path.join(folder_path, f'{file_name}.vot')
    
    if os.path.exists(parquet_path):
        # read from parquet
        df_csc_all = pd.read_parquet(parquet_path, engine='fastparquet')
    else:
        # read from vot
        df_csc_all = read_votable_to_dataframe(vot_path)
        # save as parquet for future use
        df_csc_all.to_parquet(parquet_path, index=False, engine='fastparquet')
    print('csc all finished reading!')
    return df_csc_all

def load_add_gaia_props(folder_path: str) -> pd.DataFrame:
    file_name = 'gaia_props_all'
    parquet_path = os.path.join(folder_path, f'{file_name}.parquet')
    vot_path = os.path.join(folder_path, f'{file_name}.vot.gz')
    
    if os.path.exists(parquet_path):
        # read from parquet
        gaia_props = pd.read_parquet(parquet_path, engine='fastparquet')
    else:
        # read from vot
        gaia_props = read_gzipped_votable_to_dataframe(vot_path)
        # save as parquet for future use
        gaia_props.to_parquet(parquet_path, index=False, engine='fastparquet')
    
    return gaia_props


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
        pd.read_csv(os.path.join(data_path, 'yangetal_gcs.csv'), usecols=['CSCv2_name', 'Class']),
        pd.read_csv(os.path.join(data_path, 'yangetal_training.csv'), usecols=['name', 'Class']),
        pd.read_csv(os.path.join(data_path, 'uniquely_classified.csv'), usecols=['name', 'agg_master_class'])
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

def process_full_data(df_nway_all: pd.DataFrame, df_csc_all: pd.DataFrame, classifications: List[pd.DataFrame], gaia_props: pd.DataFrame) -> pd.DataFrame:

    print(df_nway_all.columns)

    df_nway_all.rename(columns={'GAIA3_source_id':'gaia3_source_id', 'CSC21_CSCID' : 'csc21_name'}, inplace=True)
    gaia_selected_props = gaia_props[['gaia3_source_id', 'parallax_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error',
                                'phot_g_mean_flux', 'phot_g_mean_flux_error', 'phot_bp_mean_flux',
                                'phot_bp_mean_flux_error', 'phot_rp_mean_flux', 'phot_rp_mean_flux_error',
                                'radial_velocity', 'radial_velocity_error', 'vbroad', 'vbroad_error',
                                'phot_variable_flag', 'classprob_dsc_combmod_quasar', 'classprob_dsc_combmod_galaxy',
                                'classprob_dsc_combmod_star', 'distance_gspphot', 'distance_gspphot_lower',
                                'distance_gspphot_upper']]
    
    # process gaia data
    df_full = process_gaia_data(df_nway_all, gaia_selected_props)

    # add classifications
    df_with_class = add_classifications(df_full, classifications)

    # prepare
    df_prepared = prepare_final_dataset(df_with_class, df_csc_all)
    
    return df_prepared

def get_test_sources(df_pos, df_neg, random_seed=42, test_size=0.3):
    # simulate the preprocessing split to get the test sources
    X_train, X_test, Y_train, Y_test, indices_train, indices_test, _, _ = preprocess(
        df_pos, df_neg, log_transform=False, model_type='lgbm', random_seed=random_seed, test_size=test_size
    )
    
    # Combine positive and negative dataframes
    df_combined = pd.concat([df_pos, df_neg], axis=0, ignore_index=True)
    
    # Get the CSC names for the test set
    test_sources = df_combined.iloc[indices_test]['csc21_name'].unique()
    
    return test_sources

def create_benchmark_set(full_data_path, output_path, folder='', random_seed=42, test_size=0.3):
    # load full dataset
    df_full = pd.read_parquet(full_data_path) if full_data_path.endswith('.parquet') else pd.read_csv(full_data_path)
    
    separation_thresholds = {
        '0-3': 1.3,
        '3-6': 1.3,
        '6+': 2.2
    }
    results = read_data(separation_thresholds, folder)
    df_pos = results['0-3']['df_pos']
    df_neg = results['0-3']['df_neg']
    
    test_sources = get_test_sources(df_pos, df_neg, random_seed, test_size)
    
    benchmark_set = df_full[df_full['csc21_name'].isin(test_sources)]
    benchmark_set['benchmark_label'] = np.where(benchmark_set['match_flag'] == 1, 1, 0)
    
    benchmark_name = os.path.basename(folder) if folder else 'default'
    output_path = output_path.replace('.csv', f'_{benchmark_name}.parquet')
    benchmark_set.to_parquet(output_path, index=False)
    logging.info(f"Benchmark set created and saved to {output_path}")

def main(process_type='default', test_mode=False, create_benchmark=False, folder='full_negatives'):
    paths = setup_paths()
    os.makedirs(paths['out_data'], exist_ok=True)
    try:
        if create_benchmark:
            full_data_path = os.path.join(paths['out_data'], 'nway_csc21_gaia3_full.parquet')
            benchmark_output_path = os.path.join(paths['out_data'], 'benchmark_set.parquet')
            create_benchmark_set(full_data_path, benchmark_output_path, folder)
        elif process_type == 'full':
            df_nway_all = load_nway_data(os.path.join(paths['data']))
            df_csc_all = load_csc_data(os.path.join(paths['data']))
            
            if test_mode:
                df_nway_all = df_nway_all.head(1000)
                df_csc_all = df_csc_all.head(1000)
            
            classifications = load_classifications(paths['data'])
            gaia_props = load_add_gaia_props(os.path.join(paths['data']))
            
            df_full = process_full_data(df_nway_all, df_csc_all, classifications, gaia_props)
            df_full.to_parquet(os.path.join(paths['out_data'], 'nway_csc21_gaia3_full.parquet'), index=False, engine='fastparquet')
            #df_full.to_csv(os.path.join(paths['out_data'], 'nway_csc21_gaia3_full.csv'), index=False)
            logging.info("full dataset processing completed.")
        else:
            prepared_most_probable, prepared_second_prob, prepared_last_prob = process_matches(df_nway_all, df_csc_all)
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
            
            logging.info("default processing completed.")
        
        logging.info(f"data processing completed successfully in {'test' if test_mode else 'full'} mode.")
    except Exception as e:
        logging.error(f"an error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creates tables with most probable and negative matches.')
    parser.add_argument('--test', action='store_true', help='Run in test mode with limited data')
    parser.add_argument('--process', choices=['default', 'full', 'all_negs'], default='default', help='Select processing type')
    parser.add_argument('--benchmark', action='store_true', help='Create benchmark set (requires full_dataset.csv to exist)')
    args = parser.parse_args()
    
    main(process_type=args.process, test_mode=args.test, create_benchmark=args.benchmark)