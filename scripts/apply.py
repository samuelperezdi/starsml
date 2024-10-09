import pandas as pd
import numpy as np
import joblib
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import transform_features, create_new_columns, FEATURE_NAMES
from src.utils import remove_duplicate_columns

import time

def load_dtypes(json_path):
    """
    load data types from a JSON file.
    
    Args:
    json_path (str): Path to the JSON file containing data types
    
    Returns:
    dict: A dictionary of column names and their data types
    """
    with open(json_path, 'r') as f:
        dtypes_dict = json.load(f)
    
    # convert string representations of dtypes back to actual dtype objects
    for col, dtype in dtypes_dict.items():
        if dtype.startswith('datetime'):
            dtypes_dict[col] = 'datetime64[ns]'
        else:
            dtypes_dict[col] = f"{dtype}"
    
    return dtypes_dict

def apply_model_and_calculate_probabilities(df, model_path, output_path):
    """apply model to benchmark dataset, calculate probabilities, and save results"""
    def preprocess_df(df):
        df, _ = transform_features(df, model_type='lgbm', log_transform=False)
        return df

    # load model and config
    model = joblib.load(model_path)

    #df = df.sample(100)
    
    # prepare features
    create_new_columns(df)
    X = df[FEATURE_NAMES]
    X = preprocess_df(X)
    
    print('transformed features.')
    # apply model
    df['p_match_ind'] = model.predict_proba(X)[:, 1]
    df['p_prod'] = df['p_i'] * df['p_match_ind']
    
    print('model inference done.')
    # just do the thing above, not group anything.

    def process_group(group):
        start_time = time.time()

        any_prob_model, probs_model = calculate_probabilities(group['p_match_ind'].values)
        
        group['p_match_any'] = any_prob_model
        group['p_match_norm'] = probs_model
        group['p_prod_any'] = group['p_match_any'] * group['p_any']
        group['p_prod_norm'] = group['p_match_norm'] * group['p_i']

        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"source: {group['csc21_name'].iloc[0]}, time: {processing_time:.4f} seconds")
        return group
    
    # apply processing to each group
    result_df = df.groupby('csc21_name')[df.columns].apply(process_group).reset_index(drop=True)

    # save result
    result_df.to_parquet(output_path, index=False, engine='fastparquet')
    print(f"results saved to {output_path}")
    
    return 0

def calculate_probabilities(probabilities):
    """calculate probabilities for match, and any_match possibilities."""
    
    # calculate the no-match probability
    p_no_match = np.prod(1 - probabilities)
    all_hyp = np.concatenate([[p_no_match], probabilities])

    #print(p_no_match/np.sum(all_hyp), p_no_match)
    
    # calculate the any-match probability (complement of no-match)
    p_any_match = 1 - p_no_match/np.sum(all_hyp)
    
    # normalize the match probabilities
    p_normalized_matches = probabilities / np.sum(probabilities)
    
    return p_any_match, p_normalized_matches

if __name__ == "__main__":
    #loaded_dtypes = load_dtypes('column_dtypes.json')
    #df = pd.read_parquet('../out_data/nway_csc21_gaia3_full.parquet', engine='fastparquet')
    print("data loaded...")
    df = pd.read_parquet('../out_data/benchmark_set.parquet')

    print('starting processing now...')
    result_df = apply_model_and_calculate_probabilities(
        df,
        #'models/lgbm_default_nolog_none_seed42_20240829_142534/subset_9/model.joblib',
        'jobs/models/X100_hyperparam_lgbm_0-3_20240927_152823/model.joblib',
        'benchmark_results_x1000_negatives.parquet'
        #'benchmark_results_with_probabilities_1.csv'
    )