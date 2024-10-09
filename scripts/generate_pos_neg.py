#!/usr/bin/env python
# coding: utf-8

import sys
import os
import argparse
import logging
import pandas as pd

# add parent directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import (
    get_data,
    get_data_full_negatives
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)

def setup_paths():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        'out_data': os.path.join(base_path, 'out_data')
    }

def process_data(test_mode=False, full_negatives=False):
    paths = setup_paths()
    os.makedirs(paths['out_data'], exist_ok=True)

    separation_thresholds = {
        '0-3': 1.3,  # 0-3 arcmin
        '3-6': 1.3,  # 3-6 arcmin
        '6+': 2.2    # 6+ arcmin
    }

    # process and create the pos and neg datasets

    if full_negatives:
        os.makedirs(os.path.join(paths['out_data'], f'full_negatives'), exist_ok=True)
        results = get_data_full_negatives(separation_thresholds)
        out_path = os.path.join(paths['out_data'], f'full_negatives')
    else:
        results = get_data(separation_thresholds)
        out_path = paths['out_data']

    for range_key, data in results.items():
        df_pos = data['df_pos']
        df_neg = data['df_neg']

        if test_mode:
            df_pos = df_pos.head(100)  # limit to first 100 rows for testing
            df_neg = df_neg.head(100)  # limit to first 100 rows for testing

        df_pos.to_parquet(os.path.join(out_path, f'df_pos_{range_key}_X1000.parquet'), index=False)
        df_neg.to_parquet(os.path.join(out_path, f'df_neg_{range_key}_X1000.parquet'), index=False)

        logging.info(f"Saved positive and negative datasets for range {range_key}")

def main(test_mode=False, full_negatives=False):
    # try:
        process_data(test_mode, full_negatives)
        logging.info(f"Data processing completed successfully in {'test' if test_mode else 'full'} mode.")
    # except Exception as e:
    #     logging.error(f"An error occurred: {str(e)}")
    #     sys.exit(1)

if __name__ == "__main__":
    print('STARTED!')
    parser = argparse.ArgumentParser(description='Generate positive and negative tables')
    parser.add_argument('--test', action='store_true', help='Run in test mode with limited data')
    parser.add_argument('--full_negatives', action='store_true', help='Generate the negative set with all possible negatives.')
    args = parser.parse_args()
    
    main(test_mode=args.test, full_negatives=args.full_negatives)