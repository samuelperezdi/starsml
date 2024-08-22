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
    get_data
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_paths():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        'out_data': os.path.join(base_path, 'out_data')
    }

def process_data(test_mode=False):
    paths = setup_paths()
    os.makedirs(paths['out_data'], exist_ok=True)



    separation_thresholds = {
        '0-3': 1.3,  # 0-3 arcmin
        '3-6': 1.3,  # 3-6 arcmin
        '6+': 2.2    # 6+ arcmin
    }

    # process and create the pos and neg datasets
    results = get_data(separation_thresholds)

    for range_key, data in results.items():
        df_pos = data['df_pos']
        df_neg = data['df_neg']

        if test_mode:
            df_pos = df_pos.head(100)  # limit to first 100 rows for testing
            df_neg = df_neg.head(100)  # limit to first 100 rows for testing

        df_pos.to_csv(os.path.join(paths['out_data'], f'df_pos_{range_key}.csv'), index=False)
        df_neg.to_csv(os.path.join(paths['out_data'], f'df_neg_{range_key}.csv'), index=False)

        logging.info(f"Saved positive and negative datasets for range {range_key}")

def main(test_mode=False):
    try:
        process_data(test_mode)
        logging.info(f"Data processing completed successfully in {'test' if test_mode else 'full'} mode.")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate positive and negative tables')
    parser.add_argument('--test', action='store_true', help='Run in test mode with limited data')
    args = parser.parse_args()
    
    main(test_mode=args.test)