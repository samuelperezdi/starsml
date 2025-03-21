import os
import sys
import json

# add parent directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib

from src.utils import load_model, create_new_columns, transform_features, remove_duplicate_columns
from src.plot import plot_benchmark_results

def extract_and_store_dtypes(df, output_path):
    """
    Extract data types from a DataFrame and store them in a JSON file.
    
    Args:
    df (pd.DataFrame): Input DataFrame
    output_path (str): Path to save the JSON file
    
    Returns:
    dict: A dictionary of column names and their data types
    """
    # Extract dtypes
    dtypes_dict = df.dtypes.astype(str).to_dict()
    
    # Store in JSON file
    with open(output_path, 'w') as f:
        json.dump(dtypes_dict, f, indent=2)
    
    print(f"Data types stored in {output_path}")
    
    return dtypes_dict

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


def run_benchmark(model_path, benchmark_path, experiment_name):
    """run benchmark evaluation"""
    
    # load the model and its configuration
    model_info = load_model(model_path)
    model = model_info['model']
    config = model_info['experiment_info']
    
    print(f"Loaded model: {model}")
    print(f"Model configuration: {config}")

    loaded_dtypes = load_dtypes('column_dtypes.json')
    # load the benchmark dataset
    benchmark_df = pd.read_csv(benchmark_path, dtype=loaded_dtypes)
    benchmark_df = remove_duplicate_columns(benchmark_df)

    #dtypes_dict = extract_and_store_dtypes(benchmark_df, 'column_dtypes.json')
    
    # plot
    plot_benchmark_results(model, benchmark_df, config, experiment_name)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run benchmark evaluation on a trained model.")
    parser.add_argument("model_path", help="Path to the saved model file")
    parser.add_argument("benchmark_path", help="Path to the benchmark dataset CSV file")
    parser.add_argument("experiment_name", help="Name of the experiment")
    
    args = parser.parse_args()
    
    run_benchmark(args.model_path, args.benchmark_path, args.experiment_name)