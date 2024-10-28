import argparse
import logging
from pathlib import Path
import sys
import os
import joblib
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

sys.path.append(str(Path(__file__).parent.parent))
from src.data import read_data
from src.utils import preprocess, preprocess_cscid, train_and_tune_model, setup_paths, transform_features
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def downsample_negatives(df_pos, df_neg, intermediates=True, multiplier=100):
    if multiplier == 100:
        return df_neg
        
    # separate negatives by type
    clear_negs = df_neg[df_neg['negative_type'] == 'clear_negative'].copy()
    intermediate_negs = df_neg[df_neg['negative_type'] == 'intermediate'].copy() 
    random_negs = df_neg[df_neg['negative_type'] == 'random'].copy()

    # calculate samples needed for random negatives
    n_samples = len(df_pos) * multiplier
    
    # sample random negatives 
    if len(random_negs) > n_samples:
        random_negs = random_negs.sample(n=n_samples, random_state=42)

    # combine based on intermediates flag
    if intermediates:
        return pd.concat([clear_negs, intermediate_negs, random_negs])
    else:
        return pd.concat([clear_negs, random_negs])

def prepare_data(df_full):
    separation_thresholds = {'0-3': 1.3, '3-6': 1.3, '6+': 2.2}
    range_offaxis = '0-3'  # we're assuming range 0-3

    results = read_data(separation_thresholds, folder='full_negatives', suffix='')
    df_pos = results[range_offaxis]['df_pos']
    df_neg = results[range_offaxis]['df_neg']

    cscids = df_pos.csc21_name.unique()
    cscids_train, cscids_benchmark = train_test_split(cscids, test_size=0.2, random_state=42, shuffle=True)

    benchmark_set = df_full[df_full['csc21_name'].isin(cscids_benchmark)].copy()
    benchmark_set.loc[:, 'benchmark_label'] = np.where(benchmark_set['match_flag'] == 1, 1, 0)

    df_pos = df_pos[df_pos['csc21_name'].isin(cscids_train)].copy()
    df_neg = df_neg[df_neg['csc21_name'].isin(cscids_train)].copy()

    return df_pos, df_neg, benchmark_set

def train_model(experiment_name, multiplier, intermediates, df_pos, df_neg, benchmark_set):
    range_offaxis = '0-3'  # we're assuming range 0-3

    df_neg = downsample_negatives(df_pos, df_neg, intermediates, multiplier)
    X_train, X_test, y_train, y_test, _, _, cat_features, _ = preprocess_cscid(df_pos, df_neg, model_type='lgbm', test_size=0.2)

    model, y_pred, best_params = train_and_tune_model(
        X_train, X_test, y_train, y_test, cat_features, model_type='lgbm', hyperparameter_tuning=True
    )

    results_exp = {
        'model': model,
        'best_params': best_params,
        'X_test': X_test,
        'y_test': y_test,
        'benchmark_ids': benchmark_set['csc21_name'].tolist()
    }
    model_path = save_model(results_exp, experiment_name, range_offaxis)
    logging.info(f"Model saved to: {model_path}")

    # preprocess the benchmark set with method preprocess
    benchmark_set_X, cat_features = transform_features(benchmark_set, model_type='lgbm', log_transform=False)

    # apply the model to the benchmark set
    benchmark_set['predicted_label'] = model.predict(benchmark_set_X)

    y_benchmark_pred_proba = model.predict_proba(benchmark_set_X)[:, 1]
    y_benchmark_test = benchmark_set['benchmark_label']

    print("Benchmark set AUCROC: ", roc_auc_score(y_benchmark_test, y_benchmark_pred_proba))

    return roc_auc_score(y_benchmark_test, y_benchmark_pred_proba)

def save_model(results, exp_name, range_offaxis):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{exp_name}_lgbm_{range_offaxis}_{timestamp}"
    
    base_path = "models"
    model_path = os.path.join(base_path, model_name)
    os.makedirs(model_path, exist_ok=True)
    
    joblib.dump(results['model'], os.path.join(model_path, "model.joblib"))
    
    # numpy types
    best_params = {k: int(v) if isinstance(v, np.integer) else 
                     float(v) if isinstance(v, np.floating) else v 
                     for k, v in results['best_params'].items()}
    
    with open(os.path.join(model_path, "best_params.json"), 'w') as f:
        json.dump(best_params, f, indent=2)
    
    joblib.dump(results['X_test'], os.path.join(model_path, "X_test.joblib"))
    joblib.dump(results['y_test'], os.path.join(model_path, "y_test.joblib"))
    joblib.dump(results['benchmark_ids'], os.path.join(model_path, "benchmark_ids.joblib"))
    
    print(f"Model and results saved in: {model_path}")
    return model_path

def plot_results(multipliers, metrics_with_int, metrics_without_int, experiment_name):
    plt.figure(figsize=(10, 6))
    plt.plot(multipliers, metrics_with_int, label='With Intermediates', marker='o')
    plt.plot(multipliers, metrics_without_int, label='Without Intermediates', marker='s')
    plt.xlabel('Multiplier (X)')
    plt.ylabel('AUCROC')
    plt.title('AUCROC vs Number of Random Negatives (Benchmark)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{experiment_name}_metric_vs_negatives_benchmark.png')
    plt.close()

def main(experiment_name):
    paths = setup_paths()
    os.makedirs(paths['out_data'], exist_ok=True)
    full_data_path = os.path.join(paths['out_data'], 'nway_csc21_gaia3_full.parquet')
    df_full = pd.read_parquet(full_data_path) if full_data_path.endswith('.parquet') else pd.read_csv(full_data_path)
    print('full data in memory...')

    multipliers = [100]
    metrics_with_int = []
    metrics_without_int = []

    # prepare data
    df_pos, df_neg, benchmark_set = prepare_data(df_full)
    print('data prepared...')

    # train models
    for multiplier in multipliers:
        print(f"training model with multiplier {multiplier}X...")
        metric_with = train_model(f"{experiment_name}_with_int_{multiplier}X", multiplier, True, df_pos, df_neg, benchmark_set)
        metric_without = train_model(f"{experiment_name}_without_int_{multiplier}X", multiplier, False, df_pos, df_neg, benchmark_set)
        
        metrics_with_int.append(metric_with)
        metrics_without_int.append(metric_without)

    plot_results(multipliers, metrics_with_int, metrics_without_int, experiment_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LGBM model with varying negative samples')
    parser.add_argument('--experiment_name', default='negative_sampling_experiment', help='Name of the experiment')
    args = parser.parse_args()
    
    main(args.experiment_name)