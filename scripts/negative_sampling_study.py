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
from sklearn.metrics import balanced_accuracy_score

sys.path.append(str(Path(__file__).parent.parent))
from src.data import read_data
from src.utils import preprocess, train_and_tune_model
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def downsample_negatives(df_pos, df_neg, intermediates=True, multiplier=100):
    # keep clear negatives
    clear_negatives = df_neg[df_neg['negative_type'] == 'clear_negative']
    
    # filter remaining negatives based on intermediates flag
    other_negatives = df_neg[df_neg['negative_type'] != 'clear_negative']
    if not intermediates:
        other_negatives = other_negatives[other_negatives['negative_type'] == 'random']
    
    # calculate number of negatives to sample
    n_pos = df_pos.shape[0]
    n_neg_sample = n_pos * multiplier
    
    # sample from other negatives if necessary
    if len(other_negatives) > n_neg_sample:
        sampled_negatives = other_negatives.sample(n=n_neg_sample, random_state=42)
    else:
        sampled_negatives = other_negatives
    
    # combine clear negatives with sampled negatives
    return pd.concat([clear_negatives, sampled_negatives])

def train_model(experiment_name, multiplier, intermediates):
    separation_thresholds = {'0-3': 1.3, '3-6': 1.3, '6+': 2.2}
    range_offaxis = '0-3'  # as per comment, we're assuming range 0-3

    results = read_data(separation_thresholds, folder='full_negatives', suffix='')
    df_pos = results[range_offaxis]['df_pos']
    df_neg = results[range_offaxis]['df_neg']

    df_neg = downsample_negatives(df_pos, df_neg, intermediates, multiplier)
    X_train, X_test, y_train, y_test, _, _, cat_features, _ = preprocess(df_pos, df_neg, model_type='lgbm')

    model, y_pred, best_params = train_and_tune_model(
        X_train, X_test, y_train, y_test, cat_features, model_type='lgbm', hyperparameter_tuning=False
    )

    results_exp = {
        'model': model,
        'y_pred': y_pred,
        'best_params': best_params,
        'X_test': X_test,
        'y_test': y_test
    }
    model_path = save_model(results_exp, experiment_name, range_offaxis)
    logging.info(f"Model saved to: {model_path}")

    return balanced_accuracy_score(y_test, y_pred)

def save_model(results, exp_name, range_offaxis):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{exp_name}_lgbm_{range_offaxis}_{timestamp}"
    
    base_path = "models"
    model_path = os.path.join(base_path, model_name)
    os.makedirs(model_path, exist_ok=True)
    
    joblib.dump(results['model'], os.path.join(model_path, "model.joblib"))
    joblib.dump(results['y_pred'], os.path.join(model_path, "predictions.joblib"))
    
    with open(os.path.join(model_path, "best_params.json"), 'w') as f:
        json.dump(results['best_params'], f, indent=2)
    
    joblib.dump(results['X_test'], os.path.join(model_path, "X_test.joblib"))
    joblib.dump(results['y_test'], os.path.join(model_path, "y_test.joblib"))
    
    print(f"Model and results saved in: {model_path}")
    return model_path

def plot_results(multipliers, accuracies_with_int, accuracies_without_int):
    plt.figure(figsize=(10, 6))
    plt.plot(multipliers, accuracies_with_int, label='With Intermediates', marker='o')
    plt.plot(multipliers, accuracies_without_int, label='Without Intermediates', marker='s')
    plt.xlabel('Multiplier (X)')
    plt.ylabel('Balanced Accuracy')
    plt.title('Balanced Accuracy vs Number of Random Negatives')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_vs_negatives.png')
    plt.close()

def main(experiment_name):
    multipliers = [0, 20, 40, 60, 80, 100]
    accuracies_with_int = []
    accuracies_without_int = []

    for multiplier in multipliers:
        acc_with = train_model(f"{experiment_name}_with_int_{multiplier}X", multiplier, True)
        acc_without = train_model(f"{experiment_name}_without_int_{multiplier}X", multiplier, False)
        
        accuracies_with_int.append(acc_with)
        accuracies_without_int.append(acc_without)

    plot_results(multipliers, accuracies_with_int, accuracies_without_int)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LGBM model with varying negative samples')
    parser.add_argument('--experiment_name', default='negative_sampling_experiment', help='Name of the experiment')
    args = parser.parse_args()
    
    main(args.experiment_name)