#!/usr/bin/env python
# coding: utf-8

import sys
import os
import argparse
import logging

# add parent directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import read_data
from src.utils import (
    preprocess,
    train_and_tune_model,
    run_experiments,
    save_experiment_results,
    load_experiment_results
)
from src.plot import plot_experiment_results, plot_shap_analysis, feature_importance_analysis, lgbm_feature_cluster_and_importance
from src.wandb_utils import plot_subsets_experiment_wandb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_paths():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        'out_data': os.path.join(base_path, 'out_data'),
        'models': os.path.join(base_path, 'models')
    }

def hyperparameter_tuning(df_pos, df_neg, model_type):
    X_train, X_test, Y_train, Y_test, indices_train, indices_test, categorical_features = preprocess(df_pos, df_neg, log_transform=False, model_type=model_type)
    print(categorical_features)
    best_model, y_pred, best_params = train_and_tune_model(X_train, X_test, Y_train, Y_test, categorical_features, model_type=model_type, hyperparameter_tuning=True)
    logging.info(f"Best parameters for {model_type}: {best_params}")

def run_experiment(df_pos, df_neg, model_type, hyperparameter_tuning, log_transform, normalization_method):
    results_exp = run_experiments(
        df_pos, df_neg, 
        model_type=model_type, 
        hyperparameter_tuning=hyperparameter_tuning,
        log_transform=log_transform,
        normalization_method=normalization_method
    )
    experiment_path = save_experiment_results(
        results_exp, 
        model_type=model_type, 
        hyperparameter_tuning=hyperparameter_tuning,
        log_transform=log_transform,
        normalization_method=normalization_method,
        random_seed=42
    )
    logging.info(f"Experiment results saved to: {experiment_path}")

def plot_results(plot_type, model_path, df_pos, df_neg, project_name, range_label):
    output_dir = os.path.dirname(model_path)
    if plot_type == 'shap':
        plot_shap_analysis(model_path, df_pos, df_neg, output_dir)
    elif plot_type == 'lgbm_importance':
        X_train, X_test, Y_train, Y_test, indices_train, indices_test, _, _ = preprocess(df_pos, df_neg, log_transform=False, model_type='lgbm')
        feature_importance_analysis(model_path, X_train, X_test, Y_train, Y_test, output_dir)
    elif plot_type == 'lgbm_importance_cluster':
        X_train, X_test, Y_train, Y_test, indices_train, indices_test, _, _ = preprocess(df_pos, df_neg, log_transform=False, model_type='lgbm')
        lgbm_feature_cluster_and_importance(X_train, X_test, Y_train, Y_test, output_dir)
    elif plot_type == 'experiment':
        results_exp, results_info = load_experiment_results(model_path)
        plot_experiment_results(results_exp, results_info, df_pos, df_neg, project_name, range_label)
    elif plot_type == 'wandb_experiment':
        results_exp, results_info = load_experiment_results(model_path)
        plot_subsets_experiment_wandb(results_exp, results_info, df_pos, df_neg, project_name)
    else:
        logging.error(f"Invalid plot type: {plot_type}")
    logging.info(f"{plot_type.capitalize()} plots created for {project_name}, {range_label}")

def main(args):
    paths = setup_paths()
    os.makedirs(paths['out_data'], exist_ok=True)
    os.makedirs(paths['models'], exist_ok=True)

    separation_thresholds = {
        '0-3': 1.3,  # 0-3 arcmin
        '3-6': 1.3,  # 3-6 arcmin
        '6+': 2.2    # 6+ arcmin
    }

    results = read_data(separation_thresholds)
    df_pos = results[args.range]['df_pos']
    df_neg = results[args.range]['df_neg']

    if args.mode == 'tune':
        hyperparameter_tuning(df_pos, df_neg, args.model)
    elif args.mode == 'experiment':
        run_experiment(
            df_pos, df_neg, 
            args.model, 
            args.hyperparameter_tuning,
            args.log_transform,
            args.normalization_method
        )
    elif args.mode == 'plot':
        plot_results(args.plot_type, args.model_path, df_pos, df_neg, args.project_name, args.range)
    else:
        logging.error("Invalid mode specified")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run classification experiments')
    parser.add_argument('mode', choices=['tune', 'experiment', 'plot'], help='Mode to run the script in')
    parser.add_argument('--range', choices=['0-3', '3-6', '6+'], default='0-3', help='Off-axis range to use')
    parser.add_argument('--model', choices=['rf', 'lgbm'], default='rf', help='Model type to use')
    parser.add_argument('--hyperparameter-tuning', action='store_true', help='Whether to perform hyperparameter tuning')
    parser.add_argument('--log-transform', action='store_true', help='Whether to apply log transformation to applicable features')
    parser.add_argument('--normalization-method', choices=['standard', 'minmax', 'robust', 'quantile', 'power', 'none'], default='none', help='Method for normalizing features')
    parser.add_argument('--plot-type', choices=['shap', 'experiment', 'lgbm_importance', 'lgbm_importance_cluster', 'wandb_experiment'], help='Type of plot to generate (for plot mode)')
    parser.add_argument('--model-path', help='Path to saved model file (for plot mode)')
    parser.add_argument('--project-name', help='Project name for plotting')
   
    args = parser.parse_args()
   
    main(args)