import argparse
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data import read_data
from src.utils import preprocess, train_and_tune_model
import os
import joblib
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(range_offaxis, experiment_name):
    # define separation thresholds
    separation_thresholds = {'0-3': 1.3, '3-6': 1.3, '6+': 2.2}
    
    # read data
    results = read_data(separation_thresholds, folder='full_negatives')
    df_pos = results[range_offaxis]['df_pos']
    df_neg = results[range_offaxis]['df_neg']
    
    # preprocess data
    X_train, X_test, y_train, y_test, _, _, cat_features, _ = preprocess(df_pos, df_neg, model_type='lgbm')
    
    # train model
    model, y_pred, best_params = train_and_tune_model(
        X_train, X_test, y_train, y_test, cat_features, model_type='lgbm', hyperparameter_tuning=False
    )
    
    # save results
    results_exp = {
        'model': model,
        'y_pred': y_pred,
        'best_params': best_params,
        'X_test': X_test,
        'y_test': y_test
    }
    model_path = save_model(results_exp, experiment_name, range_offaxis)
    logging.info(f"Model saved to: {model_path}")

def save_model(results, exp_name, range_offaxis):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{exp_name}_lgbm_{range_offaxis}_{timestamp}"
    
    base_path = "models"
    model_path = os.path.join(base_path, model_name)
    os.makedirs(model_path, exist_ok=True)
    
    # save model
    joblib.dump(results['model'], os.path.join(model_path, "model.joblib"))
    
    # save predictions
    joblib.dump(results['y_pred'], os.path.join(model_path, "predictions.joblib"))
    
    # save best parameters
    with open(os.path.join(model_path, "best_params.json"), 'w') as f:
        json.dump(results['best_params'], f, indent=2)
    
    # save test data
    joblib.dump(results['X_test'], os.path.join(model_path, "X_test.joblib"))
    joblib.dump(results['y_test'], os.path.join(model_path, "y_test.joblib"))
    
    print(f"Model and results saved in: {model_path}")
    return model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LGBM model')
    parser.add_argument('range', choices=['0-3', '3-6', '6+'], help='Off-axis range to use')
    parser.add_argument('--experiment_name', default='experiment', help='Name of the experiment')
    args = parser.parse_args()
    
    main(args.range, args.experiment_name)