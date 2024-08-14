import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from astropy.io.votable import parse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder

import os
import joblib
import json
from datetime import datetime

# Global variables
FEATURE_NAMES = [
    'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'bp_rp', 'bp_g', 'g_rp', 'parallax', 'hard_hs', 'hard_hm', 
    'hard_ms', 'var_intra_prob_b', 'var_intra_index_b', 'var_inter_prob_b', 'var_inter_index_b', 'var_inter_sigma_b', 
    'extent_flag', 'photflux_aper_b', 'yangetal_gcs_class', 'yangetal_training_class', 'perezdiazetal_class', 'parallax_error', 
    'sqrt(pmra^2+pmdec^2)', 'phot_g_mean_flux', 'phot_bp_mean_flux', 'phot_rp_mean_flux', 'radial_velocity', 'vbroad', 
    'phot_variable_flag', 'classprob_dsc_combmod_quasar', 'classprob_dsc_combmod_galaxy', 'classprob_dsc_combmod_star', 
    'distance_gspphot'
]

LOG_TRANSFORM_FEATURES = [
    'parallax', 'parallax_error', 'photflux_aper_b', 'phot_g_mean_flux', 'phot_bp_mean_flux', 'phot_rp_mean_flux', 'radial_velocity', 'distance_gspphot', 'sqrt(pmra^2+pmdec^2)'
]

def create_new_columns(df):
    # create new features
    if 'sqrt(pmra^2+pmdec^2)' not in df.columns:
        df['sqrt(pmra^2+pmdec^2)'] = np.sqrt(df['pmra']**2 + df['pmdec']**2)
    
    return df

def transform_features(df, log_transform, feature_names=FEATURE_NAMES):
    """
    transform features with optional log transformation and encode categorical features.

    parameters:
    - df: pd.DataFrame, the input dataframe
    - log_transform: bool, whether to apply log transformation or not
    - feature_names: list, optional, list of feature names to use

    returns:
    - transformed_features: np.array, transformed features
    """

    create_new_columns(df)

    # identify categorical features
    categorical_features = ['yangetal_gcs_class', 'yangetal_training_class', 'perezdiazetal_class', 'phot_variable_flag']
    categorical_features = [feature for feature in categorical_features if feature in feature_names]
    
    # encode categorical features
    if categorical_features:
        encoder = OneHotEncoder(sparse_output=False)
        encoded_categorical = encoder.fit_transform(df[categorical_features])
    else:
        encoded_categorical = np.array([]).reshape(len(df), 0)
    
    # handle log transformation and numerical features
    transformed_numerical_features = [
        np.log10(1 + df[feature].values) if log_transform and feature in LOG_TRANSFORM_FEATURES else df[feature].values
        for feature in feature_names if feature not in categorical_features
    ]
    
    # concatenate encoded categorical features with numerical features
    transformed_features = np.concatenate([np.array(transformed_numerical_features).T, encoded_categorical], axis=1)
    
    return transformed_features

def preprocess(df_pos, df_neg, log_transform=True, random_seed=42, test_size=0.3, feature_names=FEATURE_NAMES):
    """
    preprocess dataframes with optional log transformation and split into training and test sets.

    parameters:
    - df_pos: pd.DataFrame, dataframe of positive samples
    - df_neg: pd.DataFrame, dataframe of negative samples
    - log_transform: bool, whether to apply log transformation or not
    - random_seed: int, random seed for reproducibility
    - test_size: float, proportion of the dataset to include in the test split

    returns:
    - X_train: np.array, training features
    - X_test: np.array, test features
    - Y_train: np.array, training labels
    - Y_test: np.array, test labels
    - indices_train: np.array, indices of training samples
    - indices_test: np.array, indices of test samples
    """
    # transform features
    X_pos = transform_features(df_pos, log_transform, feature_names)
    X_neg = transform_features(df_neg, log_transform, feature_names)

    # concatenate X and create Y labels
    X = np.concatenate((X_pos, X_neg), axis=0)
    Y = np.concatenate((np.ones(len(X_pos)), np.zeros(len(X_neg))), axis=0)

    # create indices
    indices = np.arange(X.shape[0])

    # split into training and test sets
    X_train, X_test, Y_train, Y_test, indices_train, indices_test = train_test_split(
        X, Y, indices, test_size=test_size, shuffle=True, random_state=random_seed
    )

    return X_train, X_test, Y_train, Y_test, indices_train, indices_test


def handle_missing_values(X_train, X_test):
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    X_train = imp_mean.fit_transform(X_train)
    X_test = imp_mean.transform(X_test)
    return X_train, X_test, imp_mean

def standardize(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def votable_to_pandas(votable_file):
    '''
    Converts votable to pandas dataframe.
    '''
    votable = parse(votable_file)
    table = votable.get_first_table().to_table(use_names_over_ids=True)
    return table.to_pandas()

def train_and_tune_model(X_train, X_test, Y_train, Y_test, model_type='rf', hyperparameter_tuning=True, random_seed=42):
    """
    train and tune a model based on the given splits and model type.

    parameters:
    - X_train: np.array, training features
    - X_test: np.array, test features
    - Y_train: np.array, training labels
    - Y_test: np.array, test labels
    - model_type: str, either "rf" for Random Forest or "lgbm" for LightGBM
    - hyperparameter_tuning: bool, whether to perform hyperparameter tuning or not
    - random_seed: int, random seed for reproducibility

    returns:
    - best_model: trained model
    - y_pred: np.array, predictions on the test set
    - best_params: dict, best hyperparameters if tuning is performed, else None
    """
    if model_type == 'rf':
        model = RandomForestClassifier(random_state=random_seed)
        param_grid = {
            'n_estimators': [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
            'max_depth': [None] + [int(x) for x in np.linspace(10, 110, num=11)],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        default_params = {
            'n_estimators': 800,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 1
        }
    elif model_type == 'lgbm':
        # placeholder
        model = None
        param_grid = {
            'num_leaves': [31, 50, 100],
            'max_depth': [-1, 10, 20, 30],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [100, 200, 300],
            'min_child_samples': [20, 30, 40]
        }
        default_params = {
            'num_leaves': 31,
            'max_depth': -1,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'min_child_samples': 20
        }
        print("LightGBM model is not implemented yet.")
        return None, None, None
    else:
        raise ValueError("Unsupported model type. Please choose 'rf' or 'lgbm'.")

    if hyperparameter_tuning:
        print("Performing hyperparameter tuning...")
        search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=10, cv=5, n_jobs=-1, verbose=2, scoring='accuracy', random_state=random_seed)
        search.fit(X_train, Y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_
        print(f"Best parameters found: {best_params}")
    else:
        print("Training the model without hyperparameter tuning...")
        model.set_params(**default_params)
        best_model = model
        best_model.fit(X_train, Y_train)
        best_params = default_params
    # preds
    y_pred = best_model.predict(X_test)

    # performance
    print("Model performance on the test set:")
    print(f"Accuracy: {accuracy_score(Y_test, y_pred)}")
    print(classification_report(Y_test, y_pred))

    return best_model, y_pred, best_params

def prepare_feature_subset(df, subset_type):
    """
    prepare the subset of features based on the given subset type.

    parameters:
    - df: pd.DataFrame, the input dataframe
    - subset_type: int, the type of feature subset to prepare

    returns:
    - df_subset: pd.DataFrame, the subset dataframe
    """
    base_features = ['bp_rp', 'bp_g', 'g_rp', 'hard_hs', 'hard_hm', 'hard_ms', 'parallax', 'parallax_error', 'sqrt(pmra^2+pmdec^2)', 'distance_gspphot']
    magnitudes = ['phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag']
    fluxes = ['phot_g_mean_flux', 'phot_bp_mean_flux', 'phot_rp_mean_flux', 'radial_velocity', 'vbroad']
    variability = ['var_intra_prob_b', 'var_inter_prob_b', 'var_inter_sigma_b']
    categories = ['var_intra_index_b', 'var_inter_index_b', 'phot_variable_flag', 'yangetal_gcs_class', 'yangetal_training_class', 'perezdiazetal_class']

    if subset_type == 1:
        selected_features = base_features
    elif subset_type == 2:
        selected_features = base_features + magnitudes
    elif subset_type == 3:
        selected_features = base_features + magnitudes + fluxes + variability
    elif subset_type == 4:
        selected_features = base_features
        # filter for NaN in classification categories
        df = df[df[['yangetal_gcs_class', 'yangetal_training_class', 'perezdiazetal_class']].isna().any(axis=1)]
    elif subset_type == 5:
        selected_features = base_features + magnitudes + fluxes + variability
        # filter for NaN in classification categories
        df = df[df[['yangetal_gcs_class', 'yangetal_training_class', 'perezdiazetal_class']].isna().any(axis=1)]
    elif subset_type == 6:
        selected_features = base_features + categories
    elif subset_type == 7:
        selected_features = base_features + magnitudes + categories
    elif subset_type == 8:
        selected_features = base_features + magnitudes + fluxes + variability + categories
    else:
        raise ValueError("Unsupported subset type.")

    return df[selected_features]


def run_experiments(df_pos, df_neg, model_type='rf', hyperparameter_tuning=False, random_seed=42):
    """
    run experiments for different subsets of features.

    parameters: 
    - df_pos: pd.DataFrame, positive samples
    - df_neg: pd.DataFrame, negative samples
    - model_type: str, either "rf" for Random Forest or "lgbm" for LightGBM
    - hyperparameter_tuning: bool, whether to perform hyperparameter tuning or not
    - random_seed: int, random seed for reproducibility

    returns:
    - results: dict, containing results for all subsets
    """
    results = {}

    for subset_type in range(6, 9):
        print(f"Running subset type {subset_type}...")

        create_new_columns(df_pos)
        create_new_columns(df_neg)
        
        # feature subset
        df_pos_subset = prepare_feature_subset(df_pos, subset_type)
        df_neg_subset = prepare_feature_subset(df_neg, subset_type)

        # preprocess
        X_train, X_test, Y_train, Y_test, indices_train, indices_test = preprocess(df_pos_subset, df_neg_subset, log_transform=False, random_seed=random_seed, feature_names=df_pos_subset.columns)

        # train
        best_model, y_pred, best_params = train_and_tune_model(X_train, X_test, Y_train, Y_test, model_type=model_type, hyperparameter_tuning=hyperparameter_tuning, random_seed=random_seed)

        results[subset_type] = {
            'best_model': best_model,
            'best_params': best_params,
            'y_pred': y_pred,
            'indices_test': indices_test
        }

    return results

def save_experiment_results(results, model_type, hyperparameter_tuning, random_seed):
    """
    save all info returned by run_experiments in a new folder

    parameters:
    - results: dict, containing results for all subsets
    - model_type: str, either "rf" for Random Forest or "lgbm" for LightGBM
    - hyperparameter_tuning: bool, whether hyperparameter tuning was performed
    - random_seed: int, random seed used for reproducibility

    returns:
    - experiment_path: str, path to the saved experiment folder
    """
    # create experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tuning_str = "tuned" if hyperparameter_tuning else "default"
    experiment_name = f"{model_type}_{tuning_str}_seed{random_seed}_{timestamp}"

    # create folder
    base_path = "models"
    experiment_path = os.path.join(base_path, experiment_name)
    os.makedirs(experiment_path, exist_ok=True)

    # save results for each subset
    for subset_type, subset_results in results.items():
        subset_path = os.path.join(experiment_path, f"subset_{subset_type}")
        os.makedirs(subset_path, exist_ok=True)

        # save model
        model_path = os.path.join(subset_path, "model.joblib")
        joblib.dump(subset_results['best_model'], model_path)

        # save best parameters
        params_path = os.path.join(subset_path, "best_params.json")
        with open(params_path, 'w') as f:
            json.dump(subset_results['best_params'], f, indent=2)

        # save predictions
        preds_path = os.path.join(subset_path, "predictions.joblib")
        joblib.dump(subset_results['y_pred'], preds_path)

        # save test indices
        indices_path = os.path.join(subset_path, "test_indices.joblib")
        joblib.dump(subset_results['indices_test'], indices_path)

    print(f"experiment results saved in: {experiment_path}")
    return experiment_path

def load_experiment_results(experiment_path):
    """
    load results saved by save_experiment_results

    parameters:
    - experiment_path: str, path to the saved experiment folder

    returns:
    - results: dict, containing loaded results for all subsets
    - experiment_info: dict, containing experiment metadata
    """
    results = {}
    
    # extract experiment info from path
    experiment_name = os.path.basename(experiment_path)
    model_type, tuning_str, seed_info, timestamp = experiment_name.split('_', 3)
    hyperparameter_tuning = tuning_str == "tuned"
    random_seed = int(seed_info[4:])  # remove "seed" prefix
    
    experiment_info = {
        'model_type': model_type,
        'hyperparameter_tuning': hyperparameter_tuning,
        'random_seed': random_seed,
        'timestamp': timestamp
    }

    # load results for each subset
    for subset_folder in os.listdir(experiment_path):
        if subset_folder.startswith("subset_"):
            subset_type = int(subset_folder.split("_")[1])
            subset_path = os.path.join(experiment_path, subset_folder)
            
            # load model
            model_path = os.path.join(subset_path, "model.joblib")
            best_model = joblib.load(model_path)
            
            # load best parameters
            params_path = os.path.join(subset_path, "best_params.json")
            with open(params_path, 'r') as f:
                best_params = json.load(f)
            
            # load predictions
            preds_path = os.path.join(subset_path, "predictions.joblib")
            y_pred = joblib.load(preds_path)
            
            # load test indices
            indices_path = os.path.join(subset_path, "test_indices.joblib")
            indices_test = joblib.load(indices_path)
            
            results[subset_type] = {
                'best_model': best_model,
                'best_params': best_params,
                'y_pred': y_pred,
                'indices_test': indices_test
            }
    
    print(f"experiment results loaded from: {experiment_path}")
    return results, experiment_info