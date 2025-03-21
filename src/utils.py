from tkinter import _test
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer
from astropy.io.votable import parse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb

import os
import joblib
import json
from datetime import datetime

import pandas as pd
from scipy.stats import loguniform, uniform

from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, rand
from sklearn.model_selection import cross_val_score
from functools import partial

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
        df.loc[:, 'sqrt(pmra^2+pmdec^2)'] = np.sqrt(df['pmra']**2 + df['pmdec']**2)
    
    return df

def transform_features(df, log_transform, model_type='rf', feature_names=FEATURE_NAMES):
    """
    transform features with optional log transformation and handle categorical features.

    parameters:
    - df: pd.DataFrame, the input dataframe
    - log_transform: bool, whether to apply log transformation or not
    - model_type: str, 'rf' for Random Forest or 'lgbm' for LightGBM
    - feature_names: list, optional, list of feature names to use

    returns:
    - transformed_features: np.array or pd.DataFrame, transformed features
    - categorical_features: list, names of categorical features (for LightGBM)
    """
    create_new_columns(df)
    # identify categorical features
    categorical_features = ['yangetal_gcs_class', 'yangetal_training_class', 'perezdiazetal_class', 'phot_variable_flag']
    categorical_features = [feature for feature in categorical_features if feature in feature_names]
   
    # handle categorical features
    if model_type == 'rf':
        # one-hot encode for Random Forest
        if categorical_features:
            encoder = OneHotEncoder(sparse_output=False)
            encoded_categorical = encoder.fit_transform(df[categorical_features])
            encoded_feature_names = encoder.get_feature_names_out(categorical_features)
        else:
            encoded_categorical = np.array([]).reshape(len(df), 0)
            encoded_feature_names = []
    elif model_type == 'lgbm':
        # for lgbm, we'll keep categorical features as they are
        encoded_categorical = df[categorical_features].values
        encoded_feature_names = categorical_features
    else:
        raise ValueError("Unsupported model type. Please choose 'rf' or 'lgbm'.")
   
    # handle log transformation and numerical features
    numerical_features = [f for f in feature_names if f not in categorical_features]
    transformed_numerical_features = [
        np.log10(1 + df[feature].values) if log_transform and feature in LOG_TRANSFORM_FEATURES else df[feature].values
        for feature in numerical_features
    ]
   
    # combine numerical and categorical features
    if model_type == 'rf':
        transformed_features = np.concatenate([np.array(transformed_numerical_features).T, encoded_categorical], axis=1)
        all_feature_names = numerical_features + list(encoded_feature_names)
        return pd.DataFrame(transformed_features, columns=all_feature_names), []
    elif model_type == 'lgbm':
        numerical_df = pd.DataFrame(np.array(transformed_numerical_features).T, columns=numerical_features)
        categorical_df = pd.DataFrame(encoded_categorical, columns=encoded_feature_names)
        for c in encoded_feature_names:
            categorical_df[c] = categorical_df[c].astype('category')
        return pd.concat([numerical_df, categorical_df], axis=1), categorical_features

def preprocess(df_pos, df_neg,  normalization_method='none', log_transform=False, model_type='rf', random_seed=42, test_size=0.3, feature_names=FEATURE_NAMES):
    """
    preprocess dataframes with optional log transformation and split into training and test sets.
    parameters:
    - df_pos: pd.DataFrame, dataframe of positive samples
    - df_neg: pd.DataFrame, dataframe of negative samples
    - log_transform: bool, whether to apply log transformation or not
    - model_type: str, 'rf' for Random Forest or 'lgbm' for LightGBM
    - random_seed: int, random seed for reproducibility
    - test_size: float, proportion of the dataset to include in the test split
    returns:
    - X_train: pd.DataFrame, training features
    - X_test: pd.DataFrame, test features
    - Y_train: np.array, training labels
    - Y_test: np.array, test labels
    - indices_train: np.array, indices of training samples
    - indices_test: np.array, indices of test samples
    - categorical_features: list, names of categorical features (for LightGBM)
    """

    # transform features
    X_pos, cat_features_pos = transform_features(df_pos, log_transform, model_type, feature_names)
    X_neg, cat_features_neg = transform_features(df_neg, log_transform, model_type, feature_names)
    
    # ensure categorical features are consistent
    assert cat_features_pos == cat_features_neg, "Categorical features mismatch between positive and negative samples"
    categorical_features = cat_features_pos
    
    # concat X and create Y labels
    X = pd.concat([X_pos, X_neg], axis=0, ignore_index=True)
    Y = np.concatenate((np.ones(len(X_pos)), np.zeros(len(X_neg))), axis=0)
    
    # create indices
    indices = np.arange(X.shape[0])
    
    # split into training and test sets
    X_train, X_test, Y_train, Y_test, indices_train, indices_test = train_test_split(
        X, Y, indices, test_size=test_size, shuffle=True, random_state=random_seed
    )
    
    X_train_norm, X_test_norm, scaler = normalize_train_test(X_train, X_test, method=normalization_method, categorical_features=categorical_features)
    
    return X_train_norm, X_test_norm, Y_train, Y_test, indices_train, indices_test, categorical_features, scaler

def preprocess_cscid(df_pos, df_neg, df_full, normalization_method='none', log_transform=False, 
                    model_type='rf', random_seed=42, eval_size=0.3, feature_names=FEATURE_NAMES):
    # split cscids into train and eval sets
    cscids = df_pos.csc21_name.unique()
    cscids_train, cscids_eval = train_test_split(cscids, test_size=eval_size, random_state=random_seed)

    # create eval set from full data (similar to benchmark)
    eval_set = df_full[df_full['csc21_name'].isin(cscids_eval)].copy()
    eval_set.loc[:, 'eval_label'] = np.where(eval_set['match_flag'] == 1, 1, 0)

    # create train dataframes
    df_pos_train = df_pos[df_pos.csc21_name.isin(cscids_train)].copy()
    df_neg_train = df_neg[df_neg.csc21_name.isin(cscids_train)].copy()

    # Check that the intersection between Chandra and Gaia IDs in the train and validation set is void
    chandra_ids_train = set(df_pos_train['csc21_name']).union(set(df_neg_train['csc21_name']))
    chandra_ids_eval = set(eval_set['csc21_name'])
    gaia_ids_train = set(df_pos_train['gaia3_source_id']).union(set(df_neg_train['gaia3_source_id']))
    gaia_ids_eval = set(eval_set['gaia3_source_id'])

    chandra_overlap = chandra_ids_train.intersection(chandra_ids_eval)
    gaia_overlap = gaia_ids_train.intersection(gaia_ids_eval)
    
    if chandra_overlap:
        print(f"Chandra IDs overlap count between train and validation sets: {len(chandra_overlap)}")
    if gaia_overlap:
        print(f"Gaia IDs overlap count between train and validation sets: {len(gaia_overlap)}")

    # transform features for training data
    X_pos_train, cat_features = transform_features(df_pos_train, log_transform, model_type, feature_names)
    X_neg_train, _ = transform_features(df_neg_train, log_transform, model_type, feature_names)

    # transform features for eval set
    eval_set_X, _ = transform_features(eval_set, log_transform, model_type, feature_names)

    # concat training data
    X_train = pd.concat([X_pos_train, X_neg_train], axis=0, ignore_index=True)
    Y_train = np.concatenate((np.ones(len(X_pos_train)), np.zeros(len(X_neg_train))))

    # normalize
    X_train_norm, eval_set_X_norm, scaler = normalize_train_test(X_train, eval_set_X, 
                                                                method=normalization_method,
                                                                categorical_features=cat_features)

    return (X_train_norm, eval_set_X_norm, Y_train, eval_set['eval_label'].values, 
            cscids_train, cscids_eval, cat_features, scaler, eval_set)

def handle_missing_values(X_train, X_test):
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    X_train = imp_mean.fit_transform(X_train)
    X_test = imp_mean.transform(X_test)
    return X_train, X_test, imp_mean

def normalize_train_test(X_train, X_test, method='standard', categorical_features=[]):
    numerical_features = [col for col in X_train.columns if col not in categorical_features]
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'quantile':
        scaler = QuantileTransformer(output_distribution='normal')
    elif method == 'power':
        scaler = PowerTransformer(method='yeo-johnson')
    elif method == 'none':
        return X_train, X_test, None
    else:
        raise ValueError("Unsupported normalization method")
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])
    
    return X_train_scaled, X_test_scaled, scaler


def votable_to_pandas(votable_file):
    '''
    Converts votable to pandas dataframe.
    '''
    votable = parse(votable_file)
    table = votable.get_first_table().to_table(use_names_over_ids=True)
    return table.to_pandas()

def train_and_tune_model(X_train, X_test, Y_train, Y_test, categorical_features=None, model_type='rf', hyperparameter_tuning=True, random_seed=42):
    if model_type == 'rf':
        model, param_grid, default_params = setup_random_forest(random_seed)
        train_data = X_train
    elif model_type == 'lgbm':
        model, param_grid, default_params, train_data, test_data = setup_lightgbm(X_train, X_test, Y_train, Y_test, categorical_features, random_seed)
    else:
        raise ValueError("Unsupported model type. Please choose 'rf' or 'lgbm'.")

    if hyperparameter_tuning:
        best_model, best_params = perform_hyperparameter_tuning_sklearn(model, param_grid, train_data, Y_train, test_data, random_seed, model_type)
    else:
        best_model, best_params = train_with_default_params(model, default_params, train_data, Y_train, test_data, model_type)

    y_pred = best_model.predict(X_test)
    y_train_pred = best_model.predict(X_train)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    y_train_pred_proba = best_model.predict_proba(X_train)[:, 1]
    print_performance_metrics(Y_train, y_train_pred, y_train_pred_proba, Y_test, y_pred, y_pred_proba)

    return best_model, y_pred, best_params

def setup_random_forest(random_seed):
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
    return model, param_grid, default_params

def setup_lightgbm(X_train, X_test, Y_train, Y_test, categorical_features, random_seed):
    if categorical_features:
        X_train = X_train.copy()
        X_test = X_test.copy()
        for c in categorical_features:
            X_train[c] = X_train[c].astype('category')
            X_test[c] = X_test[c].astype('category')
   
    train_data = lgb.Dataset(X_train, label=Y_train, categorical_feature=categorical_features)
    test_data = lgb.Dataset(X_test, label=Y_test, categorical_feature=categorical_features)
    model = lgb.LGBMClassifier(
        random_state=random_seed,
        is_unbalance=True,
        first_metric_only=True,
        force_row_wise=True,
        n_estimators = 500,
        n_jobs=4,
        verbose=-1
    )

    param_space = {
        'learning_rate': hp.loguniform('learning_rate', -7, 0),
        'num_leaves' : hp.qloguniform('num_leaves', 0, 7, 1),
        'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
        'min_data_in_leaf': hp.qloguniform('min_data_in_leaf', 0, 6, 1),
        'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', -16, 5),
        'lambda_l1': hp.choice('lambda_l1', [0, hp.loguniform('lambda_l1_positive', -16, 2)]),
        'lambda_l2': hp.choice('lambda_l2', [0, hp.loguniform('lambda_l2_positive', -16, 2)])
    }
    
    default_params = {
        'num_leaves': 127,
        'max_depth': 15,
        'learning_rate': 0.01,
        'n_estimators': 1000,
        'min_child_samples': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'force_row_wise': True 
    }
    return model, param_space, default_params, train_data, test_data

def objective(params, model, train_data, test_data, random_seed):
    params['num_leaves'] = max(int(params['num_leaves']), 2)
    params['min_data_in_leaf'] = int(params['min_data_in_leaf'])
    print(params)
    
    # update model params
    model.set_params(**params)
    
    # cross validation with roc_auc scoring
    score = cross_val_score(
        model, 
        train_data.data, 
        train_data.label,
        cv=5,
        scoring='roc_auc'
    ).mean()
    
    return {'loss': -score, 'status': STATUS_OK}

def perform_hyperparameter_tuning(model, param_space, train_data, Y_train, test_data, random_seed, model_type):
    if model_type == 'lgbm':
        trials = Trials()
        objective_with_data = partial(objective, model=model, train_data=train_data, test_data=test_data, random_seed=random_seed)
        
        best = fmin(
            fn=objective_with_data,
            space=param_space,
            algo=tpe.suggest, #rand.suggest
            max_evals=20,
            trials=trials,
            rstate=np.random.default_rng(random_seed)
        )

        
        # Just convert numeric params that need to be integers
        best_params = {k: (int(v) if k in ['num_leaves', 'min_data_in_leaf'] else v) 
                      for k, v in best.items()}
        
        best_params['n_estimators'] = 1000
        
        best_model = model.set_params(**best_params)
        best_model.fit(
            train_data.data, 
            train_data.label, 
            eval_set=[(test_data.data, test_data.label)], 
            eval_metric='auc', 
            early_stopping_rounds=10, 
            verbose=20)
    else:
        from sklearn.model_selection import RandomizedSearchCV
        search = RandomizedSearchCV(estimator=model, param_distributions=param_space, n_iter=20, cv=5, verbose=2, scoring='roc_auc', random_state=random_seed)
        search.fit(train_data, Y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_
    
    print(f"Best parameters found: {best_params}")
    return best_model, best_params

class QLogUniform:
    def __init__(self, low, high, q, is_int=False):
        self.low = low
        self.high = high
        self.q = q
        self.is_int = is_int

    def rvs(self, random_state=None):
        rng = random_state if random_state else np.random
        sample = np.exp(rng.uniform(self.low, self.high))
        result = np.round(sample / self.q) * self.q
        result = max(result, 2)
        return int(result) if self.is_int else result

class Choice:
    def __init__(self, options):
        self.options = options

    def rvs(self, random_state=None):  # Accept random_state argument
        rng = random_state if random_state else np.random
        choice = rng.choice(len(self.options))
        option = self.options[choice]
        return option.rvs() if hasattr(option, 'rvs') else option

        
def perform_hyperparameter_tuning_sklearn(model, param_space, train_data, Y_train, test_data, random_seed, model_type):

    param_space = {
        'learning_rate': loguniform(np.exp(-7), 1),  # scipy's loguniform
        'num_leaves': QLogUniform(0, 7, q=1, is_int=True),
        'feature_fraction': uniform(0.5, 0.5),
        'bagging_fraction': uniform(0.5, 0.5),
        'min_data_in_leaf': QLogUniform(0, 6, q=1, is_int=True),
        'min_sum_hessian_in_leaf': loguniform(np.exp(-16), np.exp(5)),
        'lambda_l1': Choice([0, loguniform(np.exp(-16), np.exp(2))]),
        'lambda_l2': Choice([0, loguniform(np.exp(-16), np.exp(2))])
    }

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_space,
        n_iter=500,
        scoring='roc_auc',
        cv=5,
        random_state=random_seed,
        verbose=0,
        n_jobs=8
    )

    # fit
    search.fit(train_data.data, train_data.label)

    best_model = search.best_estimator_
    best_params = search.best_params_

    best_params['n_estimators'] = 5000

    best_model = model.set_params(**best_params, early_stopping_round=10)

    best_model.fit(
    train_data.data, 
    train_data.label, 
    eval_set=[(test_data.data, test_data.label)], 
    eval_metric='auc')

    print(f"Best parameters found: {best_params}")

    return best_model, best_params


def train_with_default_params(model, default_params, train_data, Y_train, test_data, model_type):
    print("Training the model without hyperparameter tuning...")
    model.set_params(**default_params)
    if model_type == 'lgbm':
        model.fit(X=train_data.data, 
                  y=train_data.label, 
                  eval_set=[(test_data.data, test_data.label)], 
                  eval_metric='auc', early_stopping_rounds=10,
                  verbose=20)
    else:
        model.fit(train_data, Y_train)
    return model, default_params

def print_performance_metrics(Y_train, Y_train_pred, y_train_pred_proba, Y_test, y_pred, y_pred_proba):
    print("Model performance on the train set:")
    print(f"ROC AUC Train: {roc_auc_score(Y_train, y_train_pred_proba)}")
    print("Model performance on the test set:")
    print(f"ROC AUC Eval: {roc_auc_score(Y_test, y_pred_proba)}")
    print(classification_report(Y_test, y_pred))

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

    # additional features
    additional_features = [
        'extent_flag', 'photflux_aper_b', 
        'classprob_dsc_combmod_quasar', 'classprob_dsc_combmod_galaxy', 'classprob_dsc_combmod_star'
    ]

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
    elif subset_type == 9:
        # verify
        all_subset_features = base_features + magnitudes + fluxes + variability + categories + additional_features
        assert set(FEATURE_NAMES) == set(all_subset_features), "Some features are missing or extra features are present"
        selected_features = all_subset_features
    elif subset_type == 10:
        # verify
        all_subset_features = base_features + magnitudes + fluxes + variability + categories + additional_features
        assert set(FEATURE_NAMES) == set(all_subset_features), "Some features are missing or extra features are present"
        features_to_remove =  ['vbroad', 'radial_velocity', 'extent_flag', 'distance_gspphot', 'perezdiazetal_class', 'var_inter_index_b', 'var_intra_index_b',  'yangetal_training_class']
        selected_features = [feature for feature in all_subset_features if feature not in features_to_remove]
    elif subset_type == 11:
        all_subset_features = base_features + magnitudes + fluxes + variability + categories + additional_features
        assert set(FEATURE_NAMES) == set(all_subset_features), "Some features are missing or extra features are present"
        features_to_remove =  ['vbroad', 'radial_velocity', 'extent_flag', 'var_inter_index_b', 'var_intra_index_b', 'classprob_dsc_combmod_quasar', 'classprob_dsc_combmod_galaxy', 'classprob_dsc_combmod_star', 'distance_gspphot', 'perezdiazetal_class', 'yangetal_gcs_class',  'yangetal_training_class']
        selected_features = [feature for feature in all_subset_features if feature not in features_to_remove] 
    else:
        raise ValueError("Unsupported subset type.")
    
    return df[selected_features]


def run_experiments(df_pos, df_neg, model_type='rf', hyperparameter_tuning=False, log_transform=False, normalization_method='none', random_seed=42):
    """
    run experiments for different subsets of features.

    parameters: 
    - df_pos: pd.DataFrame, positive samples
    - df_neg: pd.DataFrame, negative samples
    - model_type: str, either "rf" for Random Forest or "lgbm" for LightGBM
    - hyperparameter_tuning: bool, whether to perform hyperparameter tuning or not
    - log_transform: bool, whether to apply log transformation to applicable features
    - normalization_method: str, method for normalizing features ('standard', 'minmax', 'robust', 'quantile', or 'power')
    - random_seed: int, random seed for reproducibility

    returns:
    - results: dict, containing results for all subsets
    """
    results = {}

    for subset_type in range(11, 12):
        print(f"Running subset type {subset_type}...")

        create_new_columns(df_pos)
        create_new_columns(df_neg)
        
        # feature subset
        df_pos_subset = prepare_feature_subset(df_pos, subset_type)
        df_neg_subset = prepare_feature_subset(df_neg, subset_type)

        # preprocess
        X_train, X_test, Y_train, Y_test, indices_train, indices_test, categorical_features, scaler = preprocess(
            df_pos_subset, df_neg_subset, 
            log_transform=log_transform, 
            normalization_method=normalization_method, 
            model_type=model_type, 
            random_seed=random_seed, 
            feature_names=df_pos_subset.columns
        )

        # train
        best_model, y_pred, best_params = train_and_tune_model(
            X_train, X_test, Y_train, Y_test, 
            categorical_features, 
            model_type=model_type, 
            hyperparameter_tuning=hyperparameter_tuning, 
            random_seed=random_seed
        )

        results[subset_type] = {
            'best_model': best_model,
            'best_params': best_params,
            'y_pred': y_pred,
            'indices_test': indices_test,
            'scaler': scaler
        }

    return results

def save_experiment_results(results, exp_name, model_type, hyperparameter_tuning, log_transform, normalization_method, random_seed):
    """
    save all info returned by run_experiments in a new folder
    parameters:
    - results: dict, containing results for all subsets
    - model_type: str, either "rf" for Random Forest or "lgbm" for LightGBM
    - hyperparameter_tuning: bool, whether hyperparameter tuning was performed
    - log_transform: bool, whether log transformation was applied
    - normalization_method: str, method used for normalizing features
    - random_seed: int, random seed used for reproducibility
    returns:
    - experiment_path: str, path to the saved experiment folder
    """
    # create experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tuning_str = "tuned" if hyperparameter_tuning else "default"
    log_str = "log" if log_transform else "nolog"
    experiment_name = f"{exp_name}_{model_type}_{tuning_str}_{log_str}_{normalization_method}_seed{random_seed}_{timestamp}"

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

        # save scaler
        scaler_path = os.path.join(subset_path, "scaler.joblib")
        joblib.dump(subset_results['scaler'], scaler_path)

    # save experiment configuration
    config = {
        'model_type': model_type,
        'hyperparameter_tuning': hyperparameter_tuning,
        'log_transform': log_transform,
        'normalization_method': normalization_method,
        'random_seed': random_seed
    }
    config_path = os.path.join(experiment_path, "experiment_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Experiment results saved in: {experiment_path}")
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
    model_type, tuning_str, log_str, norm_method, seed_info, timestamp = experiment_name.split('_', 5)
    hyperparameter_tuning = tuning_str == "tuned"
    
    experiment_info = {
        'model_type': model_type,
        'hyperparameter_tuning': hyperparameter_tuning,
        'log_transform': log_str == "log",
        'normalization_method': norm_method,
        'random_seed': int(seed_info[4:]),  # remove "seed" prefix
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

def load_model(model_path):
    """
    load a single model and its associated information.

    params:
    - model_path: str, path to the saved model file (.joblib)

    returns:
    - model_info: dict, containing the loaded model and associated information
    """
    model_dir = os.path.dirname(model_path)
    subset_dir = os.path.dirname(model_dir)
    experiment_dir = os.path.dirname(subset_dir)

    # load model
    model = joblib.load(model_path)

    # load best parameters
    params_path = os.path.join(model_dir, "best_params.json")
    with open(params_path, 'r') as f:
        best_params = json.load(f)

    # extract experiment info from path
    experiment_name = os.path.basename(subset_dir)
    print(subset_dir)
    model_type, tuning_str, log_str, norm_method, seed_info, timestamp = experiment_name.split('_', 5)

    experiment_info = {
        'model_type': model_type,
        'hyperparameter_tuning': tuning_str == "tuned",
        'log_transform': log_str == "log",
        'normalization_method': norm_method,
        'random_seed': int(seed_info[4:]),  # remove "seed" prefix
        'timestamp': timestamp
    }

    # extract subset type
    subset_type = int(os.path.basename(model_dir).split("_")[1])

    model_info = {
        'model': model,
        'best_params': best_params,
        'experiment_info': experiment_info,
        'subset_type': subset_type
    }

    print(f"Model loaded from: {model_path}")
    return model_info

def remove_duplicate_columns(df):
    """
    remove duplicate columns that have '_x' or '_y' suffixes resulting from merging,
    and remove these suffixes from the remaining columns.
    
    Args:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    pd.DataFrame: DataFrame with duplicate columns removed and suffixes stripped
    """
    # get list of columns
    columns = df.columns.tolist()
    
    # identify base names of columns (without _x or _y suffix)
    base_names = set(col.rsplit('_', 1)[0] for col in columns if col.endswith(('_x', '_y')))
    
    # columns to drop and rename
    to_drop = []
    to_rename = {}
    
    for base in base_names:
        variants = [col for col in columns if col.startswith(base) and col.endswith(('_x', '_y'))]
        if len(variants) > 1:
            # keep _x variant and rename it
            keep_col = next(col for col in variants if col.endswith('_x'))
            to_rename[keep_col] = base
            to_drop.extend([col for col in variants if col != keep_col])
        elif len(variants) == 1:
            # if only one variant exists, just rename it
            to_rename[variants[0]] = base
    
    # drop identified columns
    df = df.drop(columns=to_drop)
    
    # rename columns (remove suffixes)
    df = df.rename(columns=to_rename)
    
    return df

def setup_paths():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        'data': os.path.join(base_path, 'data'),
        'out_data': os.path.join(base_path, 'out_data')
    }
