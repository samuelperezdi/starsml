import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from astropy.io.votable import parse

def transform_features(df, feature_names, skip_log_features):
    transformed_features = [
        np.log10(1 + df[feature].values) if feature not in skip_log_features else df[feature].values
        for feature in feature_names
    ]
    return np.array(transformed_features).T

def preprocess(df_good, df_bad, feature_names, skip_log_features):
    # Transform features
    X_good = transform_features(df_good, feature_names, skip_log_features)
    X_bad = transform_features(df_bad, feature_names, skip_log_features)

    # Concatenate X and create Y labels
    X = np.concatenate((X_good, X_bad), axis=0)
    Y = np.concatenate((np.ones(len(X_good)), np.zeros(len(X_bad))), axis=0)

    return X, Y

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
