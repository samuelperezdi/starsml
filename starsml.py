from sklearn.metrics import accuracy_score
import numpy as np

def compute_permutation_importance(model, X_test, Y_test):
    """Computes permutation importance of features on the given test set using the provided model.
    
    Args:
        model: Trained neural network model.
        X_test: Test features.
        Y_test: True labels for the test set.
        
    Returns:
        feature_importances: A list of importance scores for each feature.
    """
    
    # Get the baseline accuracy
    baseline_pred = model.predict(X_test)
    baseline_accuracy = accuracy_score(Y_test, np.round(baseline_pred))
    
    feature_importances = []
    
    # Iterate over each feature in the dataset
    for i in range(X_test.shape[1]):
        # Make a copy of the original test set
        X_test_permuted = X_test.copy()
        
        # Shuffle the values of the feature
        np.random.shuffle(X_test_permuted[:, i])
        
        # Get predictions on the permuted test set
        permuted_pred = model.predict(X_test_permuted)
        
        # Compute the drop in accuracy
        permuted_accuracy = accuracy_score(Y_test, np.round(permuted_pred))
        importance_score = baseline_accuracy - permuted_accuracy
        
        feature_importances.append(importance_score)
    
    return feature_importances
