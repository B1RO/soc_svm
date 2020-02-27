import numpy as np

from hyperparameter_optimization.grid_search import grid_search, set_estimator_parameters, HyperparameterOptimizationResult
from validation import get_dataset_splits, get_stratified_dataset_splits


def nested_grid_search(estimator, score_fn, X, y, n, stratified=False, **kwargs):
    if n == 1:
        raise ValueError("It does not make sense to do a k fold validation with 1 split")
    best_models = np.empty(n, dtype=HyperparameterOptimizationResult)
    splits = get_stratified_dataset_splits(y, n) if stratified else get_dataset_splits(y, n)
    for i in range(n):
        holdout_indices = np.concatenate(np.delete(splits, i, axis=0))
        test_indices = splits[i]
        X_train, y_train, X_test, y_test = X[holdout_indices], y[holdout_indices], X[test_indices], y[test_indices]
        best_models[i] = grid_search(estimator, score_fn, X_train, y_train, **kwargs)
        set_estimator_parameters(estimator, best_models[i].parameters)
        score_fn(estimator.predict(X_test),y_test)
    return max(best_models, key=lambda x: x.score)

