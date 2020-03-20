import numpy as np
from tqdm import tqdm
from hyperparameter_optimization.grid_search import grid_search, set_estimator_parameters, HyperparameterOptimizationResult
from validation import get_dataset_splits, get_stratified_dataset_splits
from colorama import init
init()

def nested_grid_search(estimator, score_fn, X, y, n, innerN, stratified=False, **kwargs):
    if n == 1:
        raise ValueError("It does not make sense to do a k fold validation with 1 split")
    best_models = np.empty(n, dtype=HyperparameterOptimizationResult)
    splits = get_stratified_dataset_splits(y, n) if stratified else get_dataset_splits(y, n)
    top_score = 0
    iter = range(n)
    for i in range(n):
        holdout_indices = np.concatenate(np.delete(splits, i, axis=0))
        test_indices = splits[i]
        X_train, y_train, X_test, y_test = X[holdout_indices], y[holdout_indices], X[test_indices], y[test_indices]
        model = grid_search(estimator, score_fn, X_train, y_train, progressBar=True, n=innerN, **kwargs)
        set_estimator_parameters(estimator, model.parameters)
        best_models[i] = HyperparameterOptimizationResult(model.parameters, score_fn(y[test_indices], estimator.predict(X[test_indices])))
        print(best_models[i])
        top_score = best_models[i].score if best_models[i].score > top_score else top_score
    return max(best_models, key=lambda x: x.score)