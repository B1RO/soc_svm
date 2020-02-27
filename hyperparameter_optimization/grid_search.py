import itertools
import warnings
from typing import Callable, NamedTuple, Dict, Any

import numpy as np

from hyperparameter_optimization.HyperparameterOptimizationResult import HyperparameterOptimizationResult
from hyperparameter_optimization.common import product_dict, product_dict_len, set_estimator_parameters
from k_fold_cross_validation import k_fold_cross_validation


def grid_search(estimator,
                score_fn: Callable[[np.ndarray, np.ndarray], float],
                X: np.ndarray,
                y: np.ndarray,
                **kwargs) -> HyperparameterOptimizationResult:
    parameters_to_evaluate = product_dict(**kwargs)
    results = np.empty(product_dict_len(**kwargs), dtype=HyperparameterOptimizationResult)
    for i, parameters in enumerate(parameters_to_evaluate):
        set_estimator_parameters(estimator, parameters)
        score_at_parameters = k_fold_cross_validation(estimator, score_fn, X, y, 2, False)
        results[i] = HyperparameterOptimizationResult(parameters, score_at_parameters)
    return max(results, key=lambda x: x[1])
