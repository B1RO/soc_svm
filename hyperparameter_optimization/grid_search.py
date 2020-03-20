import itertools
import warnings
from colorama import init
from typing import Callable, NamedTuple, Dict, Any

import numpy as np

from ClassifierType import ClassifierType
from hyperparameter_optimization.HyperparameterOptimizationResult import HyperparameterOptimizationResult
from hyperparameter_optimization.common import product_dict, product_dict_len, set_estimator_parameters
from k_fold_cross_validation import k_fold_cross_validation
from KernelType import KernelType
from tqdm import tqdm


init()
def filter_out_redundant_svm_parameters(paramDict):
    if paramDict.get('kernel_type') == KernelType.LINEAR:
        if paramDict.get('classifier_type') == ClassifierType.SOFT_MARGIN:
            allowedParams = ['C', 'loss_type', 'kernel_type', 'classifier_type']
        else:
            allowedParams = []
        return {your_key: paramDict.get(your_key) for your_key in allowedParams if
                paramDict.get(your_key) is not None}
    elif paramDict.get('kernel_type') == KernelType.POLYNOMIAL:
        if paramDict.get('classifier_type') == ClassifierType.SOFT_MARGIN:
            allowedParams = ['C', 'loss_type', 'kernel_type', 'classifier_type', 'degree', 'coef0']
        else:
            allowedParams = ['kernel_type', 'classifier_type', 'degree', 'coef0']
        return {your_key: paramDict.get(your_key) for your_key in allowedParams if
                paramDict.get(your_key) is not None}
    elif paramDict.get('kernel_type') == KernelType.GAUSSIAN:
        if paramDict.get('classifier_type') == ClassifierType.SOFT_MARGIN:
            allowedParams = ['C', 'loss_type', 'kernel_type', 'classifier_type', 'sigma']
        else:
            allowedParams = ['kernel_type', 'classifier_type', 'sigma']
        return {your_key: paramDict.get(your_key) for your_key in allowedParams if
                paramDict.get(your_key) is not None}

def grid_search(estimator,
                score_fn: Callable[[np.ndarray, np.ndarray], float],
                X: np.ndarray,
                y: np.ndarray,
                n=2,
                progressBar=True,
                **kwargs) -> HyperparameterOptimizationResult:
    parameters_to_evaluate = [filter_out_redundant_svm_parameters(paramDict) for paramDict in list(product_dict(**kwargs))]
    parameters_to_evaluate_optimized = [dict(s) for s in set(frozenset(d.items() if d is not None else {}.items()) for d in parameters_to_evaluate)]
    results = np.empty(len(parameters_to_evaluate_optimized), dtype=HyperparameterOptimizationResult)
    topScore = 0
    iter = enumerate(parameters_to_evaluate_optimized);
    for i, parameters in iter:
        set_estimator_parameters(estimator, parameters)
        score_at_parameters = k_fold_cross_validation(estimator, score_fn, X, y, n, False)
        results[i] = HyperparameterOptimizationResult(parameters, score_at_parameters)
        topScore = score_at_parameters if score_at_parameters > topScore else topScore
        # if progressBar:
        #     iter.set_description(desc="top score: {0:.2f}".format(round(topScore,2)))
    return max(results, key=lambda x: x[1])
