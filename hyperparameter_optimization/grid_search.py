from colorama import init
from typing import Callable
import pandas as pd

import numpy as np
from colorama import init
from tqdm import tqdm

from ClassifierType import ClassifierType
from KernelType import KernelType
from hyperparameter_optimization.HyperparameterOptimizationResult import HyperparameterOptimizationResult
from hyperparameter_optimization.common import product_dict, set_estimator_parameters
from k_fold_cross_validation import k_fold_cross_validation

init()


def filter_out_redundant_svm_parameters(paramDict):
    if paramDict.get('kernel_type') == KernelType.LINEAR or paramDict.get('kernel_type') is None:
        if paramDict.get('classifier_type') == ClassifierType.SOFT_MARGIN:
            allowedParams = ['C', 'loss_type', 'kernel_type', 'classifier_type', 'no_bias']
        else:
            allowedParams = ['kernel_type', 'no_bias']
        return {your_key: paramDict.get(your_key) for your_key in allowedParams if
                paramDict.get(your_key) is not None}
    elif paramDict.get('kernel_type') == KernelType.POLYNOMIAL:
        if paramDict.get('classifier_type') == ClassifierType.SOFT_MARGIN:
            allowedParams = ['C', 'loss_type', 'kernel_type', 'classifier_type', 'degree', 'coef0', 'no_bias']
        else:
            allowedParams = ['kernel_type', 'classifier_type', 'degree', 'coef0', 'no_bias']
        return {your_key: paramDict.get(your_key) for your_key in allowedParams if
                paramDict.get(your_key) is not None}
    elif paramDict.get('kernel_type') == KernelType.GAUSSIAN:
        if paramDict.get('classifier_type') == ClassifierType.SOFT_MARGIN:
            allowedParams = ['C', 'loss_type', 'kernel_type', 'classifier_type', 'sigma', 'no_bias']
        else:
            allowedParams = ['kernel_type', 'classifier_type', 'sigma', 'no_bias']
        return {your_key: paramDict.get(your_key) for your_key in allowedParams if
                paramDict.get(your_key) is not None}


def grid_search(estimator,
                score_fn: Callable[[np.ndarray, np.ndarray], float],
                X: np.ndarray,
                y: np.ndarray,
                n=2,
                progressBar=True,
                **kwargs):
    df = pd.DataFrame()
    parameters_to_evaluate = [paramDict for paramDict in
                              list(product_dict(**kwargs))]
    # parameters_to_evaluate_optimized = [dict(s) for s in set(frozenset(d.items() if d is not None else {}.items()) for d in parameters_to_evaluate)]
    results = np.empty(len(parameters_to_evaluate), dtype=HyperparameterOptimizationResult)
    topScore = 0
    iter = tqdm(enumerate(parameters_to_evaluate), miniters=1, total=len(parameters_to_evaluate),
                bar_format='grid search | {l_bar}{bar}| {elapsed} {n_fmt}/{total_fmt}')
    for i, parameters in iter:
        set_estimator_parameters(estimator, parameters)
        score_at_parameters = k_fold_cross_validation(estimator, score_fn, X, y, n, False)
        results[i] = HyperparameterOptimizationResult(parameters, score_at_parameters)
        df = df.append(pd.Series(score_at_parameters), ignore_index=True)
        topScore = score_at_parameters[0] if score_at_parameters[0] > topScore else topScore
        iter.set_description(desc="top score: {0:.2f}".format(round(topScore, 2)))
    return max(results, key=lambda x: x.score[0]), df

