import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from hyperparameter_optimization.grid_search import grid_search, set_estimator_parameters, HyperparameterOptimizationResult
from k_fold_cross_validation import k_fold_cross_validation
from validation import get_dataset_splits, get_stratified_dataset_splits
from colorama import init
init()

def nested_grid_search(estimator, score_fn, X, y, n, innerN, stratified=False, **kwargs):
    if n == 1:
        raise ValueError("It does not make sense to do a k fold validation with 1 split")
    splits = get_stratified_dataset_splits(y, n) if stratified else get_dataset_splits(y, n)
    model, df = grid_search(estimator, score_fn, X, y, n=innerN, **kwargs)
    set_estimator_parameters(estimator, model.parameters)
    scores = k_fold_cross_validation(estimator,score_fn,X,y,n)
    return HyperparameterOptimizationResult(model.parameters, scores), df


