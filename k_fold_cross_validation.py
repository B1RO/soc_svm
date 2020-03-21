from typing import Callable

import numpy as np

from validation import get_dataset_splits, get_stratified_dataset_splits
from tqdm import tqdm
from colorama import init
init()
def k_fold_cross_validation(estimator,
                            score_fns: list,
                            X: np.array,
                            y: np.array,
                            n: int,
                            stratified: bool = False):
    if n == 1:
        raise ValueError("K-fold validation with 1 split is meaningless")
    scores = np.empty((n,len(score_fns)))
    splits = get_stratified_dataset_splits(y, n) if stratified else get_dataset_splits(y, n)
    for i in range(n):
        holdout_indices = np.concatenate(np.delete(splits, i, axis=0))
        test_indices = splits[i]
        X_train, y_train, X_test, y_test = X[holdout_indices], y[holdout_indices], X[test_indices], y[test_indices]
        estimator.fit(X_train, y_train)
        predicted = estimator.predict(X_test)
        for j, score_fn in enumerate(score_fns):
            scores[i][j] = score_fn(predicted,y_test)
    return scores.mean(axis=0)
