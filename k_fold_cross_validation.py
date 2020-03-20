from typing import Callable

import numpy as np

from validation import get_dataset_splits, get_stratified_dataset_splits
from tqdm import tqdm
from colorama import init
init()
def k_fold_cross_validation(estimator,
                            score_fn: Callable[[np.array, np.array], float],
                            X: np.array,
                            y: np.array,
                            n: int,
                            stratified: bool = False):
    if n == 1:
        raise ValueError("K-fold validation with 1 split is meaningless")
    scores = np.empty(n)
    splits = get_stratified_dataset_splits(y, n) if stratified else get_dataset_splits(y, n)
    iter = range(n)
    for i in iter:
        # iter.set_description(desc="split size : " + str(len(splits[i])))
        holdout_indices = np.concatenate(np.delete(splits, i, axis=0))
        test_indices = splits[i]
        X_train, y_train, X_test, y_test = X[holdout_indices], y[holdout_indices], X[test_indices], y[test_indices]
        estimator.fit(X_train, y_train)
        predicted = estimator.predict(X_test)
        scores[i] = (score_fn(y_test, predicted))
    return scores.mean()
