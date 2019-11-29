import itertools

import numpy as np


def train_test_split(X, test_size):
    n = len(X)
    X_copy = np.copy(X)
    np.random.shuffle(X_copy)
    split_point = int(n * test_size)
    return X_copy[:split_point], X_copy[split_point:]


def k_fold_cross_validation(estimator, scorefn, X, y, n, stratified=False):
    if n == 1:
        raise ValueError("It does not make sense to do a k fold validation with 1 split")
    scores = np.zeros(n)
    if stratified:
        splits = get_dataset_splits(y, n)
    else:
        splits = get_stratified_dataset_splits(y, n)
    for i in range(n):
        holdout_indices = np.delete(splits, i) if n == 2 else np.concatenate(np.delete(splits, i))
        X_train = X[holdout_indices]
        y_train = y[holdout_indices]
        X_test = X[splits[i]]
        y_test = y[splits[i]]
        estimator.train(X_train, y_train)
        predicted = estimator.predict(X_test)
        scores[i] = (scorefn(y_test, predicted))
    return scores.mean()


def get_dataset_splits(y, n):
    dataset_indices = list(range(len(y)))
    return np.array_split(dataset_indices, n)


def get_stratified_dataset_splits(y, n):
    splits = []
    l = len(y)

    # see https://docs.scipy.org/doc/numpy/reference/generated/numpy.array_split.html
    split_sizes = l / n * np.ones(n) if (l / n).is_integer() else np.array(
        (l % n) * [l // n + 1] + (n - (l % n)) * [l // n])

    dataset_positive_indices_iterator = itertools.cycle([i for i, label in enumerate(y) if label == 1])
    dataset_negative_indices_iterator = itertools.cycle([i for i, label in enumerate(y) if label == 0])
    for split_index, split_size in enumerate(split_sizes):
        number_of_positive_samples_in_this_split = split_size // 2
        number_of_negative_samples_in_this_split = split_size - number_of_positive_samples_in_this_split

        positive_split_part_indices = itertools.islice(dataset_positive_indices_iterator,
                                                       number_of_positive_samples_in_this_split)
        negative_split_part_indices = itertools.islice(dataset_negative_indices_iterator,
                                                       number_of_negative_samples_in_this_split)
        splits.append(np.array(list(itertools.chain(positive_split_part_indices,negative_split_part_indices))))
    return splits
