import numpy as np


def train_test_split(X, test_size):
    n = len(X)
    X_copy = np.copy(X)
    np.random.shuffle(X_copy)
    split_point = int(n * test_size)
    return X_copy[:split_point], X_copy[split_point:]


def k_fold_cross_validation(estimator, scorefn, X, y, n):
    if n==1:
        raise ValueError("It does not make sense to do a k fold validation with 1 split")
    scores = np.zeros(n)
    dataset_indices = list(range(len(X)))
    splits = np.array_split(dataset_indices, n)
    for i in range(n):
        holdout_indices = np.delete(splits, i) if n==2 else np.concatenate(np.delete(splits, i))
        X_train = X[holdout_indices]
        y_train = y[holdout_indices]
        X_test = X[splits[i]]
        y_test = y[splits[i]]
        estimator.train(X_train, y_train)
        predicted = estimator.predict(X_test)
        scores[i] = (scorefn(y_test, predicted))
    return scores.mean()
