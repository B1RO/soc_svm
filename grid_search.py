import itertools
import warnings

from svm_metrics import accuracy_score
from validation import k_fold_cross_validation


def product_dict(**kwargs):
    keys = kwargs.keys()
    values = kwargs.values()
    for instance in itertools.product(*values):
        yield dict(zip(keys, instance))


def grid_search(estimator, score, X, y, **kwargs):
    results = []
    for parameters in product_dict(**kwargs):
        for parameter_name, parameter_value in parameters.items():
            if hasattr(estimator, parameter_name):
                setattr(estimator, parameter_name, parameter_value)
            else:
                warnings.warn(
                    "You have provided a parameter " + parameter_name + " that is not settable on the provided estimator",
                    UserWarning)
        score_at_parameters = k_fold_cross_validation(estimator, accuracy_score, X, y, 4)
        results.append((parameters, score_at_parameters))
    return min(results, key=lambda x: x[1])
