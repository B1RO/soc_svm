import warnings
import numpy as np;
import itertools

def product_dict(**kwargs):
    keys = kwargs.keys()
    values = kwargs.values()
    for instance in itertools.product(*values):
        yield dict(zip(keys, instance))


def product_dict_len(**kwargs):
    return np.prod([len(values) for parameter_name, values in kwargs.items()])



def set_estimator_parameters(estimator, parameter_dict):
    for parameter_name, parameter_value in parameter_dict.items():
        if hasattr(estimator, parameter_name):
            setattr(estimator, parameter_name, parameter_value)
        else:
            warnings.warn(
                "You have provided a parameter " + parameter_name + " that is not settable on the provided estimator",
                UserWarning)