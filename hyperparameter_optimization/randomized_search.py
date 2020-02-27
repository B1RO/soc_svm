import math
from typing import Dict, Any

import numpy as np

from hyperparameter_optimization.grid_search import set_estimator_parameters
from k_fold_cross_validation import k_fold_cross_validation


def randomized_search(estimator, score_fn, X, y, parameter_dict, n_iter=10):
    search_space_dim = len(parameter_dict)
    random_position = np.random.rand(search_space_dim)
    sphere_lower_bounds = np.array([bounds[0] for key, bounds in parameter_dict.items()])
    sphere_upper_bounds = np.array([bounds[1] for key, bounds in parameter_dict.items()])
    best_parameters = None
    last_score = -math.inf
    for i in range(n_iter):
        normal_deviates_point = np.random.normal(0, 1, search_space_dim)
        radius = np.sqrt(np.sum(normal_deviates_point ** 2))
        point_on_surface = 1 / radius * normal_deviates_point
        estimatorParameters = {k: (random_position + point_on_surface)[i] for i, (k, v) in
                               enumerate(parameter_dict.items())}
        set_estimator_parameters(estimator, estimatorParameters)
        score_at_parameters = k_fold_cross_validation(estimator, score_fn, X, y, 2, False)
        if score_at_parameters > last_score:
            best_parameters = estimatorParameters
            random_position += point_on_surface
            last_score = score_at_parameters

    return {
        "parameters": best_parameters,
        "score": last_score
    }
