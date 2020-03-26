import math
import matplotlib.pyplot as plt
from typing import Dict, Any

import numpy as np
from tqdm import tqdm

from hyperparameter_optimization.HyperparameterOptimizationResult import HyperparameterOptimizationResult
from hyperparameter_optimization.grid_search import set_estimator_parameters
from k_fold_cross_validation import k_fold_cross_validation
from validation import get_stratified_dataset_splits, get_dataset_splits


def randomized_search(estimator, score_fn, X, y,parameter_dict, n_iter=10):
    r = 10;
    iterations_without_improvement = 0
    search_space_dim = len(parameter_dict)
    random_position = np.random.rand(search_space_dim)
    sphere_lower_bounds = np.array([bounds[0] for key, bounds in parameter_dict.items()])
    sphere_upper_bounds = np.array([bounds[1] for key, bounds in parameter_dict.items()])
    best_parameters = None
    last_score = -math.inf
    iter  = tqdm(range(n_iter), miniters=1, total=n_iter,
                bar_format='randomized search | {l_bar}{bar}| {elapsed} {n_fmt}/{total_fmt}')
    for i in iter:
        if iterations_without_improvement > 5:
            r /= 2
        normal_deviates_point = np.random.normal(0, 1, search_space_dim)
        radius = np.sqrt(np.sum(normal_deviates_point ** 2))
        point_on_surface = 1 / radius * normal_deviates_point*r
        #current radius
        point_on_surface_larger_radius = 1 / radius * normal_deviates_point*r*2
        estimatorParameters = {k: (random_position + point_on_surface)[i] for i, (k, v) in
                               enumerate(parameter_dict.items())}
        set_estimator_parameters(estimator, estimatorParameters)
        score_at_parameters = k_fold_cross_validation(estimator, score_fn, X, y, 4, False)

        #larger radius
        estimatorParameters_larger = {k: (random_position + point_on_surface_larger_radius)[i] for i, (k, v) in
                                      enumerate(parameter_dict.items())}
        set_estimator_parameters(estimator, estimatorParameters_larger)
        score_at_parameters_larger = k_fold_cross_validation(estimator, score_fn, X, y, 4, False)
        better_score = score_at_parameters[0] if score_at_parameters[0] > score_at_parameters_larger[0] else score_at_parameters_larger[0]
        if score_at_parameters_larger[0] > score_at_parameters[0] and score_at_parameters_larger[0] > last_score:
            iterations_without_improvement = 0
            best_parameters = estimatorParameters_larger
            random_position += point_on_surface_larger_radius
            last_score = better_score
            r*=2
        elif score_at_parameters[0] > last_score:
            iterations_without_improvement = 0
            best_parameters = estimatorParameters
            random_position += point_on_surface
            last_score = better_score
        else:
            iterations_without_improvement += 1
        randomized_search.scores.append(better_score)
        randomized_search.best_scores.append(last_score)
        iter.set_description(desc="top score: {0:.2f}".format(round(last_score, 2)))
    return HyperparameterOptimizationResult(best_parameters,last_score)
randomized_search.scores = []
randomized_search.best_scores = []

haf = ['r', 'b', 'g', 'y']
def nested_randomized_search(estimator, score_fn, X, y, n, n_iter,parameter_dict, stratified=False, **kwargs):
    if n == 1:
        raise ValueError("It does not make sense to do a k fold validation with 1 split")
    splits = get_stratified_dataset_splits(y, n) if stratified else get_dataset_splits(y, n)
    model = randomized_search(estimator, score_fn, X, y, parameter_dict, n_iter=n_iter, **kwargs)
    set_estimator_parameters(estimator, model.parameters)
    scores = k_fold_cross_validation(estimator,score_fn,X,y,n)
    plt.plot(randomized_search.scores, linestyle='-')
    randomized_search.scores = []
    plt.plot(randomized_search.best_scores, linestyle='--', alpha=0.7)
    randomized_search.best_scores = []
    return HyperparameterOptimizationResult(model.parameters, scores)

nested_randomized_search.scores = []
nested_randomized_search.best_scores = []