import random
from random import sample

from matplotlib import pyplot as plt

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils._testing import ignore_warnings
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning


# returns n models with
def get_hyperparameter_configuration(n):

    # define configurations to sample from
    svr_kernel = ['linear', 'rbf', 'poly']
    svr_C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    svr_degree = [2, 3, 4, 5, 6, 7, 8]
    svr_gamma = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    knn_neighbors = [1, 3, 5, 7, 9, 11]
    knn_weights = ['uniform', 'distance']

    rf_n_estimators = [25, 50, 75, 100, 125, 150, 175, 200]
    rf_criterion = ["squared_error", "absolute_error", "poisson"]
    rf_min_samples = [2, 4, 6, 8, 10]
    rf_min_samples_leaf = [1, 2, 3, 4, 5, 6]

    lasso_alpha = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    configurations = []
    for i in range(n):
        # generate uniformly distributed variable to choose the model
        uniform = np.random.uniform(0, 4, 1)

        if uniform < 1:
            params = {"kernel": sample(svr_kernel, 1)[0], "C": sample(svr_C, 1)[0]}
            # conditional hyper-parameters
            if params.get('kernel') == 'rbf':
                params["gamma"] = sample(svr_gamma, 1)[0]
            elif params.get('kernel') == 'poly':
                params["degree"] = sample(svr_degree, 1)[0]

            svr = SVR(**params)
            configurations.append(svr)
        elif uniform < 2:
            params = {"n_neighbors": sample(knn_neighbors, 1)[0], "weights": sample(knn_weights, 1)[0]}
            knn = KNeighborsRegressor(**params)
            configurations.append(knn)
        elif uniform < 3:
            params = {"n_estimators": sample(rf_n_estimators, 1)[0], "criterion": sample(rf_criterion, 1)[0],
                      "min_samples_split": sample(rf_min_samples, 1)[0],
                      "min_samples_leaf": sample(rf_min_samples_leaf, 1)[0]}
            rf = RandomForestRegressor(**params)
            configurations.append(rf)
        else:
            params = {"alpha": sample(lasso_alpha, 1)[0]}
            lasso = Lasso(**params)
            configurations.append(lasso)

    return configurations


# preprocesses data, fits ml model, runs predictions and returns RMSE, as well as applied preprocessing
# t - model, r_i - resources, X - input data, y - output variable, seed - seed for training and test split
@ignore_warnings(category=ConvergenceWarning)
def run_then_return_val_loss(t, r_i, X, y, seed):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=seed)

    # preprocess data - this can be extended and, if added to the preprocessing object,
    # be used later on for independent tests on the best model
    preprocessing = []
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    preprocessing.append(scaler)

    # if svr or lasso use r_i value to set max_iter (describes computational resources invested)
    if isinstance(t, SVR):
        params = t.get_params()
        params["max_iter"] = r_i * 100
        t = SVR(**params)
    elif isinstance(t, Lasso):
        params = t.get_params()
        params["max_iter"] = r_i * 100
        t = Lasso(**params)
    t.fit(X_train, np.ravel(y_train))
    pred = t.predict(X_test)

    # return loss (RMSE) and the preprocessing applied
    return mean_squared_error(y_test, pred, squared=False), preprocessing


# returns top k configurations
def top_k(T, L, k):
    # sort values by loss
    min_loss = np.argsort(L)[:k]

    # add used models to results
    res = []
    for i in min_loss:
        res.append(T[i])

    return res


# main function - returns min model, min RMSE and preprocessing applied to the model
# R - resources to invest, X - input data, y - output data
def hyperband_algorithm(R, X, y, eta=3):
    # get log of R with base eta
    s_max = int(np.floor(np.log(R) / np.log(eta)))
    B = (s_max + 1) * R

    # define variables needed to store best model
    min_loss = 10000000000
    min_model = None
    min_model_preprocessing = None
    counter = 0

    plt.title("Hyperband History:")
    plt.xlabel("Iterations")
    plt.ylabel("Performance [RMSE]")

    # outer loop - iterates over different n and r values
    for s in reversed(range(s_max + 1)):
        n = int(np.ceil((B / R) * ((eta ** s) / (s + 1))))
        r = R * (eta ** (-s))

        # begin successive halving
        T = get_hyperparameter_configuration(n)

        for i in range(0, s):
            n_i = int(np.floor(n * (eta ** (-i))))
            r_i = r * (eta ** i)

            # generate seed to use the same training and test split for evaluation
            seed = random.randint(a=0, b=100000)

            # train models and evaluate them
            L = [run_then_return_val_loss(t, r_i, X, y, seed) for t in T]

            # extract loss and preprocessing from results
            losses = []
            preprocessing = []
            for j in L:
                losses.append(j[0])
                preprocessing.append(j[1])

            plt.scatter(np.repeat(counter, len(losses)), losses)
            counter += 1

            for j in losses:
                # if min model - store
                if j < min_loss:
                    min_loss = j
                    min_model = T[losses.index(j)]
                    min_model_preprocessing = preprocessing[losses.index(j)]

            # retrieve best k results
            T = top_k(T, losses, int(np.floor(n_i / eta)))

    plt.savefig("plot/hyperband_history.png")
    plt.show()
    return min_model, min_loss, min_model_preprocessing
