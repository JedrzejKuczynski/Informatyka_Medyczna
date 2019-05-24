import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(filename):
    data = np.load(filename, allow_pickle=True)
    features_all = []
    targets = []

    for label in data.files:
        for features in data[label]:
            features_all.append(features)
            targets.append(label)

    features_all = np.array(features_all)
    targets = np.array(targets)

    features_all = StandardScaler().fit_transform(features_all)

    return features_all, targets


def neural_net_experiments(X, y):
    clf = MLPClassifier(solver="lbfgs", random_state=42)
    param_grid = {
        "hidden_layer_sizes": [(25,), (50,), (100,), (150,), (200,),
                               (50, 25), (100, 50),
                               (125, 75), (175, 125),
                               (25, 15, 5), (50, 30, 15),
                               (75, 45, 25), (100, 60, 35)],
        "activation": ["logistic", "tanh", "relu"],
        "alpha": [0.001, 0.0001, 0.00001, 0.15, 0.3, 0.5, 0.7],
        "max_iter": [100, 200, 300, 400, 500, 600, 700, 800]
        }
    grid_search = GridSearchCV(clf, param_grid, scoring="accuracy",
                               cv=3, iid=False, n_jobs=-1)
    grid_search.fit(X, y)

    search_results_df = pd.DataFrame.from_dict(grid_search.cv_results_)
    best_estimator = grid_search.best_estimator_
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_

    print("Neural Net: \n", best_score, best_params, "\n")

    # search_results_df.to_csv("neural_net_all_features.csv", index=False)

    """

    # Feature Selection

    feature_selection = SelectPercentile(percentile=75)
    X_selected = feature_selection.fit_transform(X, y)
    selected_features = feature_selection.get_support()

    print(X.shape, X_selected.shape)

    """

    return


def naive_bayes_experiments(X, y):
    clf = GaussianNB()
    param_grid = {
        "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5,
                          1e-4, 1e-3, 1e-2, 1e-1]
    }
    grid_search = GridSearchCV(clf, param_grid, scoring="accuracy",
                               cv=3, iid=False)
    grid_search.fit(X, y)

    search_results_df = pd.DataFrame.from_dict(grid_search.cv_results_)
    best_estimator = grid_search.best_estimator_
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_

    print("Bayes: \n", best_score, best_params, "\n")

    search_results_df.to_csv("naive_bayes_all_features.csv", index=False)

    """

    # Feature Selection

    feature_selection = SelectPercentile(percentile=75)
    X_selected = feature_selection.fit_transform(X, y)
    selected_features = feature_selection.get_support()

    print(X.shape, X_selected.shape)

    """

    return


def gaussian_process_experiments(X, y):
    kernel = 1.0 * RBF(1.0)
    clf = GaussianProcessClassifier(kernel=kernel, random_state=42)
    param_grid = {
        "n_restarts_optimizer": [0, 1, 2, 3, 4]
    }
    grid_search = GridSearchCV(clf, param_grid, scoring="accuracy",
                               cv=3, iid=False, n_jobs=-1)
    grid_search.fit(X, y)

    search_results_df = pd.DataFrame.from_dict(grid_search.cv_results_)
    best_estimator = grid_search.best_estimator_
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_

    print("Gaussian Process: \n", best_score, best_params, "\n")

    search_results_df.to_csv("gaussian_process_all_features.csv", index=False)

    """

    # Feature Selection

    feature_selection = SelectPercentile(percentile=75)
    X_selected = feature_selection.fit_transform(X, y)
    selected_features = feature_selection.get_support()

    print(X.shape, X_selected.shape)

    """

    return


features, targets = load_and_prepare_data("test.npz")
neural_net_experiments(features, targets)
naive_bayes_experiments(features, targets)
gaussian_process_experiments(features, targets)
