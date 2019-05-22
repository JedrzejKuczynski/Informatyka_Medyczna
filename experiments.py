import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
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
    pipe = Pipeline(steps=[("net", clf)])
    param_grid = {
         "net__hidden_layer_sizes": [(25,), (50,), (100,), (150,), (200,),
                                     (50, 25), (100, 50),
                                     (125, 75), (175, 125),
                                     (25, 15, 5), (50, 30, 15),
                                     (75, 45, 25), (100, 60, 35)],
         "net__activation": ["logistic", "tanh", "relu"],
         "net__alpha": [0.001, 0.0001, 0.00001, 0.15, 0.3, 0.5, 0.7],
         "net__max_iter": [100, 200, 300, 400, 500, 600, 700, 800]
         }
    grid_search = GridSearchCV(pipe, param_grid)  # cv=3, verbose=int
    grid_search.fit(X, y)

    search_results_df = pd.DataFrame.from_dict(grid_search.cv_results_)
    best_estimator = grid_search.best_estimator_
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_

    print(best_score, best_params)

    search_results_df.to_csv("neural_net_all_features.csv")
    return


features, targets = load_and_prepare_data("test.npz")
neural_net_experiments(features, targets)
