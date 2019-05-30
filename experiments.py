import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


PERCENTILES = [70, 50, 30]


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


def feature_selection(X, y, percent):
    feature_selection = SelectPercentile(percentile=percent)
    X_selected = feature_selection.fit_transform(X, y)
    selected_features = feature_selection.get_support()

    print(X.shape, X_selected.shape)
    return X_selected, selected_features


def draw_plots(classifier, x_axis, y_axis, **kwargs):
    filepath_all = f"Wyniki/{classifier}_all_features.csv"
    data_all_df = pd.read_csv(filepath_all)
    expressions = []

    for key, value in kwargs.items():
        if isinstance(value, tuple):
            bool_expression = f"(data_all_df[r'{key}']==r'{value}')"
        else:
            bool_expression = f"(data_all_df[r'{key}']=={value})"
        expressions.append(bool_expression)

    bool_expression = " & ".join(expressions)

    data_all_subset_df = data_all_df.loc[pd.eval(bool_expression,
                                                 global_dict=locals(),
                                                 local_dict=kwargs)]

    data_all_subset_df.plot(x_axis, y_axis, title=classifier, grid=True,
                            rot=45, legend=False)
    plt.ylabel("Accuracy")
    plt.show()

    percentile_dataframes = []

    for percentile in PERCENTILES:
        filepath_percent = f"Wyniki/{classifier}_{percentile}_features.csv"
        data_percentile_df = pd.read_csv(filepath_percent)
        percentile_dataframes.append(data_percentile_df)

    selected_features_df = pd.concat(percentile_dataframes, axis=0,
                                     ignore_index=True)

    plt.plot(PERCENTILES, selected_features_df[y_axis])
    plt.title(classifier)
    plt.xlabel("Procent cech wybranych do uczenia modelu")
    plt.xticks(rotation=45)
    plt.ylabel("Accuracy")
    plt.grid()
    plt.show()

    return


def main_experiment(X, y, clasifiers):
    for key, value in clasifiers.items():
        clf = value[0]
        param_grid = value[1]

        grid_search = GridSearchCV(clf, param_grid, scoring="accuracy",
                                   cv=3, iid=False, n_jobs=-1)
        grid_search.fit(X, y)

        search_results_df = pd.DataFrame.from_dict(grid_search.cv_results_)
        best_score = grid_search.best_score_
        best_params = grid_search.best_params_

        print("all features")
        print(best_score, best_params)

        search_results_df.to_csv(f"{key}_all_features.csv", index=False)

        for key_param, value_param in best_params.items():
            value_list = []
            value_list.append(value_param)
            best_params[key_param] = value_list

        # Feature Selection
        for percentile in PERCENTILES:
            X_new, selected = feature_selection(X, y, percentile)
            search = GridSearchCV(clf, best_params, scoring="accuracy",
                                  cv=3, iid=False)
            search.fit(X_new, y)
            best_score = search.best_score_
            print(f"{percentile}% features")
            print(best_score, selected)

            percentile_results_df = pd.DataFrame.from_dict(search.cv_results_)
            percentile_results_df.to_csv(f"{key}_{percentile}_features.csv",
                                         index=False)
            np.savetxt(f"{key}_{percentile}_selected_features.txt", selected)


clasifiers = {
    "kNN": (KNeighborsClassifier(),
            {
            "weights": ['uniform', 'distance'],
            "n_neighbors": [3, 5, 7, 10, 15, 20]
            }),
    "random_forest": (RandomForestClassifier(random_state=42),
                      {
                        "n_estimators": [10, 30, 50, 70, 100, 200],
                        "max_depth": [None, 2, 3, 6, 10, 15],
                        "min_samples_split": [2, 4, 7, 10]
                      }),
    "svm_rbf": (SVC(),
                {
                    "C": [1e-2, 0.5e-1, 1e-1, 1, 1e1, 0.5e2, 1e2],
                    "gamma": [0.1, 0.5, 1, 5, 10],
                    "max_iter": [100, 300, 700, -1]
                }),
    "bayes": (GaussianNB(),
              {
              "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5,
                                1e-4, 1e-3, 1e-2, 1e-1]
              }),
    "gaussian_process": (GaussianProcessClassifier(kernel=(1.0 * RBF(1.0)),
                                                   random_state=42),
                         {
                          "n_restarts_optimizer": [0, 1, 2, 3, 4]
                         }),
    "neural_net": (MLPClassifier(solver="lbfgs", random_state=42),
                   {
                    "hidden_layer_sizes": [(25,), (50,), (100,), (150,),
                                           (200,), (50, 25), (100, 50),
                                           (125, 75), (175, 125),
                                           (25, 15, 5), (50, 30, 15),
                                           (75, 45, 25), (100, 60, 35)],
                    "activation": ["logistic", "tanh", "relu"],
                    "alpha": [0.001, 0.0001, 0.00001, 0.15, 0.3, 0.5, 0.7],
                    "max_iter": [100, 200, 300, 400, 500, 600, 700, 800]
                   }),
}

features, targets = load_and_prepare_data("test.npz")
# main_experiment(features, targets, clasifiers)
draw_plots("neural_net", "param_activation", "mean_test_score",
           param_alpha=0.001, param_max_iter=100,
           param_hidden_layer_sizes=(100,))
