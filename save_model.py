import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib


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


features, targets = load_and_prepare_data("test.npz")

param_grid = {
              "max_depth": [15],
              "min_samples_split": [2],
              "n_estimators": [200]
             }

param_grid2 = {
               "activation": ["tanh"],
               "alpha": [0.15],
               "hidden_layer_sizes": [(100, 50)],
               "max_iter": [100]
              }

{'n_neighbors': 3, 'weights': 'distance'}

param_grid3 = {
               "n_neighbors": [3],
               "weights": ["distance"]
              }

clf = RandomForestClassifier()
clf2 = MLPClassifier()
clf3 = KNeighborsClassifier()
grid_search = GridSearchCV(clf3, param_grid3, scoring="accuracy",
                           cv=3, iid=False)
grid_search.fit(features, targets)
best_estimator = grid_search.best_estimator_
joblib.dump(best_estimator, "Markowska_Kuczyński_classifier.pkl")