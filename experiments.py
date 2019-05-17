import numpy as np


def load_and_prepare_data(filename):
    data = np.load(f"{filename}.npz", allow_pickle=True)
    X = []
    y = []

    for label in data.files:
        for features in data[label]:
            X.append(features)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    return X, y
