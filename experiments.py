import numpy as np


def load_and_prepare_data(filename):
    data = np.load(f"{filename}.npz", allow_pickle=True)
    features_all = []
    targets = []

    for label in data.files:
        for features in data[label]:
            features_all.append(features)
            targets.append(label)

    features_all = np.array(features_all)
    targets = np.array(targets)

    return features_all, targets


features, targets = load_and_prepare_data("test")
