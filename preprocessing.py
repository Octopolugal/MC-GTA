import numpy as np
from sklearn.neighbors import KDTree

DISTANCE_METRICS = [
    "cityblock","cosine","euclidean","haversine","manhattan","l1","l2","nan_euclidean"
]

def load_dataset(dname, feat_dim, loc_dim):
    dpath = f"data/{dname}/{dname}-data.txt"
    lpath = f"data/{dname}/{dname}-label.txt"

    data = np.loadtxt(dpath, delimiter=",")
    labels = np.loadtxt(lpath, delimiter=",")

    features, locations = data[:,:feat_dim], data[:,-loc_dim:]
    if loc_dim == 1:
        locations = np.array([np.arange(features.shape[0]), np.zeros(features.shape[0])]).T

    return features, locations, labels

def compute_neighborhood(locations, k=30, metric="euclidean"):
    assert metric in DISTANCE_METRICS, "Unknown distance metric!"
    tree = KDTree(locations, leaf_size=2, metric=metric)
    dist, ind = tree.query(locations, k=k)

    return dist, ind



