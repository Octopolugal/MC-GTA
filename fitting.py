import numpy as np
import pandas as pd
import libpysal

import warnings

import scipy
from scipy.spatial import distance

from matplotlib import pyplot as plt

from sklearn.datasets import make_sparse_spd_matrix

from sklearn.preprocessing import normalize
from sklearn.covariance import GraphicalLasso

from sklearn.cluster import DBSCAN
from collections import Counter

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from sklearn.metrics.pairwise import haversine_distances
from math import radians

from utils import *

warnings.filterwarnings("ignore")

def fit_models(features, nearest_pt, radius):
    glasso = GraphicalLasso(alpha=0.01, max_iter=1000)

    estimated_models = []
    valid_idx = []

    fails, warns = 0, 0

    warnings.filterwarnings("error")

    for i in range(features.shape[0]):
        data = features[nearest_pt[i, :radius], :]
        try:
            fit = glasso.fit(data)
        except Warning:
            warns += 1
            # continue
        except:
            fails += 1
            continue

        print("\rProcessed {} out of {}, {} warnings, {} failures.".format(i, features.shape[0], warns, fails), end='')

        valid_idx.append(i)
        estimated_models.append((fit.location_, fit.covariance_))

    warnings.filterwarnings("ignore")

    return valid_idx, estimated_models

def construct_matrices(cov_seq, locations, metric="euclidean"):
    if metric == "euclidean":
        spatial_dist_matrix = construct_2D_spatial_dist_matrix(locations)
    elif metric == "haversine":
        spatial_dist_matrix = construct_arc_spatial_dist_matrix(locations)
    wasser_dist_matrix = construct_wasser_dist_matrix_vectorized(cov_seq)

    return spatial_dist_matrix, wasser_dist_matrix


