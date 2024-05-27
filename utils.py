import numpy as np
import scipy
from scipy.spatial import distance

from sklearn.metrics.pairwise import haversine_distances
from math import radians

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

def euclidean_dist(x, y):
    return np.sqrt(np.sum(np.square(x - y), axis=1))


def wasserstein_2(m1, m2, s1, s2):
    d1 = np.sum((m1 - m2) ** 2)
    ssqrt = scipy.linalg.sqrtm(s1)
    mul = np.matmul(np.matmul(ssqrt, s2), ssqrt)
    d2 = np.trace(s1 + s2 - 2 * scipy.linalg.sqrtm(mul))

    if d2 < 0:
        d2 = 0

    return np.sqrt(d1 + d2)


def construct_wasser_dist_matrix(cov_seq):
    N = len(cov_seq)
    print(N)

    wasser_dist_matrix = np.zeros((N, N))

    for i in range(N):
        m1, s1 = cov_seq[i]
        for j in range(i + 1, N):
            m2, s2 = cov_seq[j]
            wd = wasserstein_2(m1, m2, s1, s2)
            wasser_dist_matrix[i, j] = wd
            wasser_dist_matrix[j, i] = wd
        print("\rFinished {}".format(i), end="")

    return wasser_dist_matrix

def compute_wasser_sqrt(cov_seq):
    wasser_sqrt = []

    for m, cov in cov_seq:
        ssqrt = scipy.linalg.sqrtm(cov)
        wasser_sqrt.append(ssqrt)

    return wasser_sqrt


def wasserstein_2_vectorized(m1, m2, s1, s2, ssqrt):
    d1 = np.sum((m1 - m2) ** 2)
    mul = np.matmul(np.matmul(ssqrt, s2), ssqrt)
    d2 = np.trace(s1 + s2 - 2 * scipy.linalg.sqrtm(mul))

    if d2 < 0:
        d2 = 0

    return np.sqrt(d1 + d2)


def construct_wasser_dist_matrix_vectorized(cov_seq):
    wasser_sqrt = compute_wasser_sqrt(cov_seq)

    N = len(cov_seq)
    print(N)

    wasser_dist_matrix = np.zeros((N, N))

    for i in range(N):
        m1, s1 = cov_seq[i]
        ssqrt = wasser_sqrt[i]
        for j in range(i + 1, N):
            m2, s2 = cov_seq[j]
            wd = wasserstein_2_vectorized(m1, m2, s1, s2, ssqrt)
            wasser_dist_matrix[i, j] = wd
            wasser_dist_matrix[j, i] = wd
        print("\rFinished {}".format(i), end="")

    return wasser_dist_matrix

def construct_spatial_dist_matrix(loc_seq):
    N = len(loc_seq)
    print(N)

    spatial_dist_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(i + 1, N):
            sd = np.abs(loc_seq[i] - loc_seq[j])
            spatial_dist_matrix[i, j] = sd
            spatial_dist_matrix[j, i] = sd

    return spatial_dist_matrix


def construct_2D_spatial_dist_matrix(pts):
    N = pts.shape[0]
    print(N)

    spatial_dist_matrix = np.zeros((N, N))

    for i, pt in enumerate(pts):
        dst = np.sqrt(np.sum(np.square(pt - pts), axis=1))
        spatial_dist_matrix[i, :] = dst

    return spatial_dist_matrix


def construct_arc_spatial_dist_matrix(pts):
    pts_in_radians = [[radians(_) for _ in pt] for pt in pts]
    return haversine_distances(pts_in_radians)


def construct_feature_dist_matrix(feat_seq):
    N = len(feat_seq)
    cosine_dist_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(i + 1, N):
            v1, v2 = feat_seq[i], feat_seq[j]
            cd = distance.cosine(v1, v2)
            cosine_dist_matrix[i, j] = cd
            cosine_dist_matrix[j, i] = cd

    return cosine_dist_matrix


def construct_cluster_match_matrix(labels):
    N = len(labels)
    cluster_match_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(i + 1, N):
            v1, v2 = labels[i], labels[j]
            if v1 == v2:
                l = 1
            else:
                l = 0
            cluster_match_matrix[i, j] = l
            cluster_match_matrix[j, i] = l

    return cluster_match_matrix


def construct_by_cluster_match_matrix(labels):
    N = len(labels)
    C = np.unique(labels)

    by_cluster_match_matrices = {}

    for c in C:
        cluster_match_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                v1, v2 = labels[i], labels[j]
                if v1 == c and v2 == c:
                    l = 1
                else:
                    l = 0
                cluster_match_matrix[i, j] = l
                cluster_match_matrix[j, i] = l

        by_cluster_match_matrices[c] = cluster_match_matrix

    return by_cluster_match_matrices


def compute_evaluation_metrics(ground_truth, cluster_result):
    return adjusted_rand_score(ground_truth, cluster_result), adjusted_mutual_info_score(ground_truth, cluster_result)
