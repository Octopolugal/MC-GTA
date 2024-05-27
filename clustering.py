import os
import numpy as np

from preprocessing import *
from fitting import *
from semivariogram import *

from sklearn.cluster import DBSCAN

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

from matplotlib import pyplot as plt

def mc_gta_penalty(spatial_dist_matrix, square_wasser_dist_matrix, theoretical_semivariogram, delta, spatial_lag=0.0001, rho_thres=1e-6):
    x = spatial_dist_matrix.flatten() / spatial_lag
    pred_y = theoretical_semivariogram.variogram(x)
    deriv = np.abs((pred_y[1:] - pred_y[:-1]))
    rho_idx = np.where(deriv < rho_thres)

    r = np.maximum(0, square_wasser_dist_matrix.flatten() - pred_y + delta)
    r[rho_idx] = r[rho_idx] = 0

    return r

if __name__ == '__main__':
    metric = "euclidean" # For geospatial coordinates, use "haversine"
    dname, feat_dim, loc_dim, k = "pavement", 5, 1, 30

    run_from_archive = True

    if not run_from_archive:
        features, locations, labels = load_dataset(dname, feat_dim, loc_dim)
        dist, nearest_pt = compute_neighborhood(locations, 30, metric)

        valid_idx, estimated_models = fit_models(features, nearest_pt, k)

        print("Valid estimated models: {}".format(len(valid_idx)))

        valid_features, valid_locations, valid_labels = features[valid_idx], locations[valid_idx], labels[valid_idx]

        checkpoint_path = "checkpoints/{}".format(dname)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        np.savez("checkpoints/{}/{}-k{}-fitting".format(dname, dname, k), idx=valid_idx, features=valid_features,
             locations=valid_locations, labels=valid_labels, mean=np.array([m[0] for m in estimated_models]),
             cov=np.array([m[1] for m in estimated_models]))

        # Loading archived fitted models, optional
        data = np.load("checkpoints/{}/{}-k{}-fitting.npz".format(dname, dname, k))

        means, covs, locations, labels = data["mean"], data["cov"], data["locations"], data["labels"]
        estimated_models = list(zip(means, covs))

        spatial_dist_matrix, wasser_dist_matrix = construct_matrices(estimated_models, locations)

        np.savez("checkpoints/{}/{}-k{}-matrix".format(dname, dname, k), spatial=spatial_dist_matrix,
             wasser=wasser_dist_matrix)

    # Loading archived distance matrices, optional
    data = np.load("checkpoints/{}/{}-k{}-fitting.npz".format(dname, dname, k))

    means, covs, locations, labels = data["mean"], data["cov"], data["locations"], data["labels"]
    estimated_models = list(zip(means, covs))

    data = np.load("checkpoints/{}/{}-k{}-matrix.npz".format(dname, dname, k))

    spatial_dist_matrix, wasser_dist_matrix = data["spatial"], data["wasser"]
    square_wasser_dist_matrix = wasser_dist_matrix**2

    # Select appropriate spatial lag according to the minimum and maximum values of the spatial distance matrix

    print("Spatial min: {}, spatial max {}".format(np.min(spatial_dist_matrix), np.max(spatial_dist_matrix)))
    spatial_lag = 5

    bin_center, gamma = compute_variogram(spatial_dist_matrix, square_wasser_dist_matrix, spatial_lag=spatial_lag)
    theoretical_semivariogram = fit_theoretical_semivariogram(bin_center, gamma)

    # Hyperparameter tuning, you can do grid search
    tmp, best_delta, best_beta = 0, 0, 0

    # Use a random subset of ground-truth to do hyperparameter tuning
    rand_idx = np.random.choice(labels.shape[0], 200, replace=False)
    for delta in np.arange(3, 4, 0.1):
        for beta in np.arange(0.5, 0.7, 0.01):
            penalty = mc_gta_penalty(spatial_dist_matrix, square_wasser_dist_matrix, theoretical_semivariogram, delta=delta, spatial_lag=spatial_lag)

            loss = (square_wasser_dist_matrix.flatten() + beta * penalty).reshape(square_wasser_dist_matrix.shape)

            clustering = DBSCAN(eps=2, min_samples=20, metric="precomputed").fit(loss)

            ari, nmi = adjusted_rand_score(labels[rand_idx], clustering.labels_[rand_idx]), adjusted_mutual_info_score(labels[rand_idx], clustering.labels_[rand_idx])
            if ari > tmp:
                print("delta is {}, beta is {}".format(delta, beta))
                print("ARI: {}, NMI: {}".format(ari, nmi))
                tmp, best_delta, best_beta = ari, delta, beta

    betas, aris, nmis = [], [], []
    delta = 3.4
    for beta in np.arange(0.1, 0.9, 0.01):
        penalty = mc_gta_penalty(spatial_dist_matrix, square_wasser_dist_matrix, theoretical_semivariogram, delta=delta, spatial_lag=spatial_lag)

        loss = (square_wasser_dist_matrix.flatten() + beta * penalty).reshape(square_wasser_dist_matrix.shape)

        clustering = DBSCAN(eps=2, min_samples=20, metric="precomputed").fit(loss)

        ari, nmi = adjusted_rand_score(labels, clustering.labels_), adjusted_mutual_info_score(labels, clustering.labels_)

        betas.append(beta)
        aris.append(ari)
        nmis.append(nmi)

    np.savez("beta-tuning", beta=betas, ari=aris, nmi=nmis)


    # Evaluate the clustering performance on the rest of the dataset with the tuned hyperparameters
    print("\n\nPerformance with tuned hyperparameters: delta {}, beta {}".format(best_delta, best_beta))

    penalty = mc_gta_penalty(spatial_dist_matrix, square_wasser_dist_matrix, theoretical_semivariogram, delta=best_delta,
                             spatial_lag=spatial_lag)

    loss = (square_wasser_dist_matrix.flatten() + best_beta * penalty).reshape(square_wasser_dist_matrix.shape)

    clustering = DBSCAN(eps=2, min_samples=20, metric="precomputed").fit(loss)

    test_labels, test_clustering_labels = np.delete(labels, rand_idx), np.delete(clustering.labels_, rand_idx)

    ari, nmi = adjusted_rand_score(test_labels, test_clustering_labels), adjusted_mutual_info_score(
        test_labels, test_clustering_labels)

    print("ARI: {}, NMI: {}".format(ari, nmi))