import numpy as np
import pandas as pd
import copy
import gstools as gs

from scipy.interpolate import griddata

def compute_variogram(spatial_dist_matrix, square_wasser_dist_matrix, spatial_lag=0.0001):
    idx = np.where(square_wasser_dist_matrix > 0)
    spatial_idx = (spatial_dist_matrix[idx].flatten() // spatial_lag).astype(int)

    emp_semivariance_df = pd.DataFrame.from_dict({"spatial": spatial_idx, "wasser": square_wasser_dist_matrix[idx].flatten()})
    emp_semivariance_df = emp_semivariance_df.groupby("spatial").mean()

    return emp_semivariance_df.index.to_numpy(), emp_semivariance_df["wasser"].to_numpy()

def fit_theoretical_semivariogram(bin_center, gamma, nugget=True):

    fit_model = gs.Stable(dim=2)
    fit_model.fit_variogram(bin_center, gamma, nugget=nugget)

    return fit_model

def compute_match_grid(spatial_dist_matrix, wasser_dist_matrix, cluster_match_matrix, spatial_lag=0.0001, spatial_ticks=2500, wasser_lag=0.1, wasser_ticks=120):
    grid = np.zeros((spatial_ticks, wasser_ticks))
    idx = np.where(cluster_match_matrix > 0)
    spatial_idx = (spatial_dist_matrix[idx].flatten() // spatial_lag).astype(int)
    wasser_idx = (wasser_dist_matrix[idx].flatten() // wasser_lag).astype(int)

    c, cnt = np.unique(np.array([spatial_idx, wasser_idx]), axis=1, return_counts=True)
    grid[c[0], c[1]] += cnt

    grid_denom = np.zeros((spatial_ticks, wasser_ticks))
    idx = np.where(wasser_dist_matrix > 0)
    spatial_idx = (spatial_dist_matrix[idx].flatten() // spatial_lag).astype(int)
    wasser_idx = (wasser_dist_matrix[idx].flatten() // wasser_lag).astype(int)

    c, cnt = np.unique(np.array([spatial_idx, wasser_idx]), axis=1, return_counts=True)
    grid_denom[c[0], c[1]] += cnt
    grid_denom[grid_denom==0] = 1

    grid /= grid_denom

    return grid

def compute_match_grid_by_cluster(spatial_dist_matrix, wasser_dist_matrix, cluster_match_matrix, cluster_idx, spatial_lag=0.0001, spatial_ticks=2500, wasser_lag=0.1, wasser_ticks=120):
    cluster_filter = np.zeros_like(cluster_match_matrix)
    cluster_filter[cluster_idx] = 1
    cluster_filter[:,cluster_idx] = 1

    grid = np.zeros((spatial_ticks, wasser_ticks))
    idx = (cluster_match_matrix > 0) & (cluster_filter > 0)
    spatial_idx = (spatial_dist_matrix[idx].flatten() // spatial_lag).astype(int)
    wasser_idx = (wasser_dist_matrix[idx].flatten() // wasser_lag).astype(int)

    c, cnt = np.unique(np.array([spatial_idx, wasser_idx]), axis=1, return_counts=True)
    grid[c[0], c[1]] += cnt

    grid_denom = np.zeros((spatial_ticks, wasser_ticks))
    idx = (wasser_dist_matrix > 0) & (cluster_filter > 0)
    spatial_idx = (spatial_dist_matrix[idx].flatten() // spatial_lag).astype(int)
    wasser_idx = (wasser_dist_matrix[idx].flatten() // wasser_lag).astype(int)

    c, cnt = np.unique(np.array([spatial_idx, wasser_idx]), axis=1, return_counts=True)
    grid_denom[c[0], c[1]] += cnt
    grid_denom[grid_denom==0] = 1

    grid /= grid_denom

    return grid

def compute_density_grid(spatial_dist_matrix, wasser_dist_matrix, spatial_lag=0.0001, spatial_ticks=2500, wasser_lag=0.1, wasser_ticks=120):
    # grid = np.zeros((spatial_ticks, wasser_ticks))
    # idx = np.where(cluster_match_matrix > 0)
    # spatial_idx = (spatial_dist_matrix[idx].flatten() // spatial_lag).astype(int)
    # wasser_idx = (wasser_dist_matrix[idx].flatten() // wasser_lag).astype(int)

    # c, cnt = np.unique(np.array([spatial_idx, wasser_idx]), axis=1, return_counts=True)
    # grid[c[0], c[1]] += cnt

    grid_denom = np.zeros((spatial_ticks, wasser_ticks))
    idx = np.where(wasser_dist_matrix > 0)
    spatial_idx = (spatial_dist_matrix[idx].flatten() // spatial_lag).astype(int)
    wasser_idx = (wasser_dist_matrix[idx].flatten() // wasser_lag).astype(int)

    c, cnt = np.unique(np.array([spatial_idx, wasser_idx]), axis=1, return_counts=True)
    grid_denom[c[0], c[1]] += cnt
    grid_denom[grid_denom==0] = 1

    # grid /= grid_denom

    return grid_denom