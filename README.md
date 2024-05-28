<div align="center">    
 
# MC-GTA: Metric-Constrained Model-Based Clustering using Goodness-of-fit Tests with Autocorrelations

[![Paper](http://img.shields.io/badge/paper-arxiv.2309.16020-B31B1B.svg)](https://arxiv.org/abs/2309.16020v2)
[![Conference](https://img.shields.io/badge/ICML-2024-blue)]()

![ALT TEXT](/figures/intro-image.png)

</div>

## Description

This is the official repository for the ICML-2024 paper "MC-GTA: Metric-Constrained Model-Based Clustering using Goodness-of-fit Tests with Autocorrelations".

MC-GTA is a model-based, multivariate clustering algorithm that considers complex metric constraints. It clusters observations that are both similar in terms of features and close in terms of metric distance. A wide range of realistic problems fall into this category, e.g. clustering of time series, spatially distributed POIs, or point clouds.

![ALT TEXT](/figures/method.png)

## Method

MC-GTA formulates the clustering objective as minimizing the penalized sum of goodness-of-fit test statistics. A pair of intra-cluster observations is expected to pass the goodness-of-fit test specified by the Wasserstein-2 distance between their underlying models. The criterion of passing such tests is a function of metric distance, i.e., our proposed generalized model-based semivariogram.

![ALT TEXT](/figures/experiment-results.png)

## Tutorial

Run cluster.py. It automatically loads the Pavement dataset with 1055 10-dimensional temporal observations from the `data/pavement/` folder.

For tutorial purpose, we use the first 5 dimensions of the Pavement data and MC-GTA will cluster the temporal sequences into clusters. All intermediate results of time-consuming computations (such as underlying model fitting, semivariogram fitting) are archived in the `checkpoints/` folder. 

After archiving the data, we show how to do very swift hyperparameter tuning by grid search. Set `run_from_archive` to `True` to avoid duplicate intermediate computations. Here for demo purpose only, we split 200 random observations as validation set and tune the hyperparameters against it; after tuning, we apply the hyperparameters to cluster the rest of the dataset and achieve comparable performance. For real applications, you need a separate, independently labeled validation set and repeat the process above.

The tuned hyperparameters achieve 92.76 ARI and 87.99 NMI on the validation set; and on the rest of the dataset, the same hyperparameters achieve 91.93 ARI and 88.33 NMI.