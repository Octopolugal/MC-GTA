# A simple example for runing MC-GTA

Run cluster.py. It automatically loads the Pavement dataset with 1055 10-dimensional temporal observations from the `data/pavement/` folder.

For tutorial purpose only we use the first 5 dimensions and MC-GTA will cluster the temporal sequences into clusters. All intermediate results of time-consuming computations (such as underlying model fitting, semivariogram fitting) are archived in the `checkpoints/` folder. 

After archiving the data, we show how to do very swift hyperparameter tuning by grid search. Set `run_from_archive` to `True` to avoid duplicate intermediate computations. Here for demo purpose only, we split 200 random observations as validation set and tune the hyperparameters against it; after tuning, we apply the hyperparameters to cluster the rest of the dataset and achieve comparable performance. For real applications, you need a separate, independently labeled validation set and repeat the process above.

The tuned hyperparameters achieve 92.76 ARI and 87.99 NMI on the validation set; and on the rest of the dataset, the same hyperparameters achieve 91.93 ARI and 88.33 NMI.