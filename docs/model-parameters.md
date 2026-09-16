# Complete model parameter reference

[Return to the project guide](../README.md#phase-3-modeling-guide)

Generated from `workspace/ml/catalog.py` with `python scripts/build_model_reference.py`. These are the exact exposed controls, not every parameter supported by the upstream libraries. Sample-count and design-size guards can impose tighter effective bounds.

**28 estimator entries.** Regression and classification variants have separate keys.

| Task | Entries |
| --- | --- |
| regression | 12 |
| classification | 10 |
| clustering | 3 |
| reduction | 3 |

## Regression

### Linear regression — `linear`

| Parameter | Type | Default | Allowed | Meaning |
| --- | --- | --- | --- | --- |
| `fit_intercept` | choice | `true` | `true`, `false` | Fit a constant intercept term. |
### Ridge — `ridge`

| Parameter | Type | Default | Allowed | Meaning |
| --- | --- | --- | --- | --- |
| `alpha` | float | `1` | `0.0001`–`100` | Penalty or smoothing strength; interpretation depends on the model. |
### Lasso — `lasso`

| Parameter | Type | Default | Allowed | Meaning |
| --- | --- | --- | --- | --- |
| `alpha` | float | `1` | `0.0001`–`100` | Penalty or smoothing strength; interpretation depends on the model. |
### ElasticNet — `elasticnet`

| Parameter | Type | Default | Allowed | Meaning |
| --- | --- | --- | --- | --- |
| `alpha` | float | `1` | `0.0001`–`100` | Penalty or smoothing strength; interpretation depends on the model. |
| `l1_ratio` | float | `0.5` | `0`–`1` | ElasticNet mixture: 0 is L2, 1 is L1. |
### Polynomial regression — `polynomial`

| Parameter | Type | Default | Allowed | Meaning |
| --- | --- | --- | --- | --- |
| `degree` | int | `2` | `2`–`3` | Maximum degree of the polynomial feature expansion. |
| `fit_intercept` | choice | `true` | `true`, `false` | Fit a constant intercept term. |
### Support vector regression — `svr`

| Parameter | Type | Default | Allowed | Meaning |
| --- | --- | --- | --- | --- |
| `C` | float | `1` | `0.01`–`100` | Inverse regularization strength for the margin/logistic objective. |
| `epsilon` | float | `0.1` | `0.001`–`2` | SVR loss-insensitive margin around the target. |
| `kernel` | choice | `"rbf"` | `"rbf"`, `"linear"`, `"poly"` | Kernel used to represent feature relationships. |
| `gamma` | choice | `"scale"` | `"scale"`, `"auto"` | Kernel coefficient convention. |
### Decision tree — `tree_reg`

| Parameter | Type | Default | Allowed | Meaning |
| --- | --- | --- | --- | --- |
| `max_depth` | int | `8` | `1`–`40` | Maximum depth of an individual tree. |
| `min_samples_split` | int | `2` | `2`–`30` | Minimum observations needed to split a tree node. |
| `min_samples_leaf` | int | `1` | `1`–`30` | Minimum observations remaining in a tree leaf. |
### Random forest — `forest_reg`

| Parameter | Type | Default | Allowed | Meaning |
| --- | --- | --- | --- | --- |
| `max_depth` | int | `8` | `1`–`40` | Maximum depth of an individual tree. |
| `min_samples_split` | int | `2` | `2`–`30` | Minimum observations needed to split a tree node. |
| `min_samples_leaf` | int | `1` | `1`–`30` | Minimum observations remaining in a tree leaf. |
| `n_estimators` | int | `100` | `10`–`300` | Number of trees or boosting stages. |
| `max_features` | choice | `"sqrt"` | `"sqrt"`, `"log2"`, `1.0` | Feature subset considered at a tree split. |
### Gradient boosting — `gradient_reg`

| Parameter | Type | Default | Allowed | Meaning |
| --- | --- | --- | --- | --- |
| `n_estimators` | int | `100` | `10`–`300` | Number of trees or boosting stages. |
| `learning_rate` | float | `0.1` | `0.01`–`1` | Boosting step size, or t-SNE optimization rate. |
| `max_depth` | int | `3` | `1`–`12` | Maximum depth of an individual tree. |
### XGBoost — `xgb_reg`

| Parameter | Type | Default | Allowed | Meaning |
| --- | --- | --- | --- | --- |
| `n_estimators` | int | `100` | `10`–`300` | Number of trees or boosting stages. |
| `learning_rate` | float | `0.1` | `0.01`–`1` | Boosting step size, or t-SNE optimization rate. |
| `max_depth` | int | `3` | `1`–`12` | Maximum depth of an individual tree. |
| `subsample` | float | `1` | `0.5`–`1` | Fraction of observations used per boosting iteration. |
| `colsample_bytree` | float | `1` | `0.5`–`1` | Fraction of features sampled for each XGBoost tree. |
### LightGBM — `lgbm_reg`

| Parameter | Type | Default | Allowed | Meaning |
| --- | --- | --- | --- | --- |
| `n_estimators` | int | `100` | `10`–`300` | Number of trees or boosting stages. |
| `learning_rate` | float | `0.1` | `0.01`–`1` | Boosting step size, or t-SNE optimization rate. |
| `num_leaves` | int | `31` | `2`–`100` | Maximum leaves per LightGBM tree. |
| `min_child_samples` | int | `20` | `2`–`100` | Minimum observations in a LightGBM leaf. |
### K-nearest neighbors — `knn_reg`

| Parameter | Type | Default | Allowed | Meaning |
| --- | --- | --- | --- | --- |
| `n_neighbors` | int | `5` | `1`–`50` | Neighborhood size for KNN or UMAP. |
| `weights` | choice | `"uniform"` | `"uniform"`, `"distance"` | Uniform or distance-weighted neighbor contributions. |
| `p` | choice | `2` | `1`, `2` | Minkowski distance power: 1 Manhattan, 2 Euclidean. |

## Classification

### Decision tree — `tree_class`

| Parameter | Type | Default | Allowed | Meaning |
| --- | --- | --- | --- | --- |
| `max_depth` | int | `8` | `1`–`40` | Maximum depth of an individual tree. |
| `min_samples_split` | int | `2` | `2`–`30` | Minimum observations needed to split a tree node. |
| `min_samples_leaf` | int | `1` | `1`–`30` | Minimum observations remaining in a tree leaf. |
### Random forest — `forest_class`

| Parameter | Type | Default | Allowed | Meaning |
| --- | --- | --- | --- | --- |
| `max_depth` | int | `8` | `1`–`40` | Maximum depth of an individual tree. |
| `min_samples_split` | int | `2` | `2`–`30` | Minimum observations needed to split a tree node. |
| `min_samples_leaf` | int | `1` | `1`–`30` | Minimum observations remaining in a tree leaf. |
| `n_estimators` | int | `100` | `10`–`300` | Number of trees or boosting stages. |
| `max_features` | choice | `"sqrt"` | `"sqrt"`, `"log2"`, `1.0` | Feature subset considered at a tree split. |
### Gradient boosting — `gradient_class`

| Parameter | Type | Default | Allowed | Meaning |
| --- | --- | --- | --- | --- |
| `n_estimators` | int | `100` | `10`–`300` | Number of trees or boosting stages. |
| `learning_rate` | float | `0.1` | `0.01`–`1` | Boosting step size, or t-SNE optimization rate. |
| `max_depth` | int | `3` | `1`–`12` | Maximum depth of an individual tree. |
### XGBoost — `xgb_class`

| Parameter | Type | Default | Allowed | Meaning |
| --- | --- | --- | --- | --- |
| `n_estimators` | int | `100` | `10`–`300` | Number of trees or boosting stages. |
| `learning_rate` | float | `0.1` | `0.01`–`1` | Boosting step size, or t-SNE optimization rate. |
| `max_depth` | int | `3` | `1`–`12` | Maximum depth of an individual tree. |
| `subsample` | float | `1` | `0.5`–`1` | Fraction of observations used per boosting iteration. |
| `colsample_bytree` | float | `1` | `0.5`–`1` | Fraction of features sampled for each XGBoost tree. |
### LightGBM — `lgbm_class`

| Parameter | Type | Default | Allowed | Meaning |
| --- | --- | --- | --- | --- |
| `n_estimators` | int | `100` | `10`–`300` | Number of trees or boosting stages. |
| `learning_rate` | float | `0.1` | `0.01`–`1` | Boosting step size, or t-SNE optimization rate. |
| `num_leaves` | int | `31` | `2`–`100` | Maximum leaves per LightGBM tree. |
| `min_child_samples` | int | `20` | `2`–`100` | Minimum observations in a LightGBM leaf. |
### K-nearest neighbors — `knn_class`

| Parameter | Type | Default | Allowed | Meaning |
| --- | --- | --- | --- | --- |
| `n_neighbors` | int | `5` | `1`–`50` | Neighborhood size for KNN or UMAP. |
| `weights` | choice | `"uniform"` | `"uniform"`, `"distance"` | Uniform or distance-weighted neighbor contributions. |
| `p` | choice | `2` | `1`, `2` | Minkowski distance power: 1 Manhattan, 2 Euclidean. |
### Logistic regression — `logistic`

| Parameter | Type | Default | Allowed | Meaning |
| --- | --- | --- | --- | --- |
| `C` | float | `1` | `0.01`–`100` | Inverse regularization strength for the margin/logistic objective. |
| `class_weight` | choice | `null` | `null`, `"balanced"` | No class weighting or weighting balanced by class frequencies. |
| `max_iter` | int | `500` | `100`–`2000` | Maximum optimization iterations. |
### Support vector classifier — `svm`

| Parameter | Type | Default | Allowed | Meaning |
| --- | --- | --- | --- | --- |
| `C` | float | `1` | `0.01`–`100` | Inverse regularization strength for the margin/logistic objective. |
| `kernel` | choice | `"rbf"` | `"rbf"`, `"linear"`, `"poly"` | Kernel used to represent feature relationships. |
| `gamma` | choice | `"scale"` | `"scale"`, `"auto"` | Kernel coefficient convention. |
| `class_weight` | choice | `null` | `null`, `"balanced"` | No class weighting or weighting balanced by class frequencies. |
### Gaussian naive Bayes — `gaussian_nb`

| Parameter | Type | Default | Allowed | Meaning |
| --- | --- | --- | --- | --- |
| `var_smoothing` | float | `1e-09` | `1e-12`–`0.1` | Added variance stabilization for Gaussian naive Bayes. |
### Multinomial naive Bayes — `multinomial_nb`

| Parameter | Type | Default | Allowed | Meaning |
| --- | --- | --- | --- | --- |
| `alpha` | float | `1` | `0.001`–`10` | Penalty or smoothing strength; interpretation depends on the model. |
| `fit_prior` | choice | `true` | `true`, `false` | Learn class priors from training frequencies. |

## Clustering

### K-Means — `kmeans`

| Parameter | Type | Default | Allowed | Meaning |
| --- | --- | --- | --- | --- |
| `n_clusters` | int | `3` | `2`–`15` | Requested cluster count. |
| `n_init` | int | `10` | `1`–`20` | Number of K-Means initializations. |
| `max_iter` | int | `300` | `100`–`500` | Maximum optimization iterations. |
### Hierarchical / agglomerative — `hierarchical`

| Parameter | Type | Default | Allowed | Meaning |
| --- | --- | --- | --- | --- |
| `n_clusters` | int | `3` | `2`–`15` | Requested cluster count. |
| `linkage` | choice | `"ward"` | `"ward"`, `"complete"`, `"average"`, `"single"` | Criterion used when merging hierarchical clusters. |
### DBSCAN — `dbscan`

| Parameter | Type | Default | Allowed | Meaning |
| --- | --- | --- | --- | --- |
| `eps` | float | `0.5` | `0.01`–`10` | DBSCAN neighborhood radius in preprocessed feature space. |
| `min_samples` | int | `5` | `2`–`50` | Minimum neighborhood support for a DBSCAN core point. |

## Reduction

### Principal component analysis — `pca`

| Parameter | Type | Default | Allowed | Meaning |
| --- | --- | --- | --- | --- |
| `n_components` | int | `2` | `2`–`20` | Number of retained projection/embedding coordinates. |
| `whiten` | choice | `false` | `false`, `true` | Rescale PCA components to unit variance. |
### t-SNE — `tsne`

| Parameter | Type | Default | Allowed | Meaning |
| --- | --- | --- | --- | --- |
| `n_components` | choice | `2` | `2`, `3` | Number of retained projection/embedding coordinates. |
| `perplexity` | float | `30` | `2`–`100` | t-SNE effective neighborhood scale; must also suit row count. |
| `learning_rate` | float | `200` | `10`–`1000` | Boosting step size, or t-SNE optimization rate. |
| `max_iter` | int | `500` | `250`–`1500` | Maximum optimization iterations. |
### UMAP — `umap`

| Parameter | Type | Default | Allowed | Meaning |
| --- | --- | --- | --- | --- |
| `n_components` | choice | `2` | `2`, `3` | Number of retained projection/embedding coordinates. |
| `n_neighbors` | int | `15` | `2`–`100` | Neighborhood size for KNN or UMAP. |
| `min_dist` | float | `0.1` | `0`–`0.99` | UMAP minimum separation parameter in the embedding. |

## Shared training controls

| Control | Default | Constraints |
| --- | --- | --- |
| Features | None selected | 1–50 distinct feature names; numeric/category support; convert dates first. |
| Target | None selected | Required for supervised tasks; separate from features. |
| Test fraction | 0.2 | 0.1–0.5; supervised tasks only. |
| Stratification | Enabled | Classification only; sample counts must permit the split. |
| Seed | 42 | Integer 0–2,147,483,647. |
| Scaling | standard | standard / minmax / robust / none. Multinomial NB forces minmax with clipping. |
| Search method | none | none / grid / random; supervised only. |
| CV folds | 3 | 2–5; fitted preprocessing is refit inside each fold. |
| Random-search iterations | 10 | 1–20; each searched parameter has 1–10 listed candidates. |

See the README for data-size limits, scoring, pipeline persistence and supported new-row prediction.
