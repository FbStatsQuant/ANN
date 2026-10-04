# Forest Covertype (not committed)

- Source: https://archive.ics.uci.edu/dataset/31/covertype
- 581,012 rows, 54 features, 7 classes (forest cover type). Good multiclass benchmark.
- Easiest download is through scikit-learn:

      from sklearn.datasets import fetch_covtype
      X, y = fetch_covtype(return_X_y=True, as_frame=True)
