# Customer Segmentation with K-Means

Groups customers into behavioural segments using unsupervised learning.

## Pipeline
1. `generate_data.py` – synthetic customers (age, income, spending score, visits, basket value)
2. `train.py` – scale features → test k=2..10 (elbow + silhouette) → fit best K-Means → profile clusters → save plots/model
3. `predict.py` – assign new customers to a segment

## Run
```bash
pip install -r requirements.txt
python src/generate_data.py
python src/train.py
python src/predict.py
python -m unittest discover tests
```

## Outputs
- `reports/k_selection.png` – elbow & silhouette curves
- `reports/clusters.png` – PCA view of segments
- `reports/cluster_profiles.csv` – average features per segment
- `reports/metrics.json` – silhouette, Davies-Bouldin
- `models/kmeans_pipeline.joblib` – scaler + K-Means

## Notes
- Scaling is essential: K-Means uses Euclidean distance.
- Silhouette picks k automatically; combine with business judgement.
- Replace `data/customers.csv` with real data using the same columns.
