import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["age", "annual_income_k", "spending_score", "visits_per_month", "avg_basket_value"]
K_RANGE = range(2, 11)

def evaluate_k(X_scaled):
results = []
for k in K_RANGE:
km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
results.append({"k": k, "inertia": km.inertia_,
"silhouette": silhouette_score(X_scaled, km.labels_)})
return pd.DataFrame(results)

def plot_k_selection(res, best_k, path):
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(res.k, res.inertia, "o-"); ax[0].set_title("Elbow method")
ax[0].set_xlabel("k"); ax[0].set_ylabel("Inertia")
ax[1].plot(res.k, res.silhouette, "o-", color="tab:green")
ax[1].axvline(best_k, ls="--", color="gray"); ax[1].set_title("Silhouette score")
ax[1].set_xlabel("k"); ax[1].set_ylabel("Score")
fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

def plot_clusters(X_scaled, labels, path):
pts = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
fig, ax = plt.subplots(figsize=(6.5, 5))
sc = ax.scatter(pts[:, 0], pts[:, 1], c=labels, cmap="tab10", s=14, alpha=0.8)
ax.set_title("Customer segments (PCA projection)")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.legend(*sc.legend_elements(), title="Cluster")
fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

def main():
df = pd.read_csv("data/customers.csv")
X = df[FEATURES]
X_scaled = StandardScaler().fit_transform(X)

res = evaluate_k(X_scaled)
best_k = int(res.loc[res.silhouette.idxmax(), "k"])
print(res.round(3).to_string(index=False))
print(f"\nBest k by silhouette: {best_k}")

model = Pipeline([("scale", StandardScaler()),
("kmeans", KMeans(n_clusters=best_k, n_init=20, random_state=42))])
labels = model.fit_predict(X)
df["cluster"] = labels

metrics = {"k": best_k,
"silhouette": float(silhouette_score(X_scaled, labels)),
"davies_bouldin": float(davies_bouldin_score(X_scaled, labels))}
profile = df.groupby("cluster")[FEATURES].mean().round(1)
profile.insert(0, "size", df.cluster.value_counts().sort_index())

plot_k_selection(res, best_k, "reports/k_selection.png")
plot_clusters(X_scaled, labels, "reports/clusters.png")
profile.to_csv("reports/cluster_profiles.csv")
df.to_csv("data/customers_segmented.csv", index=False)
json.dump(metrics, open("reports/metrics.json", "w"), indent=2)
joblib.dump(model, "models/kmeans_pipeline.joblib")

print("\nCluster profiles:\n", profile.to_string())
print("\nMetrics:", metrics)

if __name__ == "__main__":
main()

