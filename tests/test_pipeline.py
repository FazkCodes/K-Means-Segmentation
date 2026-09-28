import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from generate_data import generate
from train import FEATURES
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class TestPipeline(unittest.TestCase):
def setUp(self):
self.df = generate(seed=1)

def test_data_shape_and_no_nulls(self):
self.assertEqual(len(self.df), 1100)
self.assertFalse(self.df.isna().any().any())

def test_clusters_are_well_separated(self):
pipe = Pipeline([("s", StandardScaler()), ("k", KMeans(5, n_init=10, random_state=0))])
labels = pipe.fit_predict(self.df[FEATURES])
self.assertEqual(len(set(labels)), 5)
score = silhouette_score(StandardScaler().fit_transform(self.df[FEATURES]), labels)
self.assertGreater(score, 0.3)

if __name__ == "__main__":
unittest.main()

