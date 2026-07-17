---
marp: true
theme: course
paginate: true
footer: 'Ch.3 · Unsupervised Learning & Preprocessing'
---

<!-- _class: lead -->

# Chapter 3
## Unsupervised Learning & Preprocessing

Scaling, dimensionality reduction,
and clustering

---

## Learning Objectives

- Understand unsupervised learning and why it's **hard to evaluate**
- Scale features correctly (and avoid the classic leakage bug)
- Reduce dimensionality with **PCA**, **NMF**, **t-SNE**
- Group data with **k-Means**, **agglomerative**, **DBSCAN**

---

## Types of Unsupervised Learning

- **Transformations** — create a new representation of the data
  - dimensionality reduction, feature extraction, topic discovery
- **Clustering** — partition data into groups of similar items

**Challenge:** no labels → no easy accuracy score. Usually evaluated
by inspection or by whether it helps a downstream task.

---

## Preprocessing & Scaling

Many models (SVM, k-NN, neural nets) assume features are on similar scales.

| Scaler | What it does |
|--------|--------------|
| `StandardScaler` | mean 0, variance 1 |
| `MinMaxScaler` | squeeze to [0, 1] |
| `RobustScaler` | uses median/quartiles (robust to outliers) |
| `Normalizer` | scale each sample to unit length |

---

## Scale Train and Test the *Same* Way

⚠️ **Fit the scaler on training data only**, then apply to both.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)   # learn stats from TRAIN
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)     # SAME transform on TEST
```

Fitting on the test set = **data leakage** → over-optimistic results.

---

## Principal Component Analysis (PCA)

- Rotates data to **uncorrelated** axes ordered by variance
- Keep the top components → compress while preserving structure
- Uses: visualization, noise reduction, feature extraction (e.g. faces)

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(X_scaled)
X_pca = pca.transform(X_scaled)
```

Components are directions in the original feature space.

---

## Non-Negative Matrix Factorization (NMF)

- Like PCA but components are **non-negative** and additive
- Yields **parts-based**, more interpretable factors
- Great for data that is inherently additive (audio, text, images)

```python
from sklearn.decomposition import NMF
NMF(n_components=15, random_state=0).fit(X)
```

---

## Manifold Learning with t-SNE

- Nonlinear method **for visualization** (usually to 2-D)
- Keeps nearby points close → reveals cluster structure
- **No `transform`** for new data; distances/axes aren't meaningful

```python
from sklearn.manifold import TSNE
X_tsne = TSNE(random_state=42).fit_transform(X)
```

Use it to *see* structure, not as a preprocessing step for models.

---

## Clustering — k-Means

- Partition data into **k** clusters around centroids
- Must choose `k` in advance; assumes round, equally-sized clusters

```python
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=0).fit(X)
km.labels_          # cluster assignment per point
km.cluster_centers_ # centroid coordinates
```

Fast and scalable; struggles with complex shapes.

---

## Agglomerative Clustering

- **Bottom-up:** start with each point, merge closest clusters
- Produces a **dendrogram** (hierarchy) you can cut at any level
- Linkage options: `ward`, `average`, `complete`

```python
from sklearn.cluster import AgglomerativeClustering
AgglomerativeClustering(n_clusters=3).fit(X)
```

Handles varied shapes better than k-Means; still needs a cluster count.

---

## DBSCAN

- Density-based: clusters = dense regions separated by sparse areas
- **No need to set the number of clusters**
- Finds arbitrary shapes and labels outliers as **noise** (-1)
- Key params: `eps` (neighborhood size), `min_samples`

```python
from sklearn.cluster import DBSCAN
DBSCAN(eps=0.5, min_samples=5).fit(X)
```

---

## Comparing & Evaluating Clusters

- **With ground truth:** `adjusted_rand_score`, `normalized_mutual_info_score`
- **Without ground truth:** `silhouette_score` (but higher ≠ always meaningful)
- Best validation is often **manual inspection** of the resulting groups

⚠️ Don't use plain `accuracy` — cluster label numbers are arbitrary.

---

## Chapter Summary

- Unsupervised learning finds structure with **no labels**
- Always scale — and fit the scaler on **training data only**
- **PCA / NMF / t-SNE** compress or visualize high-dimensional data
- **k-Means / agglomerative / DBSCAN** each suit different cluster shapes
- Evaluation is hard — rely on downstream tasks or inspection

**Next:** Chapter 4 — Representing Data & Engineering Features
