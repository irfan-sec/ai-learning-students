---
layout: week
title: "Week 11: Unsupervised Learning"
permalink: /week11.html
---

### Navigation
[🏠 Home](index.html) | [← Previous: Week 10](week10.html) | [Next: Week 12 →](week12.html)

---

# Week 11: Unsupervised Learning

## 📚 Learning Objectives
- Understand clustering algorithms (k-means, hierarchical)
- Apply dimensionality reduction techniques (PCA, t-SNE)
- Detect anomalies in data
- Discover hidden patterns without labeled examples
- Evaluate unsupervised learning results

---

## 🎯 Clustering Algorithms

### k-Means Clustering
Group data points into k clusters.

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Example: Customer segmentation
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(customer_data)

# Visualize clusters
plt.scatter(customer_data[:, 0], customer_data[:, 1], c=clusters, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           c='red', marker='x', s=200, alpha=0.5)
plt.show()
```

### Hierarchical Clustering
Build tree-like cluster structures.

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Create dendrogram
linkage_matrix = linkage(data, method='ward')
dendrogram(linkage_matrix)
plt.show()
```

---

## 📊 Dimensionality Reduction

### Principal Component Analysis (PCA)
Find the most important directions in data.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize and reduce dimensions
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
```

### t-SNE for Visualization
Non-linear dimensionality reduction for visualization.

```python
from sklearn.manifold import TSNE

# High-dimensional data visualization
tsne = TSNE(n_components=2, random_state=42)
data_tsne = tsne.fit_transform(high_dim_data)

plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels, cmap='tab10')
plt.title('t-SNE Visualization')
plt.show()
```

---

## ⚠️ Anomaly Detection

### Identifying Outliers
Find unusual patterns in data.

```python
from sklearn.ensemble import IsolationForest

# Detect anomalies
isolation_forest = IsolationForest(contamination=0.1, random_state=42)
anomalies = isolation_forest.fit_predict(data)

# -1 indicates anomaly, 1 indicates normal
normal_data = data[anomalies == 1]
anomalous_data = data[anomalies == -1]
```

---

## 🔗 Applications & Resources

### Real-World Applications
- Market basket analysis
- Gene expression analysis
- Network analysis and community detection
- Recommendation systems

### Video Resources
- [k-Means Clustering - StatQuest](https://www.youtube.com/watch?v=4b5d3muPQmA)
- [PCA Explained - 3Blue1Brown](https://www.youtube.com/watch?v=PFDu9oVAE-g)

---

### Navigation
[🏠 Home](index.html) | [← Previous: Week 10](week10.html) | [Next: Week 12 →](week12.html)