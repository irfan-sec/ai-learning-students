# Week 11 Exercises: Unsupervised Learning

## 🎯 Learning Goals
- Implement K-Means clustering
- Apply PCA for dimensionality reduction
- Use hierarchical clustering
- Evaluate clustering quality

## 📝 Conceptual Questions

### Question 1: Clustering vs Classification
**Compare these approaches in terms of labeled data, goals, and use cases.**

### Question 2: Determining K
**Methods for choosing number of clusters:**
- Elbow method
- Silhouette score
- Domain knowledge

## 💻 Programming Exercises

### Exercise 1: K-Means from Scratch
```python
class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
    
    def fit(self, X):
        # TODO: Initialize centroids
        # TODO: Iterate until convergence
        pass
    
    def predict(self, X):
        # TODO: Assign to nearest centroid
        pass
```

### Exercise 2: PCA Implementation
```python
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
    
    def fit(self, X):
        # TODO: Compute covariance matrix
        # TODO: Find eigenvectors
        pass
    
    def transform(self, X):
        # TODO: Project onto principal components
        pass
```

### Exercise 3: Customer Segmentation
```python
from sklearn.cluster import KMeans

def segment_customers(data):
    # TODO: Preprocess data
    # TODO: Apply K-Means
    # TODO: Analyze and visualize segments
    pass
```

## 🧩 Challenge: Image Compression
Use K-Means to compress images by reducing the number of colors.

---
**Due Date: End of Week 11**
