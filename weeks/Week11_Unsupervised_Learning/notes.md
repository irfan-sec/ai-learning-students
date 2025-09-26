# Week 11: Unsupervised Learning

## 📚 Learning Objectives
By the end of this week, students will be able to:
- Understand the difference between supervised and unsupervised learning
- Implement and apply k-means clustering algorithm
- Use hierarchical clustering for data analysis
- Understand and apply Principal Component Analysis (PCA) for dimensionality reduction
- Evaluate clustering results using appropriate metrics
- Apply unsupervised learning techniques to real-world datasets

---

## 🎯 What is Unsupervised Learning?

**Key Difference:** No labeled data - we want to discover hidden patterns and structure.

### Types of Unsupervised Learning

| **Clustering** | **Dimensionality Reduction** | **Association Rules** |
|---------------|------------------------------|----------------------|
| Group similar data points | Reduce number of features | Find relationships in data |
| k-means, hierarchical | PCA, t-SNE | Market basket analysis |
| Customer segmentation | Data visualization | "People who buy X also buy Y" |

### Applications

- **Customer Segmentation:** Group customers by purchasing behavior
- **Gene Analysis:** Cluster genes with similar expression patterns
- **Image Compression:** Reduce image file sizes
- **Anomaly Detection:** Find unusual patterns in data
- **Data Visualization:** Plot high-dimensional data in 2D/3D

---

## 🎯 k-Means Clustering

### The Intuition

**Goal:** Partition data into k clusters where each point belongs to the cluster with the nearest centroid.

**Key Components:**
- **Centroids:** Center points of clusters
- **Clusters:** Groups of data points
- **Distance Metric:** Usually Euclidean distance

### k-Means Algorithm

**Steps:**
1. **Initialize:** Choose k random centroids
2. **Assign:** Each point to closest centroid
3. **Update:** Move centroids to mean of assigned points
4. **Repeat:** Steps 2-3 until convergence

**Mathematical Formulation:**
- **Objective:** Minimize within-cluster sum of squares (WCSS)
- J = Σᵢ₌₁ᵏ Σₓ∈Cᵢ ||x - μᵢ||²
- Where μᵢ is the centroid of cluster Cᵢ

### Implementation from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class KMeans:
    def __init__(self, k=3, max_iters=100, random_state=42):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        
    def initialize_centroids(self, X):
        """Initialize centroids randomly"""
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        centroids = np.random.uniform(X.min(axis=0), X.max(axis=0), 
                                    size=(self.k, n_features))
        return centroids
    
    def assign_clusters(self, X, centroids):
        """Assign each point to closest centroid"""
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def update_centroids(self, X, labels):
        """Update centroids to mean of assigned points"""
        centroids = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            if np.sum(labels == i) > 0:  # Avoid empty clusters
                centroids[i] = X[labels == i].mean(axis=0)
        return centroids
    
    def fit(self, X):
        """Fit k-means to data"""
        # Initialize centroids
        self.centroids = self.initialize_centroids(X)
        
        # Store history for visualization
        self.centroid_history = [self.centroids.copy()]
        
        for iteration in range(self.max_iters):
            # Assign points to clusters
            labels = self.assign_clusters(X, self.centroids)
            
            # Update centroids
            new_centroids = self.update_centroids(X, labels)
            
            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                print(f"Converged after {iteration + 1} iterations")
                break
                
            self.centroids = new_centroids
            self.centroid_history.append(self.centroids.copy())
        
        self.labels = labels
        return self
    
    def predict(self, X):
        """Predict cluster labels for new data"""
        return self.assign_clusters(X, self.centroids)
    
    def calculate_wcss(self, X):
        """Calculate Within-Cluster Sum of Squares"""
        wcss = 0
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                wcss += ((cluster_points - self.centroids[i])**2).sum()
        return wcss

# Example usage
# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Apply k-means
kmeans = KMeans(k=4, random_state=42)
kmeans.fit(X)

# Plot results
plt.figure(figsize=(12, 4))

# Original data
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.7)
plt.title('Original Data')

# k-means results
plt.subplot(1, 3, 2)
colors = ['red', 'blue', 'green', 'purple']
for i in range(kmeans.k):
    cluster_points = X[kmeans.labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                c=colors[i], alpha=0.7, label=f'Cluster {i}')
    plt.scatter(kmeans.centroids[i, 0], kmeans.centroids[i, 1], 
                c='black', marker='x', s=100)
plt.title('k-Means Clustering')
plt.legend()

# Centroid movement
plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], alpha=0.3, c='gray')
for i in range(kmeans.k):
    centroid_path = np.array([hist[i] for hist in kmeans.centroid_history])
    plt.plot(centroid_path[:, 0], centroid_path[:, 1], 'o-', 
             color=colors[i], markersize=8, linewidth=2)
plt.title('Centroid Movement')
plt.tight_layout()
plt.show()
```

### Choosing k: The Elbow Method

```python
def elbow_method(X, max_k=10):
    """Find optimal k using elbow method"""
    wcss_values = []
    k_values = range(1, max_k + 1)
    
    for k in k_values:
        kmeans = KMeans(k=k, random_state=42)
        kmeans.fit(X)
        wcss_values.append(kmeans.calculate_wcss(X))
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, wcss_values, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.show()
    
    return k_values, wcss_values

# Find optimal k
k_values, wcss_values = elbow_method(X, max_k=8)
```

---

## 🌳 Hierarchical Clustering

### Two Approaches

#### 1. Agglomerative (Bottom-up)
- Start: Each point is its own cluster
- Repeatedly merge closest clusters
- Stop: When desired number of clusters reached

#### 2. Divisive (Top-down)
- Start: All points in one cluster
- Repeatedly split clusters
- Less common due to computational complexity

### Linkage Criteria

**How to measure distance between clusters:**

1. **Single Linkage:** Minimum distance between any two points
2. **Complete Linkage:** Maximum distance between any two points
3. **Average Linkage:** Average distance between all pairs
4. **Ward Linkage:** Minimize within-cluster variance

### Implementation and Visualization

```python
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Generate sample data
X, _ = make_blobs(n_samples=50, centers=3, cluster_std=1.0, random_state=42)

# Perform hierarchical clustering
linkage_matrix = linkage(X, method='ward')

# Plot dendrogram
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1])
plt.title('Original Data')

plt.subplot(1, 3, 2)
dendrogram(linkage_matrix, truncate_mode='level', p=3)
plt.title('Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')

# Apply clustering with 3 clusters
agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg_clustering.fit_predict(X)

plt.subplot(1, 3, 3)
colors = ['red', 'blue', 'green']
for i in range(3):
    cluster_points = X[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                c=colors[i], alpha=0.7, label=f'Cluster {i}')
plt.title('Hierarchical Clustering Results')
plt.legend()
plt.tight_layout()
plt.show()
```

---

## 📊 Principal Component Analysis (PCA)

### The Curse of Dimensionality

**Problems with high-dimensional data:**
- **Visualization:** Can't plot more than 3 dimensions
- **Computation:** Algorithms slow down
- **Storage:** Requires more memory
- **Noise:** More features may include irrelevant information

### PCA Intuition

**Goal:** Find the directions (principal components) that capture the most variance in the data.

**Key Ideas:**
- **First PC:** Direction of maximum variance
- **Second PC:** Direction of second most variance (orthogonal to first)
- **Continue:** Until all variance is captured

### Mathematical Foundation

**Steps:**
1. **Standardize** data (mean=0, std=1)
2. **Compute covariance matrix** C = (1/n)XᵀX
3. **Find eigenvalues and eigenvectors** of C
4. **Sort** eigenvectors by eigenvalue (descending)
5. **Project** data onto top k eigenvectors

### Implementation from Scratch

```python
class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        
    def fit(self, X):
        """Fit PCA to data"""
        # Standardize data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store components and explained variance
        self.components = eigenvectors[:, :self.n_components]
        self.eigenvalues = eigenvalues
        self.explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
        
        return self
    
    def transform(self, X):
        """Transform data to lower dimensions"""
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_reduced):
        """Transform back to original dimensions"""
        return np.dot(X_reduced, self.components.T) + self.mean

# Example: Reduce Iris dataset from 4D to 2D
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

# Apply PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Plot results
plt.figure(figsize=(15, 5))

# Original data (first 2 features)
plt.subplot(1, 3, 1)
for i, target_name in enumerate(iris.target_names):
    plt.scatter(X[y == i, 0], X[y == i, 1], 
                alpha=0.8, label=target_name)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Original Data (First 2 Features)')
plt.legend()

# PCA results
plt.subplot(1, 3, 2)
for i, target_name in enumerate(iris.target_names):
    plt.scatter(X_reduced[y == i, 0], X_reduced[y == i, 1], 
                alpha=0.8, label=target_name)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Results (4D → 2D)')
plt.legend()

# Explained variance
plt.subplot(1, 3, 3)
plt.bar(range(1, len(pca.explained_variance_ratio) + 1), 
        pca.explained_variance_ratio)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Component')
plt.tight_layout()
plt.show()

print(f"Explained variance ratio: {pca.explained_variance_ratio}")
print(f"Total variance explained: {np.sum(pca.explained_variance_ratio):.2%}")
```

---

## 📈 Clustering Evaluation Metrics

### Internal Metrics (No ground truth needed)

#### 1. Silhouette Score
**Measures:** How similar points are to their own cluster vs. other clusters

**Formula:** s = (b - a) / max(a, b)
- a = average distance to points in same cluster
- b = average distance to points in nearest cluster
- Range: [-1, 1], higher is better

```python
from sklearn.metrics import silhouette_score, silhouette_samples

def plot_silhouette_analysis(X, labels, n_clusters):
    """Plot silhouette analysis"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Silhouette plot
    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)
    
    y_lower = 10
    for i in range(n_clusters):
        cluster_silhouette_values = sample_silhouette_values[labels == i]
        cluster_silhouette_values.sort()
        
        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                         0, cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_xlabel('Silhouette Coefficient Values')
    ax1.set_ylabel('Cluster Label')
    ax1.set_title(f'Silhouette Plot (avg score: {silhouette_avg:.3f})')
    
    # Cluster plot
    colors = plt.cm.nipy_spectral(labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors)
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.set_title('Clustering Results')
    
    plt.tight_layout()
    plt.show()

# Example usage
kmeans = KMeans(k=3)
kmeans.fit(X)
plot_silhouette_analysis(X, kmeans.labels, 3)
```

#### 2. Inertia/WCSS
**Within-Cluster Sum of Squares** - lower is better

#### 3. Calinski-Harabasz Index
**Ratio of between-cluster to within-cluster variance** - higher is better

### External Metrics (Ground truth available)

#### 1. Adjusted Rand Index (ARI)
**Measures similarity between predicted and true clusters**
- Range: [-1, 1], higher is better
- Adjusted for chance

#### 2. Normalized Mutual Information (NMI)
**Measures mutual information between clusters**
- Range: [0, 1], higher is better

---

## 💻 Practical Applications

### Customer Segmentation Example

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# Simulate customer data
np.random.seed(42)
n_customers = 1000

# Generate synthetic customer data
data = {
    'annual_spending': np.random.normal(50000, 15000, n_customers),
    'frequency_of_purchases': np.random.poisson(12, n_customers),
    'avg_order_value': np.random.normal(150, 50, n_customers),
    'customer_age': np.random.normal(40, 12, n_customers),
    'loyalty_score': np.random.uniform(0, 10, n_customers)
}

# Create DataFrame
df = pd.DataFrame(data)
df = df[df['annual_spending'] > 0]  # Remove negative spending
df = df[df['avg_order_value'] > 0]   # Remove negative order values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Apply k-means clustering
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Analyze clusters
cluster_summary = df.groupby('cluster').agg({
    'annual_spending': ['mean', 'std'],
    'frequency_of_purchases': ['mean', 'std'],
    'avg_order_value': ['mean', 'std'],
    'customer_age': ['mean', 'std'],
    'loyalty_score': ['mean', 'std']
}).round(2)

print("Cluster Summary:")
print(cluster_summary)

# Visualize clusters (using PCA for 2D visualization)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
colors = ['red', 'blue', 'green', 'purple']
for i in range(4):
    cluster_points = X_pca[df['cluster'] == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                c=colors[i], alpha=0.7, label=f'Cluster {i}')

plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Customer Segmentation Results')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 🤔 Discussion Questions

1. **k-means vs. Hierarchical:** When would you choose k-means over hierarchical clustering and vice versa?

2. **PCA Interpretation:** How do you decide how many principal components to keep? What's the trade-off?

3. **Clustering Validation:** Without ground truth labels, how can you determine if your clustering results are meaningful?

4. **Curse of Dimensionality:** How does high dimensionality affect clustering algorithms differently?

---

## 🔍 Looking Ahead

Unsupervised learning helps us understand data structure without labels. Next week we'll explore how AI techniques are applied in practice and survey modern topics!

**Practical Assignment:**
1. Apply k-means clustering to a real dataset and find the optimal number of clusters
2. Use PCA to visualize a high-dimensional dataset in 2D
3. Compare different clustering algorithms on the same dataset
4. Implement a simple recommendation system using clustering