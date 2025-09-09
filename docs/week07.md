---
layout: week
title: "Week 7: Machine Learning Fundamentals"
subtitle: "Introduction to Learning from Data"
description: "Discover the foundations of machine learning, including supervised and unsupervised learning, training and testing, and the bias-variance tradeoff."
week_number: 7
total_weeks: 14
github_folder: "Week07_ML_Fundamentals"
notes_link: "https://github.com/irfan-sec/ai-learning-students/blob/main/weeks/Week07_ML_Fundamentals/notes.md"
resources_link: "https://github.com/irfan-sec/ai-learning-students/blob/main/weeks/Week07_ML_Fundamentals/resources.md"
exercises_link: "https://github.com/irfan-sec/ai-learning-students/blob/main/weeks/Week07_ML_Fundamentals/exercises.md"
code_link: "https://github.com/irfan-sec/ai-learning-students/tree/main/weeks/Week07_ML_Fundamentals/code"
prev_week:
  title: "Adversarial Search"
  url: "/week04.html"
next_week:
  title: "Ethics in AI"
  url: "/week13.html"
objectives:
  - "Understand the core concepts of machine learning"
  - "Distinguish between supervised, unsupervised, and reinforcement learning"
  - "Learn about training, validation, and testing datasets"
  - "Understand overfitting, underfitting, and the bias-variance tradeoff"
  - "Implement basic machine learning algorithms from scratch"
  - "Evaluate model performance using appropriate metrics"
key_concepts:
  - "Supervised Learning"
  - "Unsupervised Learning"
  - "Training and Test Sets"
  - "Cross-Validation"
  - "Overfitting and Underfitting"
  - "Bias-Variance Tradeoff"
  - "Feature Engineering"
  - "Model Evaluation"
---

## 🧠 Learning from Data

Welcome to the exciting world of Machine Learning! This week marks a major transition in our AI journey - from hand-crafted algorithms to systems that learn from data. We'll explore how computers can automatically improve their performance through experience, opening the door to applications that seemed impossible just decades ago.

## 📚 What You'll Learn

### Core Concepts

1. **What is Machine Learning?**
   - Learning from data vs. explicit programming
   - Pattern recognition and prediction
   - The role of statistics and probability

2. **Types of Machine Learning**
   - **Supervised Learning:** Learning with labeled examples
   - **Unsupervised Learning:** Finding patterns in unlabeled data
   - **Reinforcement Learning:** Learning through interaction and rewards

3. **The Learning Process**
   - Training, validation, and testing
   - Feature selection and engineering
   - Model selection and hyperparameter tuning

4. **Common Challenges**
   - Overfitting and underfitting
   - The bias-variance tradeoff
   - Data quality and quantity issues

## 🎯 Supervised Learning

Learning from input-output pairs:

### Classification Examples
- **Email Spam Detection:** Email → {Spam, Not Spam}
- **Image Recognition:** Image → {Cat, Dog, Bird, ...}
- **Medical Diagnosis:** Symptoms → {Disease A, Disease B, Healthy}

### Regression Examples
- **House Price Prediction:** House features → Price
- **Stock Market:** Market data → Future price
- **Weather Forecasting:** Current conditions → Temperature

### Basic Algorithm: k-Nearest Neighbors (k-NN)

```python
def knn_classify(train_data, test_point, k):
    # Calculate distances to all training points
    distances = []
    for train_point in train_data:
        dist = euclidean_distance(test_point, train_point.features)
        distances.append((dist, train_point.label))
    
    # Sort by distance and take k nearest
    distances.sort()
    nearest_k = distances[:k]
    
    # Vote: return most common label
    votes = {}
    for _, label in nearest_k:
        votes[label] = votes.get(label, 0) + 1
    
    return max(votes, key=votes.get)
```

## 🔍 Unsupervised Learning

Finding hidden patterns without labels:

### Clustering
Group similar data points together:
- **Customer Segmentation:** Group customers by behavior
- **Gene Analysis:** Identify related genes
- **Image Segmentation:** Separate objects in images

### Dimensionality Reduction
Simplify data while preserving important information:
- **Data Visualization:** Plot high-dimensional data in 2D/3D
- **Noise Reduction:** Remove irrelevant features
- **Compression:** Store data more efficiently

### Basic Algorithm: k-Means Clustering

```python
def kmeans(data, k, max_iterations=100):
    # Initialize centroids randomly
    centroids = random_sample(data, k)
    
    for _ in range(max_iterations):
        # Assign points to closest centroid
        clusters = [[] for _ in range(k)]
        for point in data:
            closest_centroid = find_closest(point, centroids)
            clusters[closest_centroid].append(point)
        
        # Update centroids
        new_centroids = []
        for cluster in clusters:
            if cluster:  # Avoid empty clusters
                centroid = compute_mean(cluster)
                new_centroids.append(centroid)
        
        # Check convergence
        if centroids == new_centroids:
            break
        centroids = new_centroids
    
    return centroids, clusters
```

## ⚖️ The Bias-Variance Tradeoff

Understanding this fundamental concept is crucial for effective ML:

### Bias
- **High Bias:** Model is too simple, misses important patterns
- **Example:** Using a linear model for complex curved data
- **Result:** Underfitting - poor performance on both training and test data

### Variance
- **High Variance:** Model is too sensitive to training data variations
- **Example:** Using a very complex model with little training data
- **Result:** Overfitting - great training performance, poor test performance

### Finding the Sweet Spot
```
Total Error = Bias² + Variance + Irreducible Error

Goal: Minimize total error by balancing bias and variance
```

## 📊 Model Evaluation

### Cross-Validation
Split data systematically to get robust performance estimates:

```python
def k_fold_cross_validation(data, labels, k, model_class):
    fold_size = len(data) // k
    scores = []
    
    for i in range(k):
        # Create train/test split for this fold
        test_start = i * fold_size
        test_end = test_start + fold_size
        
        test_data = data[test_start:test_end]
        test_labels = labels[test_start:test_end]
        
        train_data = data[:test_start] + data[test_end:]
        train_labels = labels[:test_start] + labels[test_end:]
        
        # Train and evaluate
        model = model_class()
        model.fit(train_data, train_labels)
        score = model.evaluate(test_data, test_labels)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)
```

### Common Metrics

**Classification:**
- **Accuracy:** Correct predictions / Total predictions
- **Precision:** True Positives / (True Positives + False Positives)
- **Recall:** True Positives / (True Positives + False Negatives)
- **F1-Score:** Harmonic mean of precision and recall

**Regression:**
- **Mean Squared Error (MSE):** Average squared differences
- **Root MSE (RMSE):** Square root of MSE (same units as target)
- **Mean Absolute Error (MAE):** Average absolute differences

## 🛠️ Feature Engineering

The art of preparing data for machine learning:

### Feature Selection
- Remove irrelevant or redundant features
- Use domain knowledge
- Apply statistical tests

### Feature Creation
- **Polynomial Features:** x² to capture non-linearity
- **Interaction Features:** x₁ × x₂ to capture relationships
- **Binning:** Convert continuous to categorical

### Feature Scaling
```python
# Standardization (zero mean, unit variance)
def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

# Min-max scaling (0 to 1 range)
def min_max_scale(X):
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    return (X - min_val) / (max_val - min_val)
```

## 🎮 Reinforcement Learning Preview

Learning through trial and error:

### Key Concepts
- **Agent:** The learner/decision maker
- **Environment:** What the agent interacts with
- **Actions:** What the agent can do
- **Rewards:** Feedback for actions taken
- **Policy:** Strategy for choosing actions

### Famous Applications
- **Game Playing:** AlphaGo, OpenAI Five
- **Robotics:** Learning to walk, grasp objects
- **Autonomous Driving:** Navigation and safety
- **Recommendation Systems:** Learning user preferences

## 🚀 Real-World Applications

### Healthcare
- **Drug Discovery:** Identifying promising compounds
- **Medical Imaging:** Detecting tumors, fractures
- **Personalized Treatment:** Tailoring therapy to individuals

### Technology
- **Search Engines:** Ranking web pages
- **Voice Assistants:** Understanding speech
- **Image Recognition:** Photo tagging, autonomous vehicles

### Business
- **Fraud Detection:** Identifying suspicious transactions
- **Customer Analytics:** Predicting churn, lifetime value
- **Supply Chain:** Optimizing inventory and logistics

## 💡 Best Practices

### Data Preparation
1. **Understand your data:** Explore distributions, missing values
2. **Clean thoroughly:** Handle outliers, inconsistencies
3. **Split properly:** Keep test set truly separate

### Model Development
1. **Start simple:** Baseline models first
2. **Iterate quickly:** Fast feedback loops
3. **Validate rigorously:** Use cross-validation

### Deployment Considerations
1. **Monitor performance:** Models degrade over time
2. **Plan for updates:** Retrain with new data
3. **Consider ethics:** Bias, fairness, privacy

## 🔗 Connections

- **Previous Weeks:** Search and logic provide foundation for some ML algorithms
- **Next Weeks:** Deep learning, advanced algorithms, ethical considerations
- **Throughout Course:** ML principles apply to many AI domains

---

*Ready to start your machine learning journey? Dive into the hands-on exercises and build your first learning algorithms!*