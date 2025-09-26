# Week 8: Supervised Learning I - Regression & Classification

## 📚 Learning Objectives
By the end of this week, students will be able to:
- Understand the difference between regression and classification problems
- Implement linear regression with gradient descent from scratch
- Apply logistic regression for binary classification
- Use k-Nearest Neighbors (k-NN) for both regression and classification
- Evaluate model performance using appropriate metrics
- Apply these algorithms using Scikit-learn on real datasets

---

## 📊 Supervised Learning Overview

**Supervised Learning** uses labeled training data to learn a mapping from inputs to outputs.

### Key Components
- **Training Data:** {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}
- **Features (X):** Input variables/attributes
- **Labels (Y):** Target outputs we want to predict
- **Hypothesis (h):** Our learned function h: X → Y

### Two Main Types

| **Regression** | **Classification** |
|----------------|-------------------|
| Predict continuous values | Predict discrete categories |
| Output: Real numbers | Output: Class labels |
| Examples: Price, temperature | Examples: Spam/not spam, diagnosis |
| Metrics: MSE, MAE | Metrics: Accuracy, precision, recall |

---

## 📈 Linear Regression

### The Foundation of Machine Learning

**Goal:** Find the best line through data points.

**Mathematical Form:**
h(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ = θᵀx

Where:
- **θ** = parameters/weights [θ₀, θ₁, ..., θₙ]
- **x** = features [1, x₁, x₂, ..., xₙ] (1 for bias term)

### Cost Function: Mean Squared Error (MSE)

**Intuition:** Minimize the sum of squared errors between predictions and actual values.

J(θ) = (1/2m) Σᵢ₌₁ᵐ (hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²

Where:
- **m** = number of training examples
- **hθ(x⁽ⁱ⁾)** = prediction for example i
- **y⁽ⁱ⁾** = actual label for example i

### Gradient Descent Algorithm

**Idea:** Iteratively move in the direction that reduces cost.

**Algorithm:**
1. Initialize θ randomly
2. Repeat until convergence:
   - Calculate gradients: ∂J/∂θⱼ = (1/m) Σᵢ₌₁ᵐ (hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾) × xⱼ⁽ⁱ⁾
   - Update parameters: θⱼ := θⱼ - α × ∂J/∂θⱼ
   - α is the **learning rate**

### Implementation from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.costs = []
    
    def fit(self, X, y):
        # Add bias column (column of 1s)
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        
        # Initialize parameters
        self.theta = np.random.randn(X_with_bias.shape[1]) * 0.01
        
        m = X.shape[0]  # number of examples
        
        for i in range(self.n_iterations):
            # Forward pass: compute predictions
            predictions = X_with_bias.dot(self.theta)
            
            # Compute cost
            cost = (1/(2*m)) * np.sum((predictions - y)**2)
            self.costs.append(cost)
            
            # Compute gradients
            gradients = (1/m) * X_with_bias.T.dot(predictions - y)
            
            # Update parameters
            self.theta -= self.learning_rate * gradients
    
    def predict(self, X):
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        return X_with_bias.dot(self.theta)
    
    def plot_cost_history(self):
        plt.plot(self.costs)
        plt.title('Cost Function Over Time')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.show()

# Example usage
np.random.seed(42)
X = np.random.randn(100, 1) * 2
y = 3 * X.flatten() + 2 + np.random.randn(100) * 0.5  # y = 3x + 2 + noise

model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)

print(f"Learned parameters: {model.theta}")
print(f"True parameters: [2, 3]")
```

---

## 🔢 Logistic Regression

### From Linear to Logistic

**Problem with Linear Regression for Classification:**
- Linear regression can output any real number
- Classification needs probabilities between 0 and 1

**Solution:** Use the **sigmoid function** to squash outputs to [0,1]

### Sigmoid Function

σ(z) = 1 / (1 + e⁻ᶻ)

**Properties:**
- Maps any real number to [0, 1]
- σ(0) = 0.5 (decision boundary)
- Smooth and differentiable

### Logistic Regression Model

**Hypothesis:**
hθ(x) = σ(θᵀx) = 1 / (1 + e⁻θᵀˣ)

**Interpretation:**
- hθ(x) = P(y = 1 | x; θ)
- If hθ(x) ≥ 0.5, predict y = 1
- If hθ(x) < 0.5, predict y = 0

### Cost Function: Log-Likelihood

**Can't use MSE** (non-convex for logistic regression)

**Log-Likelihood Cost:**
J(θ) = -(1/m) Σᵢ₌₁ᵐ [y⁽ⁱ⁾ log(hθ(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾) log(1-hθ(x⁽ⁱ⁾))]

**Intuition:**
- If y = 1, we want hθ(x) close to 1
- If y = 0, we want hθ(x) close to 0
- Heavily penalizes confident wrong predictions

### Implementation

```python
class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.costs = []
    
    def sigmoid(self, z):
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        self.theta = np.random.randn(X_with_bias.shape[1]) * 0.01
        
        m = X.shape[0]
        
        for i in range(self.n_iterations):
            # Forward pass
            z = X_with_bias.dot(self.theta)
            predictions = self.sigmoid(z)
            
            # Compute cost (with small epsilon to prevent log(0))
            epsilon = 1e-15
            predictions = np.clip(predictions, epsilon, 1 - epsilon)
            cost = -(1/m) * np.sum(y * np.log(predictions) + 
                                 (1-y) * np.log(1-predictions))
            self.costs.append(cost)
            
            # Compute gradients
            gradients = (1/m) * X_with_bias.T.dot(predictions - y)
            
            # Update parameters
            self.theta -= self.learning_rate * gradients
    
    def predict_proba(self, X):
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        return self.sigmoid(X_with_bias.dot(self.theta))
    
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
```

---

## 👥 k-Nearest Neighbors (k-NN)

### Instance-Based Learning

**Key Idea:** "Similar inputs should have similar outputs"

**Algorithm:**
1. Store all training examples
2. For a new point, find k closest training examples
3. **Classification:** Vote among k neighbors (majority wins)
4. **Regression:** Average the k neighbors' values

### Distance Metrics

**Euclidean Distance** (most common):
d(x₁, x₂) = √(Σᵢ(x₁ᵢ - x₂ᵢ)²)

**Manhattan Distance:**
d(x₁, x₂) = Σᵢ|x₁ᵢ - x₂ᵢ|

### Implementation

```python
class KNearestNeighbors:
    def __init__(self, k=3, task='classification'):
        self.k = k
        self.task = task
    
    def fit(self, X, y):
        # Lazy learning: just store the data
        self.X_train = X
        self.y_train = y
    
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def predict(self, X):
        predictions = []
        
        for x in X:
            # Calculate distances to all training points
            distances = []
            for i, x_train in enumerate(self.X_train):
                dist = self.euclidean_distance(x, x_train)
                distances.append((dist, self.y_train[i]))
            
            # Sort by distance and get k nearest
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.k]
            
            # Make prediction based on k nearest neighbors
            if self.task == 'classification':
                # Majority vote
                k_labels = [label for _, label in k_nearest]
                prediction = max(set(k_labels), key=k_labels.count)
            else:  # regression
                # Average of k nearest values
                k_values = [value for _, value in k_nearest]
                prediction = np.mean(k_values)
            
            predictions.append(prediction)
        
        return np.array(predictions)
```

### Choosing k

**Small k (k=1):**
- More complex decision boundary
- Sensitive to noise
- Can overfit

**Large k:**
- Smoother decision boundary  
- More robust to noise
- May oversimplify

**Rule of thumb:** Try k = √n where n is the number of training examples

---

## 📊 Model Evaluation

### Regression Metrics

**Mean Squared Error (MSE):**
MSE = (1/n) Σᵢ(ŷᵢ - yᵢ)²

**Root Mean Squared Error (RMSE):**
RMSE = √MSE

**Mean Absolute Error (MAE):**
MAE = (1/n) Σᵢ|ŷᵢ - yᵢ|

**R² Score (Coefficient of Determination):**
R² = 1 - (SS_res / SS_tot)
- R² = 1: Perfect predictions
- R² = 0: Model as good as mean
- R² < 0: Model worse than mean

### Classification Metrics

**Accuracy:**
Accuracy = (TP + TN) / (TP + TN + FP + FN)

**Precision:**
Precision = TP / (TP + FP)

**Recall (Sensitivity):**
Recall = TP / (TP + FN)

**F1-Score:**
F1 = 2 × (Precision × Recall) / (Precision + Recall)

Where TP=True Positives, TN=True Negatives, FP=False Positives, FN=False Negatives

---

## 💻 Practical Examples with Scikit-learn

### House Price Prediction (Regression)

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Load dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}")
print(f"RMSE: {np.sqrt(mse):.2f}")
print(f"R² Score: {r2:.2f}")
```

### Iris Classification (k-NN)

```python
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Try different k values
k_values = [1, 3, 5, 7, 9]
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"k={k}: Accuracy = {accuracy:.3f}")
```

---

## 🤔 Discussion Questions

1. **Bias vs. Variance:** How do linear regression, logistic regression, and k-NN differ in terms of bias and variance?

2. **Feature Scaling:** Why might k-NN be more sensitive to feature scaling than linear regression?

3. **Overfitting:** Which of these algorithms is most prone to overfitting? How can we prevent it?

4. **Interpretability:** Rank these algorithms by interpretability. Why does interpretability matter?

---

## 🔍 Looking Ahead

Next week we'll explore more sophisticated models like decision trees and ensemble methods that can capture complex patterns in data!

**Practical Assignment:** 
1. Implement linear regression from scratch and compare with Scikit-learn
2. Build a binary classifier using logistic regression to predict email spam
3. Use k-NN to classify handwritten digits and experiment with different k values