# Week 7 Exercises: Machine Learning Fundamentals

## 🎯 Learning Goals Review
After completing these exercises, you should be able to:
- Distinguish between different types of machine learning
- Apply the ML workflow to real problems
- Implement gradient descent from scratch
- Understand and identify overfitting/underfitting
- Evaluate models using appropriate metrics

---

## 📝 Conceptual Questions

### Question 1: ML Problem Classification
**Classify each as supervised (S), unsupervised (U), or reinforcement learning (RL):**

a) Predicting house prices from features ___
b) Grouping customers by purchasing behavior ___
c) Teaching a robot to walk ___
d) Identifying spam emails ___
e) Recommending movies (with ratings) ___
f) Finding topics in documents ___
g) Playing chess ___
h) Predicting stock prices ___

### Question 2: Bias-Variance Tradeoff
**Scenario:** You train models of increasing complexity on the same dataset.

| Model Complexity | Training Error | Test Error |
|-----------------|----------------|------------|
| Very Simple     | 25%            | 26%        |
| Simple          | 15%            | 16%        |
| Moderate        | 5%             | 7%         |
| Complex         | 2%             | 12%        |
| Very Complex    | 0.5%           | 20%        |

**Questions:**
a) Which model is underfitting? Why?
b) Which model is overfitting? Why?
c) Which model would you choose? Why?
d) How would you reduce overfitting for the complex models?

### Question 3: Training/Validation/Test Split
**Why do we need three separate datasets?**

Explain the purpose of each:
- Training set: ___
- Validation set: ___
- Test set: ___

**What happens if:**
- You tune hyperparameters on test set? ___
- You have no validation set? ___
- Your test set is too small? ___

---

## 💻 Programming Exercises

### Exercise 1: Implement Gradient Descent

```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    """
    Implement gradient descent for linear regression.
    
    Args:
        X: Feature matrix (m x n)
        y: Target vector (m x 1)
        learning_rate: Step size
        iterations: Number of iterations
    
    Returns:
        theta: Learned parameters
        cost_history: Cost at each iteration
    """
    m, n = X.shape
    theta = np.zeros((n, 1))
    cost_history = []
    
    for i in range(iterations):
        # TODO: Implement gradient descent
        # 1. Compute predictions: h = X @ theta
        # 2. Compute cost: J = (1/2m) * sum((h - y)^2)
        # 3. Compute gradients: grad = (1/m) * X.T @ (h - y)
        # 4. Update parameters: theta = theta - learning_rate * grad
        pass
    
    return theta, cost_history

# Test with synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]  # Add bias term

theta, costs = gradient_descent(X_b, y)
print(f"Learned parameters: {theta.flatten()}")
print(f"Expected: [4, 3]")

# Plot cost history
plt.plot(costs)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Gradient Descent Convergence')
plt.show()
```

### Exercise 2: Cross-Validation

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def evaluate_polynomial_models(X, y, max_degree=10):
    """
    Evaluate polynomial models of different degrees using cross-validation.
    """
    cv_scores = []
    
    for degree in range(1, max_degree + 1):
        # TODO: Create polynomial features
        # TODO: Train model
        # TODO: Compute cross-validation score
        pass
    
    # Plot results
    plt.plot(range(1, max_degree + 1), cv_scores)
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Cross-Validation Score')
    plt.title('Model Complexity vs Performance')
    plt.show()
    
    return cv_scores
```

### Exercise 3: Feature Scaling

```python
def standardize(X):
    """Standardize features to zero mean and unit variance."""
    # TODO: Implement standardization
    # mean = X.mean(axis=0)
    # std = X.std(axis=0)
    # X_scaled = (X - mean) / std
    pass

def normalize(X):
    """Normalize features to [0, 1] range."""
    # TODO: Implement min-max normalization
    pass

# Compare gradient descent with and without scaling
def compare_scaling_effect():
    """Demonstrate importance of feature scaling."""
    # TODO: Create dataset with features of different scales
    # TODO: Run gradient descent with and without scaling
    # TODO: Compare convergence speed
    pass
```

---

## 🧩 Real-World Application

### Challenge: Build an End-to-End ML Pipeline

**Task:** Create a complete ML pipeline for house price prediction.

**Requirements:**
1. Load and explore data (use sklearn's California housing dataset)
2. Handle missing values
3. Feature engineering
4. Train/validation/test split
5. Train multiple models
6. Evaluate and compare
7. Final model selection

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

def build_ml_pipeline():
    """Build complete ML pipeline."""
    # 1. Load data
    # TODO
    
    # 2. Explore data
    # TODO
    
    # 3. Preprocess
    # TODO
    
    # 4. Split data
    # TODO
    
    # 5. Train models
    # TODO
    
    # 6. Evaluate
    # TODO
    
    # 7. Select best model
    # TODO
    
    pass
```

---

## 📊 Analysis Questions

### Question 4: Learning Curves
**Interpret these learning curves:**

```
High bias (underfitting):
Training error: High, flat
Validation error: High, flat (close to training)

High variance (overfitting):
Training error: Low
Validation error: High (large gap)
```

**For each scenario, what would you do?**

### Question 5: Regularization
**Explain how regularization helps prevent overfitting:**

L1 (Lasso): ___
L2 (Ridge): ___

When would you use each?

---

## 📝 Submission Guidelines

### What to Submit:
1. Completed programming exercises with working code
2. ML pipeline for house price prediction
3. Report (500-750 words) covering:
   - Key concepts learned
   - Challenges encountered
   - Results and insights
   - Practical applications

### Due Date: End of Week 7

---

**🎯 Key Takeaway:** ML is about learning patterns from data. Understanding the fundamentals—the workflow, bias-variance tradeoff, and evaluation—is crucial before diving into complex algorithms!
