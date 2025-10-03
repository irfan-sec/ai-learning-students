# Week 8 Exercises: Supervised Learning I - Regression & Classification

## 🎯 Learning Goals
- Implement linear and logistic regression
- Apply regularization techniques
- Evaluate classification models with proper metrics

## 📝 Conceptual Questions

### Question 1: Regression vs Classification
**When would you use each? Provide examples.**

### Question 2: Regularization
**Explain L1 vs L2 regularization and their effects on model coefficients.**

## 💻 Programming Exercises

### Exercise 1: Linear Regression from Scratch
```python
import numpy as np

class LinearRegression:
    def fit(self, X, y):
        # TODO: Implement normal equation or gradient descent
        pass
    
    def predict(self, X):
        # TODO: Return predictions
        pass
```

### Exercise 2: Logistic Regression
```python
class LogisticRegression:
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        # TODO: Implement gradient descent with logistic cost
        pass
```

### Exercise 3: Model Evaluation
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_classifier(y_true, y_pred):
    # TODO: Calculate and display all metrics
    # Include confusion matrix visualization
    pass
```

## 🧩 Challenge: Titanic Survival Prediction
Build a logistic regression model to predict Titanic survival. Include:
- Feature engineering
- Model training
- Evaluation with multiple metrics
- Analysis of feature importance

---
**Due Date: End of Week 8**
