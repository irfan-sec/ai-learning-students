---
layout: week
title: "Week 8: Supervised Learning I - Regression & Classification"
permalink: /week08.html
---

### Navigation
[🏠 Home](index.html) | [← Previous: Week 7](week07.html) | [Next: Week 9 →](week09.html)

---

# Week 8: Supervised Learning I - Regression & Classification

## 📚 Learning Objectives
By the end of this week, students will be able to:
- Implement linear and logistic regression from scratch
- Understand and apply regularization techniques (Ridge, Lasso)
- Use k-Nearest Neighbors for classification and regression
- Build and interpret decision trees
- Compare different supervised learning algorithms

---

## 📈 Linear Regression Deep Dive

### Multiple Linear Regression
Learn to model relationships between multiple features and continuous targets.

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Example: Housing price prediction
features = ['size', 'bedrooms', 'age', 'location_score']
model = LinearRegression()
model.fit(X_train[features], y_train)

# Make predictions
predictions = model.predict(X_test[features])
```

### Regularization Techniques
- **Ridge Regression (L2):** Prevents overfitting by penalizing large coefficients
- **Lasso Regression (L1):** Feature selection through sparse coefficients
- **Elastic Net:** Combines Ridge and Lasso benefits

---

## 🎯 Logistic Regression for Classification

### Binary Classification
Transform linear regression for probability prediction using the sigmoid function.

```python
from sklearn.linear_model import LogisticRegression

# Example: Email spam detection
model = LogisticRegression()
model.fit(X_train, y_train)

# Get probabilities
probabilities = model.predict_proba(X_test)
predictions = model.predict(X_test)
```

### Multi-class Classification
Extend to multiple classes using one-vs-rest or multinomial approaches.

---

## 🏠 k-Nearest Neighbors (k-NN)

### Instance-Based Learning
Make predictions based on similarity to training examples.

```python
from sklearn.neighbors import KNeighborsClassifier

# Example: Handwritten digit recognition
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

accuracy = knn.score(X_test, y_test)
```

### Choosing k and Distance Metrics
- Cross-validation for optimal k
- Euclidean, Manhattan, and custom distance functions
- Curse of dimensionality considerations

---

## 🌳 Decision Trees

### Tree Construction
Learn hierarchical decision rules from data.

```python
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Example: Medical diagnosis
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)

# Visualize the tree
from sklearn.tree import plot_tree
plt.figure(figsize=(15, 10))
plot_tree(tree, feature_names=feature_names, class_names=class_names, filled=True)
plt.show()
```

### Preventing Overfitting
- Maximum depth limits
- Minimum samples per leaf
- Pruning techniques

---

## 💻 Hands-On Exercises

### Exercise 1: Boston Housing Regression
Compare linear regression, Ridge, and Lasso on housing data.

### Exercise 2: Iris Classification
Build multiple classifiers and compare performance.

### Exercise 3: Credit Approval Decision Tree
Create an interpretable model for loan decisions.

---

## 🔗 Curated Resources

### Essential Reading
- [Scikit-learn Documentation](https://scikit-learn.org/stable/) - Implementation guides
- [Regularization Explained](https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a)

### Video Resources
- [Linear Regression - StatQuest](https://www.youtube.com/watch?v=nk2CQITm_eo)
- [Logistic Regression - Josh Starmer](https://www.youtube.com/watch?v=yIYKR4sgzI8)

### Interactive Tools
- [Decision Tree Visualizer](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)
- [Regression Playground](https://playground.tensorflow.org/)

---

## 🤔 Discussion Questions

1. When should you use regularization and which type?
2. How do you interpret logistic regression coefficients?
3. What are the advantages and disadvantages of k-NN?
4. How do decision trees handle categorical features?

---

### Navigation
[🏠 Home](index.html) | [← Previous: Week 7](week07.html) | [Next: Week 9 →](week09.html)

---

*"The best algorithm is the one you understand and can explain to others."*