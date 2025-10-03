# Week 9 Exercises: Supervised Learning II - Advanced Models

## 🎯 Learning Goals
- Implement decision trees
- Apply ensemble methods
- Use SVMs with different kernels
- Compare model performance

## 📝 Conceptual Questions

### Question 1: Decision Tree Splits
**Explain information gain and Gini impurity. When does splitting stop?**

### Question 2: Ensemble Methods
**Compare:**
- Bagging (Random Forest)
- Boosting (AdaBoost, XGBoost)
- Stacking

## 💻 Programming Exercises

### Exercise 1: Decision Tree from Scratch
```python
class DecisionTree:
    def calculate_gini(self, y):
        # TODO: Calculate Gini impurity
        pass
    
    def find_best_split(self, X, y):
        # TODO: Find best feature and threshold
        pass
    
    def fit(self, X, y):
        # TODO: Build tree recursively
        pass
```

### Exercise 2: Random Forest Comparison
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def compare_models():
    # TODO: Compare single tree vs random forest
    # Show performance improvement
    pass
```

### Exercise 3: SVM with Different Kernels
```python
from sklearn.svm import SVC

def compare_kernels():
    # TODO: Try linear, RBF, polynomial kernels
    # Visualize decision boundaries
    pass
```

## 🧩 Challenge: Win a Kaggle Competition
Participate in a Kaggle competition using ensemble methods. Document your approach and results.

---
**Due Date: End of Week 9**
