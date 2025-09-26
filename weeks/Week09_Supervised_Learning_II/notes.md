# Week 9: Supervised Learning II - Advanced Models

## 📚 Learning Objectives
By the end of this week, students will be able to:
- Understand and implement decision trees from scratch
- Apply impurity measures (Gini, Entropy) for optimal splits
- Comprehend the concept of ensemble methods and their benefits
- Implement and use Random Forests for classification and regression
- Evaluate and prevent overfitting in tree-based models
- Compare and contrast different supervised learning approaches

---

## 🌳 Decision Trees

### Intuitive and Interpretable Learning

**Decision Trees** mimic human decision-making by asking a series of questions to reach a conclusion.

**Key Advantages:**
- **Interpretable:** Easy to understand and visualize
- **No assumptions:** Works with any type of data
- **Handles both numerical and categorical features**
- **Feature selection:** Automatically identifies important features

**Example Decision Tree (Play Tennis?):**
```
Outlook = ?
├── Sunny
│   └── Humidity = ?
│       ├── High → No
│       └── Normal → Yes
├── Overcast → Yes
└── Rainy
    └── Windy = ?
        ├── True → No
        └── False → Yes
```

### How Decision Trees Work

**Training Process:**
1. Start with all training examples at root
2. Choose the best feature to split on
3. Create branches for each value of that feature
4. Recursively repeat for each branch
5. Stop when stopping criteria are met

**Key Question:** How do we choose the "best" feature to split on?

---

## 📊 Impurity Measures

### Measuring Information Gain

**Goal:** Choose splits that create the most "pure" child nodes.

### Entropy (Information Theory)

**Definition:** Entropy measures disorder/randomness in a set.

**Formula:**
Entropy(S) = -Σᵢ pᵢ log₂(pᵢ)

Where pᵢ is the proportion of examples belonging to class i.

**Examples:**
- **Pure set** (all same class): Entropy = 0
- **50/50 split**: Entropy = 1 (maximum for binary)
- **Mixed but uneven**: 0 < Entropy < 1

**Calculation Example:**
- Dataset: 9 Yes, 5 No (14 total)
- p(Yes) = 9/14, p(No) = 5/14
- Entropy = -(9/14)log₂(9/14) - (5/14)log₂(5/14) ≈ 0.940

### Gini Impurity

**Alternative measure** often faster to compute:

**Formula:**
Gini(S) = 1 - Σᵢ pᵢ²

**Same Example:**
- Gini = 1 - (9/14)² - (5/14)² ≈ 0.459

### Information Gain

**Definition:** Reduction in entropy after splitting on an attribute.

**Formula:**
Gain(S, A) = Entropy(S) - Σᵥ (|Sᵥ|/|S|) × Entropy(Sᵥ)

Where:
- S = original set
- A = attribute we're considering
- Sᵥ = subset where attribute A has value v

**Algorithm:** Choose the attribute with highest information gain.

---

## 🛠️ Decision Tree Implementation

### Building a Tree from Scratch

```python
import numpy as np
from collections import Counter
import math

class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Feature index to split on
        self.threshold = threshold  # Threshold for split (for continuous features)
        self.left = left           # Left subtree
        self.right = right         # Right subtree
        self.value = value         # Value if leaf node

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, criterion='entropy'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
    
    def entropy(self, y):
        """Calculate entropy of a label array"""
        if len(y) == 0:
            return 0
        
        counts = Counter(y)
        probabilities = [count/len(y) for count in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
        return entropy
    
    def gini(self, y):
        """Calculate Gini impurity of a label array"""
        if len(y) == 0:
            return 0
        
        counts = Counter(y)
        probabilities = [count/len(y) for count in counts.values()]
        gini = 1 - sum(p**2 for p in probabilities)
        return gini
    
    def impurity(self, y):
        """Calculate impurity based on chosen criterion"""
        if self.criterion == 'entropy':
            return self.entropy(y)
        elif self.criterion == 'gini':
            return self.gini(y)
    
    def information_gain(self, X_column, y, threshold):
        """Calculate information gain from a split"""
        parent_impurity = self.impurity(y)
        
        # Split data
        left_mask = X_column <= threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0  # No gain if one side is empty
        
        # Calculate weighted average of child impurities
        n = len(y)
        n_left, n_right = np.sum(left_mask), np.sum(right_mask)
        left_impurity = self.impurity(y[left_mask])
        right_impurity = self.impurity(y[right_mask])
        
        child_impurity = (n_left/n) * left_impurity + (n_right/n) * right_impurity
        
        return parent_impurity - child_impurity
    
    def best_split(self, X, y):
        """Find the best feature and threshold to split on"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature in range(n_features):
            X_column = X[:, feature]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                gain = self.information_gain(X_column, y, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def build_tree(self, X, y, depth=0):
        """Recursively build the decision tree"""
        # Stopping criteria
        if (depth >= self.max_depth or 
            len(y) < self.min_samples_split or 
            len(np.unique(y)) == 1):
            # Create leaf node
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionTreeNode(value=leaf_value)
        
        # Find best split
        feature, threshold, gain = self.best_split(X, y)
        
        if gain == 0:  # No more information gain
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionTreeNode(value=leaf_value)
        
        # Split the data
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        # Recursively build subtrees
        left_subtree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return DecisionTreeNode(feature, threshold, left_subtree, right_subtree)
    
    def fit(self, X, y):
        """Train the decision tree"""
        self.root = self.build_tree(X, y)
    
    def predict_sample(self, x, node):
        """Predict a single sample"""
        if node.value is not None:  # Leaf node
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self.predict_sample(x, node.left)
        else:
            return self.predict_sample(x, node.right)
    
    def predict(self, X):
        """Predict multiple samples"""
        return [self.predict_sample(x, self.root) for x in X]
```

---

## 🌲 Ensemble Methods: The Wisdom of Crowds

### Why Combine Multiple Models?

**Single Model Limitations:**
- May overfit to training data
- Sensitive to small changes in data
- May have high variance

**Ensemble Benefits:**
- **Reduced overfitting:** Averaging reduces variance
- **Improved accuracy:** Multiple perspectives
- **Increased robustness:** Less sensitive to outliers

### Types of Ensemble Methods

1. **Bagging (Bootstrap Aggregating):**
   - Train multiple models on different subsets of data
   - Combine predictions by voting/averaging
   - Example: Random Forest

2. **Boosting:**
   - Train models sequentially, each focusing on previous errors
   - Examples: AdaBoost, Gradient Boosting

3. **Stacking:**
   - Train a meta-model to combine base model predictions

---

## 🌲 Random Forest

### Bagging + Feature Randomness

**Random Forest Algorithm:**
1. For each tree (typically 100-500 trees):
   - **Bootstrap sampling:** Randomly sample training data with replacement
   - **Feature randomness:** At each split, consider only √(total features) random features
   - Build decision tree normally
2. **Prediction:**
   - **Classification:** Majority vote
   - **Regression:** Average predictions

### Key Advantages

- **Reduces overfitting** compared to single decision trees
- **Handles missing values** naturally
- **Provides feature importance** rankings
- **Robust to outliers**
- **Parallel training** (trees are independent)

### Implementation

```python
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from collections import Counter

class RandomForest:
    def __init__(self, n_trees=100, max_depth=10, min_samples_split=2, 
                 max_features='sqrt'):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        self.feature_importances_ = None
    
    def bootstrap_sample(self, X, y):
        """Create bootstrap sample of the data"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]
    
    def get_random_features(self, n_features):
        """Get random subset of features"""
        if self.max_features == 'sqrt':
            n_random_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            n_random_features = int(np.log2(n_features))
        else:
            n_random_features = min(self.max_features, n_features)
        
        return np.random.choice(n_features, size=n_random_features, replace=False)
    
    def fit(self, X, y):
        """Train the random forest"""
        self.trees = []
        n_features = X.shape[1]
        
        for _ in range(self.n_trees):
            # Create bootstrap sample
            X_bootstrap, y_bootstrap = self.bootstrap_sample(X, y)
            
            # Create decision tree with limited features
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            
            # For simplicity, we'll use all features here
            # In practice, you'd modify the tree to use random features at each split
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
    
    def predict(self, X):
        """Make predictions using all trees"""
        # Get predictions from all trees
        tree_predictions = []
        for tree in self.trees:
            predictions = tree.predict(X)
            tree_predictions.append(predictions)
        
        # Combine predictions (majority vote for classification)
        final_predictions = []
        for i in range(len(X)):
            sample_predictions = [pred[i] for pred in tree_predictions]
            final_prediction = Counter(sample_predictions).most_common(1)[0][0]
            final_predictions.append(final_prediction)
        
        return np.array(final_predictions)
```

---

## 📊 Model Evaluation and Comparison

### Comparing Different Models

```python
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
data = load_wine()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'k-NN': KNeighborsClassifier(n_neighbors=5)
}

# Compare models
for name, model in models.items():
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    # Train and test
    model.fit(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"{name}:")
    print(f"  CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    print(f"  Test Score: {test_score:.3f}")
    print()
```

### Feature Importance in Trees

```python
# Feature importance from Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

feature_importance = rf.feature_importances_
feature_names = data.feature_names

# Sort features by importance
indices = np.argsort(feature_importance)[::-1]

print("Feature Importance Ranking:")
for i in range(10):  # Top 10 features
    print(f"{i+1}. {feature_names[indices[i]]}: {feature_importance[indices[i]]:.3f}")
```

---

## ⚖️ Bias-Variance Tradeoff

### Understanding Model Complexity

**Bias:** Error from oversimplifying assumptions
**Variance:** Error from sensitivity to small fluctuations in training data

| Model | Bias | Variance | Notes |
|-------|------|----------|-------|
| Single Decision Tree | Low | High | Can overfit easily |
| Random Forest | Low-Medium | Low | Bagging reduces variance |
| Linear Regression | High | Low | Strong assumptions |
| k-NN (small k) | Low | High | Flexible but noisy |
| k-NN (large k) | High | Low | Smooth but rigid |

### Preventing Overfitting in Trees

**Techniques:**
1. **Limit tree depth** (`max_depth`)
2. **Minimum samples per split** (`min_samples_split`)
3. **Minimum samples per leaf** (`min_samples_leaf`)
4. **Pruning:** Remove branches that don't improve validation performance
5. **Ensemble methods:** Use multiple trees (Random Forest)

---

## 💻 Practical Examples with Scikit-learn

### Complete Pipeline Example

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_rf = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Evaluate
y_pred = best_rf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Feature importance visualization
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 8))
plt.title("Feature Importances")
plt.bar(range(10), importances[indices[:10]])
plt.xticks(range(10), [data.feature_names[i] for i in indices[:10]], rotation=45)
plt.tight_layout()
plt.show()
```

---

## 🤔 Discussion Questions

1. **Interpretability vs. Accuracy:** Random Forests are more accurate but less interpretable than single decision trees. When might you choose each?

2. **Feature Selection:** How does the automatic feature selection in decision trees compare to manual feature engineering?

3. **Ensemble Diversity:** What factors contribute to the diversity of trees in a Random Forest? Why is diversity important?

4. **Computational Complexity:** Compare the training and prediction time complexity of decision trees vs. Random Forests.

---

## 🔍 Looking Ahead

Next week we'll dive into neural networks - a completely different approach that can learn complex patterns but with less interpretability than tree-based methods.

**Practical Assignment:**
1. Implement a decision tree from scratch and compare with Scikit-learn
2. Build a Random Forest to predict house prices and analyze feature importance
3. Compare decision trees, Random Forests, and previous algorithms on a classification dataset