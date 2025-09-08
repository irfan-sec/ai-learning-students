---
layout: default
title: "Week 9: Supervised Learning II - Advanced Models"
permalink: /week09.html
---

### Navigation
[🏠 Home](index.html) | [← Previous: Week 8](week08.html) | [Next: Week 10 →](week10.html)

---

# Week 9: Supervised Learning II - Advanced Models

## 📚 Learning Objectives
- Understand Support Vector Machines (SVM) and kernel methods
- Implement ensemble methods: Random Forests and Gradient Boosting
- Apply Naive Bayes for text classification
- Compare and select appropriate algorithms for different problems
- Understand model interpretability and feature importance

---

## 🎯 Support Vector Machines (SVM)

### Maximum Margin Classification
Find the decision boundary that maximally separates classes.

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Example: Non-linear classification with RBF kernel
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_scaled, y_train)
```

### Kernel Trick
Handle non-linear problems by mapping to higher dimensions.

---

## 🌲 Ensemble Methods

### Random Forest
Combine multiple decision trees for better performance.

```python
from sklearn.ensemble import RandomForestClassifier

# Example: Feature importance analysis
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
importances = rf.feature_importances_
feature_ranking = sorted(zip(feature_names, importances), 
                        key=lambda x: x[1], reverse=True)
```

### Gradient Boosting
Sequentially improve predictions by learning from mistakes.

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
gb.fit(X_train, y_train)
```

---

## 📧 Naive Bayes for Text Classification

### Probabilistic Classification
Apply Bayes' theorem with strong independence assumptions.

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

# Example: Document classification
vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(documents)

nb = MultinomialNB()
nb.fit(X_text, labels)
```

---

## 🔗 Resources & Practice

### Video Resources
- [SVM Explained - StatQuest](https://www.youtube.com/watch?v=efR1C6CvhmE)
- [Random Forest - Josh Starmer](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)

### Hands-On Projects
- Image classification with SVM
- Text sentiment analysis with Naive Bayes
- Financial prediction with Random Forest

---

### Navigation
[🏠 Home](index.html) | [← Previous: Week 8](week08.html) | [Next: Week 10 →](week10.html)