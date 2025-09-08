# Week 7: Machine Learning Fundamentals

## 📚 Learning Objectives
By the end of this week, students will be able to:
- Understand the core concepts and motivation behind machine learning
- Distinguish between supervised, unsupervised, and reinforcement learning
- Explain the machine learning workflow and data pipeline
- Understand key concepts: overfitting, underfitting, bias-variance tradeoff
- Implement basic gradient descent and apply it to simple problems

---

## 🤖 What is Machine Learning?

**Machine Learning (ML)** is the field of study that gives computers the ability to learn patterns from data without being explicitly programmed for every scenario.

![ML Overview](https://miro.medium.com/max/1400/1*bhFifratH9DjKqMBTeQG5A.png)

### Traditional Programming vs. Machine Learning

| Traditional Programming | Machine Learning |
|------------------------|------------------|
| Data + Program → Output | Data + Output → Program |
| Rules are hand-coded | Rules are learned |
| Limited to known scenarios | Generalizes to new scenarios |
| Deterministic behavior | Probabilistic predictions |

### Why Machine Learning Now?

1. **Big Data:** Massive datasets available (internet, sensors, digital records)
2. **Computing Power:** GPUs, cloud computing, distributed systems
3. **Better Algorithms:** Deep learning, ensemble methods, optimization
4. **Real-world Success:** Image recognition, translation, recommendations

---

## 📊 Types of Machine Learning

### 1. Supervised Learning
**Goal:** Learn a mapping from inputs to outputs using labeled training data

**Examples:**
- **Classification:** Email spam detection, image recognition, medical diagnosis
- **Regression:** House price prediction, stock forecasting, temperature prediction

**Training Data Format:**
```
Input (features) → Output (labels)
[house_size, location, age] → price
[pixel_values] → cat/dog
[email_text] → spam/not_spam
```

![Supervised Learning](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f7/Supervised_learning.svg/400px-Supervised_learning.svg.png)

### 2. Unsupervised Learning  
**Goal:** Find hidden patterns or structure in data without labeled examples

**Examples:**
- **Clustering:** Customer segmentation, gene sequencing, market research
- **Dimensionality Reduction:** Data visualization, compression, feature extraction
- **Anomaly Detection:** Fraud detection, network intrusion, quality control

**Training Data Format:**
```
Input (features only) → Find patterns
[customer_purchases] → customer_groups
[document_words] → topic_categories  
[network_traffic] → normal/anomalous
```

### 3. Reinforcement Learning
**Goal:** Learn optimal actions through interaction with environment and rewards

**Examples:**
- **Game Playing:** Chess, Go, video games
- **Robotics:** Navigation, manipulation, autonomous vehicles
- **Resource Management:** Traffic control, energy optimization

**Learning Process:**
```
State → Action → Reward → New State
Environment provides feedback through rewards/penalties
Agent learns policy to maximize long-term reward
```

![RL Loop](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Reinforcement_learning_diagram.svg/400px-Reinforcement_learning_diagram.svg.png)

---

## 🔄 The Machine Learning Workflow

### 1. Problem Definition
- **What are we trying to predict?** (target variable)
- **What type of ML problem?** (classification, regression, clustering)
- **What constitutes success?** (accuracy, precision, speed)
- **What constraints exist?** (time, interpretability, data privacy)

### 2. Data Collection & Exploration
- **Gather relevant data** from various sources
- **Explore data characteristics:** size, types, distributions
- **Identify data quality issues:** missing values, outliers, errors
- **Understand domain context** and feature meanings

### 3. Data Preprocessing
- **Clean the data:** handle missing values, remove duplicates
- **Feature engineering:** create new features, transform existing ones
- **Data encoding:** convert categorical variables to numbers
- **Normalization/Scaling:** ensure features have similar scales

### 4. Model Selection & Training
- **Choose appropriate algorithms** based on problem type and data
- **Split data:** training, validation, test sets
- **Train models** using training data
- **Tune hyperparameters** using validation data

### 5. Model Evaluation
- **Assess performance** using test data
- **Compare different models** and approaches
- **Analyze errors** and failure cases
- **Validate assumptions** about data and model

### 6. Deployment & Monitoring
- **Deploy model** to production environment
- **Monitor performance** over time
- **Retrain periodically** as new data becomes available
- **Handle concept drift** when data patterns change

![ML Workflow](https://miro.medium.com/max/1400/1*KzmIUYPmxgEHhXX7SlbP4w.png)

---

## 📈 Key Machine Learning Concepts

### Loss Functions
**Loss function** measures how wrong our model's predictions are:

#### Regression Loss Functions
```python
# Mean Squared Error (MSE)
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Mean Absolute Error (MAE)  
def mae_loss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
```

#### Classification Loss Functions
```python
# 0-1 Loss (accuracy)
def zero_one_loss(y_true, y_pred):
    return np.mean(y_true != y_pred)

# Cross-entropy Loss
def cross_entropy_loss(y_true, y_pred_proba):
    return -np.mean(y_true * np.log(y_pred_proba))
```

### Gradient Descent
**Core optimization algorithm** that iteratively minimizes the loss function:

```python
def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    # Initialize parameters
    m, n = X.shape
    theta = np.zeros(n)
    
    for epoch in range(epochs):
        # Forward pass: compute predictions
        predictions = X.dot(theta)
        
        # Compute loss
        loss = np.mean((predictions - y) ** 2)
        
        # Compute gradients
        gradients = (2/m) * X.T.dot(predictions - y)
        
        # Update parameters
        theta = theta - learning_rate * gradients
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return theta
```

**Variants:**
- **Batch GD:** Use entire dataset for each update
- **Stochastic GD:** Use single example for each update  
- **Mini-batch GD:** Use small batches (most common)

---

## ⚖️ The Bias-Variance Tradeoff

### Understanding Model Complexity

![Bias-Variance](https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Bias_and_variance_contributing_to_total_error.svg/400px-Bias_and_variance_contributing_to_total_error.svg.png)

#### High Bias (Underfitting)
- **Model is too simple** to capture underlying patterns
- **Poor performance** on both training and test data  
- **Examples:** Linear model for non-linear data
- **Solutions:** Use more complex model, add features

#### High Variance (Overfitting)
- **Model is too complex** and memorizes training noise
- **Good training performance, poor test performance**
- **Examples:** Deep neural network on small dataset
- **Solutions:** More data, regularization, simpler model

#### The Sweet Spot
- **Balance between bias and variance**
- **Good generalization** to unseen data
- **Achieved through:** Cross-validation, regularization, ensemble methods

### Detecting Overfitting/Underfitting

```python
import matplotlib.pyplot as plt

def plot_learning_curves(train_sizes, train_scores, val_scores):
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores, 'o-', label='Training Score')
    plt.plot(train_sizes, val_scores, 'o-', label='Validation Score')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score') 
    plt.legend()
    plt.title('Learning Curves')
    plt.show()
    
    # Analysis:
    # - Large gap: overfitting (high variance)
    # - Both curves plateau low: underfitting (high bias)  
    # - Curves converge high: good fit
```

---

## 🔄 Training, Validation, and Test Sets

### The Three-Way Split

```python
from sklearn.model_selection import train_test_split

# Split data into train/temp (80/20)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Split temp into validation/test (10/10 of original)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"Training set: {len(X_train)} samples")     # 80%
print(f"Validation set: {len(X_val)} samples")     # 10%  
print(f"Test set: {len(X_test)} samples")          # 10%
```

### Purpose of Each Set
- **Training Set (60-80%):** Learn model parameters
- **Validation Set (10-20%):** Tune hyperparameters, model selection
- **Test Set (10-20%):** Final unbiased performance estimate

### Cross-Validation
**Alternative to validation set** for small datasets:

```python
from sklearn.model_selection import cross_val_score, KFold

# 5-fold cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")
```

---

## 📊 Model Evaluation Metrics

### Classification Metrics

#### Confusion Matrix
```python
from sklearn.metrics import confusion_matrix, classification_report

# Example for binary classification
y_true = [0, 1, 1, 0, 1, 0, 1, 1]
y_pred = [0, 1, 1, 0, 0, 1, 1, 1]

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print("    Pred")
print("    0  1")  
print(f"0 [[{cm[0,0]}  {cm[0,1]}]]")
print(f"1 [[{cm[1,0]}  {cm[1,1]}]]")
```

#### Key Classification Metrics
```python
def calculate_metrics(tp, fp, fn, tn):
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)  # Of predicted positive, how many correct?
    recall = tp / (tp + fn)     # Of actual positive, how many found?
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return accuracy, precision, recall, f1_score
```

### Regression Metrics
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mae:.3f}")
    print(f"R²:   {r2:.3f}")
```

---

## 🛠️ Feature Engineering

### Creating Better Features

#### 1. **Polynomial Features**
```python
from sklearn.preprocessing import PolynomialFeatures

# Create x², x³, xy features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
```

#### 2. **Categorical Encoding**
```python
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# One-hot encoding for nominal categories
encoder = OneHotEncoder(sparse=False)
X_categorical_encoded = encoder.fit_transform(X_categorical)

# Label encoding for ordinal categories
label_encoder = LabelEncoder()
X_ordinal_encoded = label_encoder.fit_transform(X_ordinal)
```

#### 3. **Feature Scaling**
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization (mean=0, std=1)
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Min-Max normalization (range 0-1)
minmax = MinMaxScaler()
X_normalized = minmax.fit_transform(X)
```

### Feature Selection
```python
from sklearn.feature_selection import SelectKBest, f_regression

# Select top k features based on statistical tests
selector = SelectKBest(score_func=f_regression, k=10)
X_selected = selector.fit_transform(X, y)

# Get selected feature names
selected_features = X.columns[selector.get_support()]
```

---

## 🎯 Linear Regression Example

### Mathematical Foundation
**Linear Regression** assumes relationship: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε

**Goal:** Find parameters β that minimize sum of squared errors

### Implementation from Scratch
```python
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.epochs):
            # Forward pass
            y_pred = self.predict(X)
            
            # Compute cost (MSE)
            cost = np.mean((y - y_pred) ** 2)
            self.cost_history.append(cost)
            
            # Compute gradients
            dw = (2/n_samples) * np.dot(X.T, (y_pred - y))
            db = (2/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def plot_cost_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title('Cost Function During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error')
        plt.show()
```

---

## 🤔 Discussion Questions

1. When would you choose supervised vs. unsupervised learning?
2. How do you know if you have enough training data?
3. What are the ethical implications of algorithmic decision-making?
4. How does machine learning relate to traditional statistical methods?

---

## 🔍 Looking Ahead

Next week, we'll dive deeper into **supervised learning algorithms**, starting with linear and logistic regression, then exploring k-nearest neighbors. We'll implement these algorithms and apply them to real datasets!

---

*"Machine learning is not about the algorithm, it's about understanding your data and solving real problems."*