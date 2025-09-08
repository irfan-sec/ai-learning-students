---
layout: default
title: "Week 7: Machine Learning Fundamentals"
permalink: /week07.html
---

### Navigation
[🏠 Home](index.html) | [← Previous: Week 6](week06.html) | [Next: Week 8 →](week08.html)

---

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

![ML Overview](images/machine-learning-overview.png)

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

### Key Insight: Pattern Recognition

```python
# Traditional approach: Hard-coded rules
def detect_spam_traditional(email):
    spam_words = ['free', 'money', 'click', 'buy', 'discount']
    spam_count = sum(1 for word in spam_words if word in email.lower())
    return spam_count >= 2

# Machine learning approach: Learn from examples
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def train_spam_detector(emails, labels):
    """Learn spam detection from examples"""
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(emails)
    
    classifier = MultinomialNB()
    classifier.fit(X, labels)
    
    return vectorizer, classifier

def predict_spam_ml(email, vectorizer, classifier):
    """Use learned model to classify new email"""
    X = vectorizer.transform([email])
    prediction = classifier.predict(X)[0]
    confidence = classifier.predict_proba(X)[0].max()
    
    return prediction, confidence

# Example usage
emails = ["Free money click now!", "Meeting at 3pm today", "Buy discount pills"]
labels = ["spam", "not spam", "spam"]

vectorizer, classifier = train_spam_detector(emails, labels)
prediction, confidence = predict_spam_ml("Free discount offer", vectorizer, classifier)
print(f"Prediction: {prediction} (confidence: {confidence:.2f})")
```

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

![Supervised Learning](images/supervised-learning-example.png)

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

![RL Loop](images/reinforcement-learning-loop.png)

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

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(df):
    """Basic data exploration"""
    print("Dataset shape:", df.shape)
    print("\nData types:")
    print(df.dtypes)
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    print("\nBasic statistics:")
    print(df.describe())
    
    # Visualize distributions
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    plt.figure(figsize=(12, 8))
    for i, column in enumerate(numeric_columns[:6], 1):
        plt.subplot(2, 3, i)
        df[column].hist(bins=30)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

# Example usage with housing data
# df = pd.read_csv('housing_data.csv')
# explore_data(df)
```

### 3. Data Preprocessing
- **Clean the data:** handle missing values, remove duplicates
- **Feature engineering:** create new features, transform existing ones
- **Data encoding:** convert categorical variables to numbers
- **Normalization/Scaling:** ensure features have similar scales

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def preprocess_data(df, numeric_features, categorical_features, target):
    """Complete preprocessing pipeline"""
    
    # Separate features and target
    X = df[numeric_features + categorical_features].copy()
    y = df[target].copy()
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', sparse=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor, X, y
```

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

![ML Workflow](images/ml-workflow-diagram.png)

---

## 📈 Key Machine Learning Concepts

### Loss Functions
**Loss function** measures how wrong our model's predictions are:

#### Regression Loss Functions
```python
import numpy as np

def mean_squared_error(y_true, y_pred):
    """MSE - penalizes large errors heavily"""
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    """MAE - robust to outliers"""
    return np.mean(np.abs(y_true - y_pred))

def huber_loss(y_true, y_pred, delta=1.0):
    """Huber - combines MSE and MAE benefits"""
    residual = np.abs(y_true - y_pred)
    condition = residual <= delta
    return np.mean(
        np.where(condition, 
                0.5 * residual**2,
                delta * (residual - 0.5 * delta))
    )

# Example comparison
y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.1, 2.2, 2.8, 4.5, 10])  # Note the outlier

print(f"MSE: {mean_squared_error(y_true, y_pred):.2f}")
print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")  
print(f"Huber: {huber_loss(y_true, y_pred):.2f}")
```

#### Classification Loss Functions
```python
def zero_one_loss(y_true, y_pred):
    """0-1 Loss (accuracy-based)"""
    return np.mean(y_true != y_pred)

def cross_entropy_loss(y_true, y_pred_proba):
    """Cross-entropy for probabilistic predictions"""
    # Add small epsilon to prevent log(0)
    epsilon = 1e-15
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred_proba) + 
                   (1 - y_true) * np.log(1 - y_pred_proba))

def hinge_loss(y_true, y_pred):
    """Hinge loss for SVM"""
    return np.mean(np.maximum(0, 1 - y_true * y_pred))
```

### Gradient Descent
**Core optimization algorithm** that iteratively minimizes the loss function:

```python
class GradientDescent:
    """Gradient Descent implementation with different variants"""
    
    def __init__(self, learning_rate=0.01, max_epochs=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.cost_history = []
    
    def batch_gradient_descent(self, X, y):
        """Standard batch gradient descent"""
        m, n = X.shape
        theta = np.zeros(n)
        
        for epoch in range(self.max_epochs):
            # Forward pass
            predictions = X.dot(theta)
            cost = np.mean((predictions - y) ** 2)
            self.cost_history.append(cost)
            
            # Compute gradients
            gradients = (2/m) * X.T.dot(predictions - y)
            
            # Update parameters
            theta = theta - self.learning_rate * gradients
            
            # Check convergence
            if len(self.cost_history) > 1:
                if abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                    print(f"Converged after {epoch + 1} epochs")
                    break
        
        return theta
    
    def stochastic_gradient_descent(self, X, y):
        """Stochastic gradient descent (SGD)"""
        m, n = X.shape
        theta = np.zeros(n)
        
        for epoch in range(self.max_epochs):
            # Shuffle the data
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_cost = 0
            for i in range(m):
                # Use single example
                xi = X_shuffled[i:i+1]
                yi = y_shuffled[i:i+1]
                
                # Forward pass
                prediction = xi.dot(theta)
                cost = (prediction - yi) ** 2
                epoch_cost += cost[0]
                
                # Compute gradients
                gradient = 2 * xi.T.dot(prediction - yi)
                
                # Update parameters
                theta = theta - self.learning_rate * gradient.flatten()
            
            self.cost_history.append(epoch_cost / m)
        
        return theta
    
    def mini_batch_gradient_descent(self, X, y, batch_size=32):
        """Mini-batch gradient descent"""
        m, n = X.shape
        theta = np.zeros(n)
        
        for epoch in range(self.max_epochs):
            # Shuffle the data
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_cost = 0
            num_batches = m // batch_size
            
            for i in range(0, m, batch_size):
                # Get mini-batch
                end_idx = min(i + batch_size, m)
                X_batch = X_shuffled[i:end_idx]
                y_batch = y_shuffled[i:end_idx]
                
                # Forward pass
                predictions = X_batch.dot(theta)
                cost = np.mean((predictions - y_batch) ** 2)
                epoch_cost += cost
                
                # Compute gradients
                batch_m = X_batch.shape[0]
                gradients = (2/batch_m) * X_batch.T.dot(predictions - y_batch)
                
                # Update parameters
                theta = theta - self.learning_rate * gradients
            
            self.cost_history.append(epoch_cost / num_batches)
        
        return theta
    
    def plot_convergence(self):
        """Plot the cost function over time"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title('Cost Function During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Cost')
        plt.yscale('log')
        plt.grid(True)
        plt.show()

# Example usage
np.random.seed(42)
X = np.random.randn(1000, 3)
true_theta = np.array([2, -1, 0.5])
y = X.dot(true_theta) + 0.1 * np.random.randn(1000)

gd = GradientDescent(learning_rate=0.01, max_epochs=1000)
learned_theta = gd.batch_gradient_descent(X, y)

print(f"True parameters: {true_theta}")
print(f"Learned parameters: {learned_theta}")
print(f"Error: {np.linalg.norm(true_theta - learned_theta):.4f}")
```

---

## ⚖️ The Bias-Variance Tradeoff

### Understanding Model Complexity

![Bias-Variance](images/bias-variance-tradeoff.png)

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
from sklearn.model_selection import learning_curve

def plot_learning_curves(estimator, X, y, cv=5):
    """Plot learning curves to diagnose bias/variance"""
    
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error'
    )
    
    # Convert to positive values
    train_scores = -train_scores
    val_scores = -val_scores
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_mean - train_std, 
                     train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, val_mean - val_std,
                     val_mean + val_std, alpha=0.1, color='red')
    
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.title('Learning Curves')
    plt.grid(True)
    plt.show()
    
    # Interpretation guide
    final_gap = val_mean[-1] - train_mean[-1]
    if final_gap > 0.1:
        print("⚠️  High variance (overfitting): Large gap between curves")
        print("   Solutions: More data, regularization, simpler model")
    elif train_mean[-1] > 0.5:  # Assuming normalized error
        print("⚠️  High bias (underfitting): Both curves plateau high")
        print("   Solutions: More complex model, more features")
    else:
        print("✅ Good fit: Curves converge at reasonable error level")

# Example with polynomial regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Generate synthetic data
np.random.seed(42)
X = np.linspace(0, 1, 100).reshape(-1, 1)
y = 0.5 * X.ravel() + np.sin(2 * np.pi * X.ravel()) + 0.1 * np.random.randn(100)

# Different complexity models
models = {
    'Linear (degree=1)': Pipeline([
        ('poly', PolynomialFeatures(degree=1)),
        ('linear', LinearRegression())
    ]),
    'Quadratic (degree=2)': Pipeline([
        ('poly', PolynomialFeatures(degree=2)), 
        ('linear', LinearRegression())
    ]),
    'High-degree (degree=15)': Pipeline([
        ('poly', PolynomialFeatures(degree=15)),
        ('linear', LinearRegression())
    ])
}

for name, model in models.items():
    print(f"\n{name}:")
    plot_learning_curves(model, X, y)
```

---

## 🔄 Training, Validation, and Test Sets

### The Three-Way Split

```python
from sklearn.model_selection import train_test_split

# Split data into train/temp (80/20)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify for classification
)

# Split temp into validation/test (10/10 of original)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
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
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression

def comprehensive_cross_validation(model, X, y, problem_type='classification'):
    """Perform multiple types of cross-validation"""
    
    if problem_type == 'classification':
        cv_strategies = {
            'Stratified 5-Fold': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            'Standard 5-Fold': KFold(n_splits=5, shuffle=True, random_state=42),
            'Stratified 10-Fold': StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        }
        scoring = 'accuracy'
    else:
        cv_strategies = {
            'Standard 5-Fold': KFold(n_splits=5, shuffle=True, random_state=42),
            'Standard 10-Fold': KFold(n_splits=10, shuffle=True, random_state=42)
        }
        scoring = 'neg_mean_squared_error'
    
    results = {}
    for name, cv in cv_strategies.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        if scoring == 'neg_mean_squared_error':
            scores = -scores  # Convert to positive RMSE
            scores = np.sqrt(scores)
        
        results[name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        
        print(f"{name}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    
    return results

# Example usage
model = LogisticRegression(random_state=42)
cv_results = comprehensive_cross_validation(model, X_train, y_train)
```

---

## 💻 Hands-On Exercise: Linear Regression from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

class LinearRegressionFromScratch:
    """Complete linear regression implementation with visualization"""
    
    def __init__(self, learning_rate=0.01, max_epochs=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.weight_history = []
    
    def _add_bias_term(self, X):
        """Add bias column to feature matrix"""
        return np.c_[np.ones(X.shape[0]), X]
    
    def fit(self, X, y, method='gradient_descent'):
        """Fit the model using specified method"""
        n_samples, n_features = X.shape
        
        if method == 'gradient_descent':
            self._fit_gradient_descent(X, y)
        elif method == 'normal_equation':
            self._fit_normal_equation(X, y)
        else:
            raise ValueError("Method must be 'gradient_descent' or 'normal_equation'")
    
    def _fit_gradient_descent(self, X, y):
        """Fit using gradient descent"""
        # Initialize parameters
        n_samples, n_features = X.shape
        X_with_bias = self._add_bias_term(X)
        theta = np.zeros(n_features + 1)  # +1 for bias
        
        prev_cost = float('inf')
        
        for epoch in range(self.max_epochs):
            # Forward pass
            predictions = X_with_bias.dot(theta)
            
            # Compute cost
            cost = np.mean((predictions - y) ** 2)
            self.cost_history.append(cost)
            self.weight_history.append(theta.copy())
            
            # Compute gradients
            gradients = (2/n_samples) * X_with_bias.T.dot(predictions - y)
            
            # Update parameters
            theta = theta - self.learning_rate * gradients
            
            # Check convergence
            if abs(prev_cost - cost) < self.tolerance:
                print(f"Converged after {epoch + 1} epochs")
                break
            prev_cost = cost
        
        # Store final parameters
        self.bias = theta[0]
        self.weights = theta[1:]
    
    def _fit_normal_equation(self, X, y):
        """Fit using normal equation (closed form solution)"""
        X_with_bias = self._add_bias_term(X)
        
        # Normal equation: θ = (X^T X)^(-1) X^T y
        theta = np.linalg.inv(X_with_bias.T.dot(X_with_bias)).dot(X_with_bias.T).dot(y)
        
        self.bias = theta[0]
        self.weights = theta[1:]
        
        # Calculate final cost for comparison
        predictions = self.predict(X)
        final_cost = np.mean((predictions - y) ** 2)
        self.cost_history = [final_cost]
    
    def predict(self, X):
        """Make predictions on new data"""
        return X.dot(self.weights) + self.bias
    
    def score(self, X, y):
        """Calculate R-squared score"""
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def plot_training_progress(self):
        """Visualize training progress"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot cost history
        ax1.plot(self.cost_history)
        ax1.set_title('Training Cost Over Time')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Mean Squared Error')
        ax1.set_yscale('log')
        ax1.grid(True)
        
        # Plot weight evolution (for 1D case)
        if len(self.weight_history) > 0 and len(self.weight_history[0]) <= 3:
            weight_history = np.array(self.weight_history)
            for i in range(weight_history.shape[1]):
                label = 'bias' if i == 0 else f'weight_{i}'
                ax2.plot(weight_history[:, i], label=label)
            ax2.set_title('Parameter Evolution')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Parameter Value')
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, X, y, X_test=None, y_test=None):
        """Plot predictions vs actual (for 1D features)"""
        if X.shape[1] != 1:
            print("Plotting only available for 1D features")
            return
        
        plt.figure(figsize=(12, 5))
        
        # Training data and predictions
        plt.subplot(1, 2, 1)
        predictions = self.predict(X)
        
        plt.scatter(X[:, 0], y, alpha=0.5, label='Training Data')
        
        # Sort for smooth line
        sort_idx = np.argsort(X[:, 0])
        plt.plot(X[sort_idx, 0], predictions[sort_idx], 'r-', linewidth=2, label='Predictions')
        
        plt.xlabel('Feature Value')
        plt.ylabel('Target Value')
        plt.title(f'Training Data (R² = {self.score(X, y):.3f})')
        plt.legend()
        plt.grid(True)
        
        # Test data if provided
        if X_test is not None and y_test is not None:
            plt.subplot(1, 2, 2)
            test_predictions = self.predict(X_test)
            
            plt.scatter(X_test[:, 0], y_test, alpha=0.5, label='Test Data', color='orange')
            sort_idx = np.argsort(X_test[:, 0])
            plt.plot(X_test[sort_idx, 0], test_predictions[sort_idx], 'g-', 
                    linewidth=2, label='Predictions')
            
            plt.xlabel('Feature Value')
            plt.ylabel('Target Value')
            plt.title(f'Test Data (R² = {self.score(X_test, y_test):.3f})')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# Comprehensive example
def demonstrate_linear_regression():
    """Complete demonstration of linear regression"""
    
    # Generate synthetic dataset
    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("=== Linear Regression from Scratch ===\n")
    
    # Method 1: Gradient Descent
    print("1. Training with Gradient Descent:")
    model_gd = LinearRegressionFromScratch(learning_rate=0.1, max_epochs=1000)
    model_gd.fit(X_train_scaled, y_train, method='gradient_descent')
    
    train_r2_gd = model_gd.score(X_train_scaled, y_train)
    test_r2_gd = model_gd.score(X_test_scaled, y_test)
    print(f"   Training R²: {train_r2_gd:.4f}")
    print(f"   Test R²: {test_r2_gd:.4f}")
    
    # Method 2: Normal Equation
    print("\n2. Training with Normal Equation:")
    model_ne = LinearRegressionFromScratch()
    model_ne.fit(X_train_scaled, y_train, method='normal_equation')
    
    train_r2_ne = model_ne.score(X_train_scaled, y_train)
    test_r2_ne = model_ne.score(X_test_scaled, y_test)
    print(f"   Training R²: {train_r2_ne:.4f}")
    print(f"   Test R²: {test_r2_ne:.4f}")
    
    # Compare parameters
    print(f"\n3. Parameter Comparison:")
    print(f"   Gradient Descent - Weight: {model_gd.weights[0]:.4f}, Bias: {model_gd.bias:.4f}")
    print(f"   Normal Equation - Weight: {model_ne.weights[0]:.4f}, Bias: {model_ne.bias:.4f}")
    
    # Visualization
    model_gd.plot_training_progress()
    model_gd.plot_predictions(X_train_scaled, y_train, X_test_scaled, y_test)
    
    return model_gd, model_ne

# Run the demonstration
model_gd, model_ne = demonstrate_linear_regression()
```

---

## 🔗 Curated Resources

### Essential Reading
- [AIMA Chapter 18](https://aima.cs.berkeley.edu/) - Learning from Examples
- [Introduction to Statistical Learning](https://www.statlearning.com/) - Free comprehensive textbook
- [Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) - Advanced mathematical treatment

### Video Resources
- [Machine Learning Course - Andrew Ng](https://www.coursera.org/learn/machine-learning) - Classic introduction
- [ML Fundamentals - 3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk) - Neural networks from first principles
- [Gradient Descent - Josh Starmer](https://www.youtube.com/watch?v=sDv4f4s2SB8) - StatQuest explanation

### Interactive Learning
- [ML Playground](https://ml-playground.com/) - Visualize algorithms
- [Seeing Theory](https://seeing-theory.brown.edu/) - Interactive statistics
- [Google's AI Education](https://ai.google/education/) - Courses and tools

### Practice Problems
- [Kaggle Learn](https://www.kaggle.com/learn) - Free micro-courses
- [Machine Learning Mastery](https://machinelearningmastery.com/) - Practical tutorials
- [Papers with Code](https://paperswithcode.com/) - Latest research implementations

---

## 🎯 Key Concepts Summary

### Core Principles
- **Learning from data** instead of explicit programming
- **Generalization** to unseen examples is the goal
- **Different types** for different problem structures

### Critical Concepts
- **Bias-variance tradeoff** affects model performance
- **Training/validation/test splits** prevent overfitting
- **Cross-validation** for robust model evaluation
- **Gradient descent** as universal optimization tool

### Practical Workflow
1. **Problem formulation** and success metrics
2. **Data collection** and exploratory analysis  
3. **Preprocessing** and feature engineering
4. **Model selection** and hyperparameter tuning
5. **Evaluation** and performance analysis
6. **Deployment** and ongoing monitoring

---

## 🤔 Discussion Questions

1. When would you choose supervised vs. unsupervised learning?
2. How do you know if you have enough training data?
3. What are the ethical implications of algorithmic decision-making?
4. How does machine learning relate to traditional statistical methods?
5. What role does domain expertise play in successful ML projects?

---

## 🔍 Looking Ahead

Next week, we'll dive deeper into **supervised learning algorithms**, starting with linear and logistic regression, then exploring k-nearest neighbors and decision trees. We'll implement these algorithms and apply them to real datasets!

**Preview of Week 8:**
- Linear regression and regularization techniques
- Logistic regression for classification
- k-Nearest Neighbors algorithm
- Decision trees and feature importance

---

### Navigation
[🏠 Home](index.html) | [← Previous: Week 6](week06.html) | [Next: Week 8 →](week08.html)

---

*"Machine learning is not about the algorithm, it's about understanding your data and solving real problems."*