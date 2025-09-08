#!/usr/bin/env python3
"""
Week 7: Machine Learning Fundamentals
This module demonstrates core ML concepts with implementations from scratch.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import seaborn as sns
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
np.random.seed(42)


class LinearRegressionScratch:
    """Linear Regression implemented from scratch using gradient descent."""
    
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.trained = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the linear regression model."""
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0
        self.cost_history = []
        
        # Gradient descent
        for epoch in range(self.epochs):
            # Forward pass: make predictions
            y_pred = self.predict(X)
            
            # Compute cost (Mean Squared Error)
            cost = np.mean((y - y_pred) ** 2)
            self.cost_history.append(cost)
            
            # Compute gradients
            dw = (2/n_samples) * np.dot(X.T, (y_pred - y))
            db = (2/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        self.trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if not self.trained and self.weights is None:
            raise ValueError("Model must be trained before making predictions")
        return np.dot(X, self.weights) + self.bias
    
    def plot_cost_history(self) -> None:
        """Plot the cost function during training."""
        if not self.cost_history:
            print("No training history available")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title('Cost Function During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error')
        plt.grid(True)
        plt.show()


class LogisticRegressionScratch:
    """Logistic Regression implemented from scratch."""
    
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.trained = False
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the logistic regression model."""
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0
        self.cost_history = []
        
        # Gradient descent
        for epoch in range(self.epochs):
            # Forward pass
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(linear_pred)
            
            # Compute cost (Cross-entropy loss)
            # Add small epsilon to prevent log(0)
            epsilon = 1e-15
            predictions = np.clip(predictions, epsilon, 1 - epsilon)
            cost = -np.mean(y * np.log(predictions) + (1-y) * np.log(1-predictions))
            self.cost_history.append(cost)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        self.trained = True
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        linear_pred = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_pred)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Make binary predictions."""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)


class DatasetGenerator:
    """Generate synthetic datasets for demonstration."""
    
    @staticmethod
    def generate_regression_data(n_samples: int = 100, noise: float = 10, 
                               n_features: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Generate regression dataset."""
        X, y = make_regression(n_samples=n_samples, n_features=n_features, 
                              noise=noise, random_state=42)
        return X, y
    
    @staticmethod
    def generate_classification_data(n_samples: int = 100, n_features: int = 2,
                                   n_classes: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """Generate classification dataset."""
        X, y = make_classification(n_samples=n_samples, n_features=n_features,
                                 n_classes=n_classes, n_clusters_per_class=1,
                                 random_state=42)
        return X, y


def demonstrate_bias_variance_tradeoff():
    """Demonstrate bias-variance tradeoff with polynomial regression."""
    print("=" * 60)
    print("BIAS-VARIANCE TRADEOFF DEMONSTRATION")
    print("=" * 60)
    
    # Generate true function: y = x^2 + noise
    np.random.seed(42)
    X_true = np.linspace(-1, 1, 100).reshape(-1, 1)
    y_true = X_true.ravel() ** 2
    
    # Generate noisy training data
    n_train = 20
    X_train = np.random.uniform(-1, 1, n_train).reshape(-1, 1)
    y_train = X_train.ravel() ** 2 + np.random.normal(0, 0.1, n_train)
    
    # Test different polynomial degrees
    degrees = [1, 2, 9]
    
    plt.figure(figsize=(15, 5))
    
    for i, degree in enumerate(degrees):
        plt.subplot(1, 3, i+1)
        
        # Create polynomial features
        X_train_poly = np.column_stack([X_train**j for j in range(1, degree+1)])
        X_true_poly = np.column_stack([X_true**j for j in range(1, degree+1)])
        
        # Train model
        model = LinearRegressionScratch(learning_rate=0.01, epochs=2000)
        model.fit(X_train_poly, y_train)
        
        # Make predictions
        y_pred = model.predict(X_true_poly)
        
        # Plot results
        plt.scatter(X_train, y_train, alpha=0.7, label='Training Data')
        plt.plot(X_true, y_true, 'r--', label='True Function', linewidth=2)
        plt.plot(X_true, y_pred, 'g-', label=f'Degree {degree} Fit', linewidth=2)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Polynomial Degree {degree}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Calculate training error
        y_train_pred = model.predict(X_train_poly)
        train_mse = mean_squared_error(y_train, y_train_pred)
        
        # Calculate test error on true function
        test_mse = mean_squared_error(y_true, y_pred)
        
        # Add error information
        plt.text(0.05, 0.95, f'Train MSE: {train_mse:.3f}\nTest MSE: {test_mse:.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Interpretation
        if degree == 1:
            plt.text(0.05, 0.05, 'HIGH BIAS\n(Underfitting)', 
                    transform=plt.gca().transAxes, color='red', fontweight='bold')
        elif degree == 2:
            plt.text(0.05, 0.05, 'GOOD FIT', 
                    transform=plt.gca().transAxes, color='green', fontweight='bold')
        else:
            plt.text(0.05, 0.05, 'HIGH VARIANCE\n(Overfitting)', 
                    transform=plt.gca().transAxes, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def demonstrate_gradient_descent():
    """Visualize gradient descent optimization."""
    print("\n" + "=" * 60)
    print("GRADIENT DESCENT VISUALIZATION")
    print("=" * 60)
    
    # Generate simple 1D regression data
    X, y = DatasetGenerator.generate_regression_data(n_samples=50, noise=5, n_features=1)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model and track progress
    model = LinearRegressionScratch(learning_rate=0.1, epochs=100)
    
    # Modified fit method to save intermediate parameters
    n_samples, n_features = X_scaled.shape
    model.weights = np.random.normal(0, 0.01, n_features)
    model.bias = 0
    
    weight_history = []
    bias_history = []
    
    for epoch in range(model.epochs):
        # Forward pass
        y_pred = model.predict(X_scaled)
        
        # Compute cost
        cost = np.mean((y - y_pred) ** 2)
        model.cost_history.append(cost)
        
        # Save current parameters
        weight_history.append(model.weights[0])
        bias_history.append(model.bias)
        
        # Compute gradients
        dw = (2/n_samples) * np.dot(X_scaled.T, (y_pred - y))
        db = (2/n_samples) * np.sum(y_pred - y)
        
        # Update parameters
        model.weights -= model.learning_rate * dw
        model.bias -= model.learning_rate * db
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Cost function over time
    axes[0].plot(model.cost_history)
    axes[0].set_title('Cost Function During Training')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Mean Squared Error')
    axes[0].grid(True)
    
    # Plot 2: Parameter evolution
    axes[1].plot(weight_history, label='Weight')
    axes[1].plot(bias_history, label='Bias')
    axes[1].set_title('Parameter Evolution')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Parameter Value')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot 3: Final fit
    X_plot = np.linspace(X_scaled.min(), X_scaled.max(), 100).reshape(-1, 1)
    y_plot = model.predict(X_plot)
    
    axes[2].scatter(X_scaled, y, alpha=0.7, label='Data Points')
    axes[2].plot(X_plot, y_plot, 'r-', linewidth=2, label='Fitted Line')
    axes[2].set_title('Final Linear Fit')
    axes[2].set_xlabel('X (scaled)')
    axes[2].set_ylabel('Y')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Final weight: {model.weights[0]:.3f}")
    print(f"Final bias: {model.bias:.3f}")
    print(f"Final cost: {model.cost_history[-1]:.3f}")


def compare_learning_rates():
    """Compare different learning rates in gradient descent."""
    print("\n" + "=" * 60)
    print("LEARNING RATE COMPARISON")
    print("=" * 60)
    
    # Generate data
    X, y = DatasetGenerator.generate_regression_data(n_samples=100, noise=10)
    X = StandardScaler().fit_transform(X)
    
    learning_rates = [0.001, 0.01, 0.1, 1.0]
    
    plt.figure(figsize=(15, 10))
    
    for i, lr in enumerate(learning_rates):
        plt.subplot(2, 2, i+1)
        
        # Train model
        model = LinearRegressionScratch(learning_rate=lr, epochs=200)
        model.fit(X, y)
        
        # Plot cost history
        plt.plot(model.cost_history)
        plt.title(f'Learning Rate: {lr}')
        plt.xlabel('Epoch')
        plt.ylabel('Cost')
        plt.grid(True)
        
        # Add final cost information
        final_cost = model.cost_history[-1]
        plt.text(0.7, 0.9, f'Final Cost: {final_cost:.2f}', 
                transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Determine convergence behavior
        if lr <= 0.01:
            behavior = "Too slow"
        elif lr == 0.1:
            behavior = "Good"
        else:
            behavior = "Unstable/Too fast"
        
        plt.text(0.7, 0.8, behavior, transform=plt.gca().transAxes,
                color='green' if behavior == 'Good' else 'red',
                fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def demonstrate_train_val_test_split():
    """Demonstrate proper data splitting and evaluation."""
    print("\n" + "=" * 60)
    print("TRAIN/VALIDATION/TEST SPLIT DEMONSTRATION")
    print("=" * 60)
    
    # Generate classification data
    X, y = DatasetGenerator.generate_classification_data(n_samples=1000, n_features=2)
    
    # Split data: 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Dataset sizes:")
    print(f"Training: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Standardize features (fit on training data only!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)  # Don't fit again!
    X_test_scaled = scaler.transform(X_test)  # Don't fit again!
    
    # Train models with different complexities (different learning rates as proxy)
    learning_rates = [0.01, 0.1, 1.0, 10.0]
    results = []
    
    for lr in learning_rates:
        # Train model
        model = LogisticRegressionScratch(learning_rate=lr, epochs=1000)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate on all sets
        train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
        val_acc = accuracy_score(y_val, model.predict(X_val_scaled))
        test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
        
        results.append({
            'learning_rate': lr,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc
        })
    
    # Display results
    print(f"\n{'Learning Rate':<15} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12} {'Status'}")
    print("-" * 65)
    
    best_val_acc = max(results, key=lambda x: x['val_acc'])
    
    for result in results:
        lr = result['learning_rate']
        train_acc = result['train_acc']
        val_acc = result['val_acc']
        test_acc = result['test_acc']
        
        # Determine status
        if abs(train_acc - val_acc) > 0.1:
            status = "Overfitting"
        elif val_acc < 0.7:
            status = "Underfitting"
        elif result == best_val_acc:
            status = "BEST MODEL"
        else:
            status = "OK"
        
        print(f"{lr:<15} {train_acc:<12.3f} {val_acc:<12.3f} {test_acc:<12.3f} {status}")
    
    print(f"\nBest model (LR={best_val_acc['learning_rate']}) final test accuracy: {best_val_acc['test_acc']:.3f}")
    
    # Visualize the results
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Accuracy comparison
    plt.subplot(1, 2, 1)
    lrs = [r['learning_rate'] for r in results]
    train_accs = [r['train_acc'] for r in results]
    val_accs = [r['val_acc'] for r in results]
    test_accs = [r['test_acc'] for r in results]
    
    plt.plot(lrs, train_accs, 'o-', label='Training', linewidth=2)
    plt.plot(lrs, val_accs, 's-', label='Validation', linewidth=2)
    plt.plot(lrs, test_accs, '^-', label='Test', linewidth=2)
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.title('Model Performance vs Learning Rate')
    plt.legend()
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Decision boundary for best model
    plt.subplot(1, 2, 2)
    best_model = LogisticRegressionScratch(learning_rate=best_val_acc['learning_rate'], epochs=1000)
    best_model.fit(X_train_scaled, y_train)
    
    # Create a mesh for decision boundary
    h = 0.1
    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = best_model.predict_proba(mesh_points)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
    scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], 
                         c=y_train, cmap='RdYlBu', edgecolors='black')
    plt.colorbar(scatter)
    plt.title(f'Best Model Decision Boundary\n(LR={best_val_acc["learning_rate"]})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()


def main():
    """Run all demonstrations."""
    print("Week 7: Machine Learning Fundamentals")
    print("Comprehensive demonstration of core ML concepts")
    
    # Run demonstrations
    demonstrate_bias_variance_tradeoff()
    demonstrate_gradient_descent()
    compare_learning_rates()
    demonstrate_train_val_test_split()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS:")
    print("1. Bias-variance tradeoff is fundamental to ML model selection")
    print("2. Gradient descent requires proper learning rate tuning")
    print("3. Always use separate train/validation/test sets")
    print("4. Feature scaling improves optimization convergence")
    print("5. Overfitting occurs when models memorize training noise")
    print("=" * 60)


if __name__ == "__main__":
    main()