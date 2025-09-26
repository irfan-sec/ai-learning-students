# Week 10: Introduction to Neural Networks & Deep Learning

## 📚 Learning Objectives
By the end of this week, students will be able to:
- Understand the biological inspiration and mathematical foundation of neural networks
- Implement a simple neural network from scratch using NumPy
- Comprehend the backpropagation algorithm for training neural networks
- Apply different activation functions and understand their properties
- Build neural networks using high-level libraries (Keras/TensorFlow)
- Apply neural networks to real-world classification problems like image recognition

---

## 🧠 From Biology to Computation

### Biological Inspiration (Brief Overview)

**Biological Neuron:**
- **Dendrites:** Receive signals from other neurons
- **Cell Body:** Processes incoming signals
- **Axon:** Transmits output signal to other neurons
- **Synapses:** Connections between neurons with varying strengths

**Key Insight:** Learning occurs by adjusting synaptic strengths.

### The Artificial Neuron (Perceptron)

**Mathematical Model:**
```
Output = f(Σᵢ wᵢxᵢ + b)
```

Where:
- **xᵢ** = input features
- **wᵢ** = weights (like synaptic strengths)
- **b** = bias term
- **f** = activation function

**Visual Representation:**
```
x₁ ──w₁──┐
x₂ ──w₂──┤
x₃ ──w₃──┤ Σ ──f(·)── output
  ...    │
xₙ ──wₙ──┘
     +b
```

---

## ⚡ Activation Functions

### Why Non-linearity Matters

**Without activation functions:** Neural networks would just be linear transformations (no matter how deep).

**With non-linear activations:** Networks can learn complex patterns and decision boundaries.

### Common Activation Functions

#### 1. Sigmoid
**Formula:** σ(x) = 1 / (1 + e⁻ˣ)

**Properties:**
- Output range: (0, 1)
- Smooth and differentiable
- **Problem:** Vanishing gradients for large |x|

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)
```

#### 2. ReLU (Rectified Linear Unit)
**Formula:** ReLU(x) = max(0, x)

**Properties:**
- Output range: [0, ∞)
- Computationally efficient
- Solves vanishing gradient problem
- **Problem:** "Dead neurons" (always output 0)

```python
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)
```

#### 3. Tanh (Hyperbolic Tangent)
**Formula:** tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)

**Properties:**
- Output range: (-1, 1)
- Zero-centered (better than sigmoid)
- Still suffers from vanishing gradients

```python
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2
```

---

## 🏗️ Neural Network Architecture

### Multi-Layer Perceptron (MLP)

**Structure:**
- **Input Layer:** Features (no computation)
- **Hidden Layer(s):** Where the "magic" happens
- **Output Layer:** Final predictions

**Example Architecture:**
```
Input Layer    Hidden Layer    Output Layer
   x₁ ────────────●────────────── ŷ₁
   x₂ ─────────┬──●──┬─────────── ŷ₂
   x₃ ─────────┴──●──┴───────────
   x₄ ────────────●──────────────
```

### Forward Propagation

**Process:** Compute outputs layer by layer from input to output.

**For each layer:**
1. **Linear transformation:** z = Wx + b
2. **Non-linear activation:** a = f(z)
3. Pass activated values to next layer

**Mathematical Notation:**
- z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
- a⁽ˡ⁾ = f(z⁽ˡ⁾)

Where l indicates layer number.

---

## 📈 Training Neural Networks: Backpropagation

### The Learning Problem

**Goal:** Find weights W and biases b that minimize prediction errors.

**Loss Function (for classification):**
Cross-entropy: L = -Σᵢ yᵢ log(ŷᵢ)

**Challenge:** How do we adjust weights in hidden layers?

### Backpropagation Algorithm

**Key Insight:** Use chain rule to compute gradients of loss with respect to all parameters.

**Steps:**
1. **Forward pass:** Compute predictions
2. **Compute loss:** Compare with true labels
3. **Backward pass:** Compute gradients using chain rule
4. **Update weights:** Use gradient descent

### Mathematical Details

**Output layer error:**
δ⁽ᴸ⁾ = (a⁽ᴸ⁾ - y) ⊙ f'(z⁽ᴸ⁾)

**Hidden layer error:**
δ⁽ˡ⁾ = ((W⁽ˡ⁺¹⁾)ᵀδ⁽ˡ⁺¹⁾) ⊙ f'(z⁽ˡ⁾)

**Weight gradients:**
∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾(a⁽ˡ⁻¹⁾)ᵀ

**Bias gradients:**
∂L/∂b⁽ˡ⁾ = δ⁽ˡ⁾

---

## 💻 Implementation from Scratch

### Simple 2-Layer Neural Network

```python
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights randomly (small values)
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        
        # Store for plotting
        self.costs = []
    
    def sigmoid(self, x):
        # Clip to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward_propagation(self, X):
        """Forward pass through the network"""
        # Input to hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Hidden to output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def compute_cost(self, y_true, y_pred):
        """Compute cross-entropy cost"""
        m = y_true.shape[0]
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        cost = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / m
        return cost
    
    def backward_propagation(self, X, y):
        """Backward pass to compute gradients"""
        m = X.shape[0]
        
        # Output layer gradients
        dz2 = self.a2 - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Hidden layer gradients
        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        return dW1, db1, dW2, db2
    
    def update_parameters(self, dW1, db1, dW2, db2):
        """Update weights and biases using gradient descent"""
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def train(self, X, y, epochs=1000, print_cost=False):
        """Train the neural network"""
        for i in range(epochs):
            # Forward propagation
            y_pred = self.forward_propagation(X)
            
            # Compute cost
            cost = self.compute_cost(y, y_pred)
            self.costs.append(cost)
            
            # Backward propagation
            dW1, db1, dW2, db2 = self.backward_propagation(X, y)
            
            # Update parameters
            self.update_parameters(dW1, db1, dW2, db2)
            
            # Print cost every 100 iterations
            if print_cost and i % 100 == 0:
                print(f"Cost after iteration {i}: {cost:.4f}")
    
    def predict(self, X):
        """Make predictions"""
        y_pred = self.forward_propagation(X)
        return (y_pred > 0.5).astype(int)
    
    def plot_cost(self):
        """Plot the cost function over time"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.costs)
        plt.title('Cost Function Over Time')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.show()

# Example usage: XOR problem (non-linearly separable)
# XOR truth table: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create and train neural network
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=1.0)
nn.train(X, y, epochs=5000, print_cost=True)

# Test predictions
predictions = nn.predict(X)
print("\nPredictions:")
for i in range(len(X)):
    print(f"Input: {X[i]}, True: {y[i][0]}, Predicted: {predictions[i][0]}")

# Plot cost
nn.plot_cost()
```

---

## 🔥 Deep Learning with Keras/TensorFlow

### High-Level Implementation

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build neural network
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dropout(0.3),  # Regularization to prevent overfitting
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Print model architecture
model.summary()

# Train model
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
```

---

## 🖼️ Image Classification Example: MNIST

### Handwritten Digit Recognition

```python
# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape data: flatten 28x28 images to 784-dimensional vectors
x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

# Convert labels to categorical (one-hot encoding)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Build neural network for multi-class classification
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')  # 10 classes (digits 0-9)
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_data=(x_test, y_test),
    verbose=1
)

# Evaluate model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Make predictions on a few test samples
predictions = model.predict(x_test[:5])
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test[:5], axis=1)

print("Sample Predictions:")
for i in range(5):
    print(f"True: {true_classes[i]}, Predicted: {predicted_classes[i]}")
```

---

## 🛠️ Key Concepts and Best Practices

### 1. Weight Initialization
**Problem:** Poor initialization can lead to vanishing/exploding gradients.

**Solutions:**
- **Xavier/Glorot:** Good for sigmoid/tanh
- **He initialization:** Good for ReLU
- **Random normal with small std:** General purpose

```python
# In Keras
layers.Dense(64, activation='relu', 
            kernel_initializer='he_normal')
```

### 2. Regularization Techniques

**Dropout:**
```python
layers.Dropout(0.3)  # Randomly set 30% of inputs to 0
```

**L1/L2 Regularization:**
```python
layers.Dense(64, activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.001))
```

### 3. Optimization Algorithms

**Gradient Descent variants:**
- **SGD:** Simple but slow
- **Adam:** Adaptive learning rates (most popular)
- **RMSprop:** Good for RNNs

### 4. Batch Size Effects
- **Small batches:** More noise, better generalization
- **Large batches:** Faster training, may overfit
- **Typical range:** 32-256

---

## 📊 Understanding Neural Network Behavior

### Visualization: Decision Boundaries

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# Create a 2D dataset
X, y = make_moons(n_samples=300, noise=0.3, random_state=42)

# Train neural network
model = keras.Sequential([
    layers.Dense(10, activation='relu', input_shape=(2,)),
    layers.Dense(10, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=0)

# Plot decision boundary
def plot_decision_boundary(model, X, y):
    plt.figure(figsize=(10, 8))
    
    # Create a mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # Make predictions on the mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.colorbar(scatter)
    plt.title('Neural Network Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

plot_decision_boundary(model, X, y)
```

---

## 🤔 Discussion Questions

1. **Universal Approximation:** Neural networks can theoretically approximate any continuous function. What are the practical limitations?

2. **Interpretability:** How do neural networks compare to decision trees in terms of interpretability? When does this matter?

3. **Overfitting:** What signs indicate a neural network is overfitting? How can you prevent it?

4. **Architecture Design:** How do you decide the number of hidden layers and neurons per layer?

---

## 🔍 Looking Ahead

Neural networks are the foundation of deep learning! Next week we'll explore unsupervised learning methods that can discover hidden patterns without labeled data.

**Practical Assignment:**
1. Implement a neural network from scratch and train it on the XOR problem
2. Build a neural network using Keras to classify images in the CIFAR-10 dataset
3. Experiment with different architectures, activation functions, and regularization techniques
4. Compare neural network performance with previous algorithms on a classification task