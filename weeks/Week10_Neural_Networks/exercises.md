# Week 10 Exercises: Introduction to Neural Networks

## 🎯 Learning Goals
- Implement neural network from scratch
- Use TensorFlow/Keras
- Understand backpropagation
- Apply to image classification

## 📝 Conceptual Questions

### Question 1: Neural Network Components
**Explain:**
- Weights and biases
- Activation functions (sigmoid, ReLU, softmax)
- Forward propagation
- Backpropagation

### Question 2: Architecture Design
**Design networks for:**
- Binary classification
- Multi-class classification (10 classes)
- Regression

## 💻 Programming Exercises

### Exercise 1: Neural Network from Scratch
```python
class NeuralNetwork:
    def __init__(self, layers):
        # Initialize weights and biases
        pass
    
    def forward(self, X):
        # TODO: Implement forward pass
        pass
    
    def backward(self, X, y):
        # TODO: Implement backpropagation
        pass
    
    def train(self, X, y, epochs=1000):
        # TODO: Training loop
        pass
```

### Exercise 2: MNIST with Keras
```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# TODO: Load MNIST, train, evaluate
```

### Exercise 3: Hyperparameter Tuning
```python
def tune_hyperparameters():
    # Experiment with:
    # - Number of layers
    # - Number of neurons per layer
    # - Learning rate
    # - Batch size
    # - Epochs
    pass
```

## 🧩 Challenge: CIFAR-10 Classification
Build a neural network to classify CIFAR-10 images (10 classes). Achieve >70% accuracy.

---
**Due Date: End of Week 10**
