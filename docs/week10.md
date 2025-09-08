---
layout: default
title: "Week 10: Introduction to Neural Networks & Deep Learning"
permalink: /week10.html
---

### Navigation
[🏠 Home](index.html) | [← Previous: Week 9](week09.html) | [Next: Week 11 →](week11.html)

---

# Week 10: Introduction to Neural Networks & Deep Learning

## 📚 Learning Objectives
- Understand the biological inspiration for neural networks
- Implement perceptrons and multi-layer networks
- Learn backpropagation algorithm for training
- Explore convolutional neural networks (CNNs)
- Apply neural networks to image and text problems

---

## 🧠 Neural Network Fundamentals

### The Perceptron
The building block of neural networks.

```python
import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1] + 1)
        
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                self.weights[1:] += self.learning_rate * (target - prediction) * xi
                self.weights[0] += self.learning_rate * (target - prediction)
    
    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)
    
    def activation(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]
```

### Multi-Layer Networks
Stack perceptrons to learn complex patterns.

---

## 🔄 Backpropagation

### Training Deep Networks
Learn how to compute gradients through chain rule.

```python
# Example with TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

---

## 🖼️ Convolutional Neural Networks (CNNs)

### Image Recognition
Specialized networks for visual data.

```python
# CNN for image classification
cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

---

## 🔗 Resources & Applications

### Essential Reading
- [Deep Learning Book](https://www.deeplearningbook.org/) - Comprehensive theoretical foundation
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) - Intuitive explanations

### Video Resources
- [Neural Networks - 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Deep Learning Specialization - Andrew Ng](https://www.coursera.org/specializations/deep-learning)

### Hands-On Projects
- MNIST digit recognition
- CIFAR-10 image classification
- Text generation with RNNs

---

### Navigation
[🏠 Home](index.html) | [← Previous: Week 9](week09.html) | [Next: Week 11 →](week11.html)