# Week 12 Exercises: AI in Practice

## 🎯 Learning Goals
- Apply NLP techniques
- Use computer vision for images
- Build practical AI applications
- Deploy models

## 📝 Conceptual Questions

### Question 1: NLP Pipeline
**Describe:** Tokenization, Stemming/Lemmatization, Vectorization, Modeling

### Question 2: CNNs for Vision
**Why are CNNs effective for images? Explain convolution, pooling, and feature learning.**

## 💻 Programming Exercises

### Exercise 1: Sentiment Analysis
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def build_sentiment_analyzer(texts, labels):
    # TODO: Vectorize text
    # TODO: Train classifier
    # TODO: Evaluate
    pass
```

### Exercise 2: Image Classification with CNN
```python
from tensorflow import keras

def build_cnn():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3,3), activation='relu'),
        keras.layers.MaxPooling2D((2,2)),
        # TODO: Add more layers
        keras.layers.Dense(10, activation='softmax')
    ])
    return model
```

### Exercise 3: Transfer Learning
```python
def use_pretrained_model():
    # TODO: Load pre-trained model (VGG16, ResNet)
    # TODO: Fine-tune for your task
    pass
```

## 🧩 Challenge: Build a Web App
Create a web application using Flask or Streamlit that uses your ML model.

---
**Due Date: End of Week 12**
