# Week 12: AI in Practice & Selected Topics

## 📚 Learning Objectives
By the end of this week, students will be able to:
- Understand key concepts in Natural Language Processing (NLP)
- Implement basic text processing and sentiment analysis
- Comprehend the fundamentals of Reinforcement Learning
- Understand the basics of Computer Vision and image processing
- Explore modern AI applications and emerging trends
- Recognize the practical challenges of deploying AI systems

---

## 📝 Natural Language Processing (NLP)

### The Challenge of Understanding Language

**Why is NLP difficult?**
- **Ambiguity:** "I saw her duck" (verb or noun?)
- **Context dependency:** "It's hot" (temperature or spicy?)
- **Idioms and metaphors:** "Break a leg"
- **Cultural references:** Domain-specific knowledge

### Text Preprocessing Pipeline

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter

# Download required NLTK data
# nltk.download('punkt')
# nltk.download('stopwords') 
# nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text):
        """Tokenize text into words"""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """Remove common stop words"""
        return [token for token in tokens if token not in self.stop_words]
    
    def stem_tokens(self, tokens):
        """Apply stemming to tokens"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize_tokens(self, tokens):
        """Apply lemmatization to tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text, use_stemming=True):
        """Complete preprocessing pipeline"""
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stop words
        tokens = self.remove_stopwords(tokens)
        
        # Stem or lemmatize
        if use_stemming:
            tokens = self.stem_tokens(tokens)
        else:
            tokens = self.lemmatize_tokens(tokens)
        
        return tokens

# Example usage
preprocessor = TextPreprocessor()

sample_text = """
Natural Language Processing is fascinating! It's amazing how computers 
can understand human language, despite all its complexities and ambiguities.
"""

processed_tokens = preprocessor.preprocess(sample_text)
print("Processed tokens:", processed_tokens)
```

### Bag of Words and TF-IDF

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample documents
documents = [
    "I love machine learning and artificial intelligence",
    "Machine learning is a subset of artificial intelligence", 
    "Deep learning is a powerful technique in AI",
    "Natural language processing is challenging but rewarding",
    "Computer vision applications are everywhere today"
]

# Bag of Words
bow_vectorizer = CountVectorizer(stop_words='english', lowercase=True)
bow_matrix = bow_vectorizer.fit_transform(documents)

print("Bag of Words Vocabulary:")
print(bow_vectorizer.get_feature_names_out()[:10])  # First 10 words
print("\nBag of Words Matrix Shape:", bow_matrix.shape)

# TF-IDF (Term Frequency-Inverse Document Frequency)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

print("\nTF-IDF Matrix Shape:", tfidf_matrix.shape)

# Document similarity using cosine similarity
similarity_matrix = cosine_similarity(tfidf_matrix)
print("\nDocument Similarity Matrix:")
print(similarity_matrix.round(3))

# Find most similar documents
for i, doc in enumerate(documents):
    similarities = similarity_matrix[i]
    most_similar_idx = np.argsort(similarities)[-2]  # -2 because -1 is the document itself
    print(f"Doc {i} most similar to Doc {most_similar_idx} (similarity: {similarities[most_similar_idx]:.3f})")
```

### Simple Sentiment Analysis

```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Simulate sentiment data (in practice, you'd use real datasets like IMDB reviews)
np.random.seed(42)

# Create synthetic sentiment data
positive_words = ['great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'perfect', 'brilliant']
negative_words = ['terrible', 'awful', 'horrible', 'hate', 'worst', 'disgusting', 'disappointing', 'useless']
neutral_words = ['okay', 'fine', 'average', 'normal', 'standard', 'typical', 'regular', 'common']

def generate_synthetic_review(sentiment, length=10):
    """Generate a synthetic review based on sentiment"""
    if sentiment == 'positive':
        words = positive_words + ['good', 'nice', 'happy', 'satisfied']
    elif sentiment == 'negative':
        words = negative_words + ['bad', 'poor', 'sad', 'disappointed'] 
    else:  # neutral
        words = neutral_words + ['product', 'service', 'experience', 'item']
    
    # Add some random common words
    common_words = ['the', 'is', 'was', 'very', 'really', 'quite', 'somewhat', 'pretty']
    words.extend(common_words)
    
    review = ' '.join(np.random.choice(words, size=length, replace=True))
    return review

# Generate synthetic dataset
n_samples = 1000
sentiments = ['positive', 'negative', 'neutral']
data = []

for sentiment in sentiments:
    for _ in range(n_samples // 3):
        review = generate_synthetic_review(sentiment)
        data.append({'text': review, 'sentiment': sentiment})

df = pd.DataFrame(data)
print("Dataset shape:", df.shape)
print("\nSentiment distribution:")
print(df['sentiment'].value_counts())

# Prepare data for machine learning
X = df['text']
y = df['sentiment']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train models
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

for name, model in models.items():
    # Train
    model.fit(X_train_vec, y_train)
    
    # Predict
    y_pred = model.predict(X_test_vec)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {accuracy:.3f}")
    print(f"Classification Report:")
    print(classification_report(y_test, y_pred))
```

### Word Embeddings (Brief Introduction)

```python
# Simple word co-occurrence based embeddings
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

def create_cooccurrence_matrix(documents, window_size=2):
    """Create word co-occurrence matrix"""
    # Get vocabulary
    all_words = []
    for doc in documents:
        words = doc.lower().split()
        all_words.extend(words)
    
    vocab = list(set(all_words))
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    vocab_size = len(vocab)
    
    # Create co-occurrence matrix
    cooccur_matrix = np.zeros((vocab_size, vocab_size))
    
    for doc in documents:
        words = doc.lower().split()
        for i, word in enumerate(words):
            if word in word_to_idx:
                word_idx = word_to_idx[word]
                
                # Look at words in window
                start = max(0, i - window_size)
                end = min(len(words), i + window_size + 1)
                
                for j in range(start, end):
                    if i != j and words[j] in word_to_idx:
                        context_idx = word_to_idx[words[j]]
                        cooccur_matrix[word_idx][context_idx] += 1
    
    return cooccur_matrix, vocab, word_to_idx

# Create embeddings using SVD
cooccur_matrix, vocab, word_to_idx = create_cooccurrence_matrix(documents)

# Reduce dimensionality to create embeddings
svd = TruncatedSVD(n_components=10, random_state=42)
word_embeddings = svd.fit_transform(cooccur_matrix)

print("Word embedding shape:", word_embeddings.shape)
print("Sample words and their embeddings:")
for i, word in enumerate(vocab[:5]):
    print(f"{word}: {word_embeddings[i][:3]}...")  # Show first 3 dimensions
```

---

## 🎮 Reinforcement Learning (RL)

### The RL Framework

**Key Components:**
- **Agent:** The learner/decision maker
- **Environment:** The world the agent interacts with
- **State (s):** Current situation of the agent
- **Action (a):** What the agent can do
- **Reward (r):** Feedback from environment
- **Policy (π):** Strategy for choosing actions

**Goal:** Learn a policy that maximizes cumulative reward.

### Simple Grid World Example

```python
import numpy as np
import matplotlib.pyplot as plt

class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.start_state = (0, 0)
        self.goal_state = (size-1, size-1)
        self.current_state = self.start_state
        
        # Define rewards
        self.rewards = np.full((size, size), -1)  # Small negative reward for each step
        self.rewards[self.goal_state] = 10  # Large positive reward for goal
        
        # Define possible actions: up, down, left, right
        self.actions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        self.action_names = ['right', 'left', 'up', 'down']
    
    def reset(self):
        """Reset environment to start state"""
        self.current_state = self.start_state
        return self.current_state
    
    def step(self, action):
        """Take action and return next state, reward, done"""
        row, col = self.current_state
        d_row, d_col = self.actions[action]
        
        # Calculate new position
        new_row = max(0, min(self.size-1, row + d_row))
        new_col = max(0, min(self.size-1, col + d_col))
        
        self.current_state = (new_row, new_col)
        reward = self.rewards[new_row, new_col]
        done = (self.current_state == self.goal_state)
        
        return self.current_state, reward, done
    
    def get_valid_actions(self, state):
        """Get list of valid actions from given state"""
        return list(range(len(self.actions)))  # All actions always valid (walls block movement)

# Q-Learning Algorithm
class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, 
                 discount_factor=0.95, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        
        # Initialize Q-table
        self.q_table = np.zeros((state_size, state_size, action_size))
    
    def get_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            # Explore: choose random action
            return np.random.choice(self.action_size)
        else:
            # Exploit: choose best known action
            row, col = state
            return np.argmax(self.q_table[row, col])
    
    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table using Q-learning update rule"""
        row, col = state
        next_row, next_col = next_state
        
        # Q-learning update rule
        current_q = self.q_table[row, col, action]
        max_next_q = np.max(self.q_table[next_row, next_col])
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        
        self.q_table[row, col, action] = new_q

# Training the agent
env = GridWorld(size=4)
agent = QLearningAgent(state_size=4, action_size=4)

# Training parameters
n_episodes = 1000
max_steps_per_episode = 100

# Track training progress
episode_rewards = []
episode_lengths = []

for episode in range(n_episodes):
    state = env.reset()
    total_reward = 0
    steps = 0
    
    for step in range(max_steps_per_episode):
        # Choose action
        action = agent.get_action(state)
        
        # Take action
        next_state, reward, done = env.step(action)
        
        # Update Q-table
        agent.update_q_table(state, action, reward, next_state)
        
        # Update tracking variables
        state = next_state
        total_reward += reward
        steps += 1
        
        if done:
            break
    
    episode_rewards.append(total_reward)
    episode_lengths.append(steps)
    
    # Decay epsilon (reduce exploration over time)
    if episode % 100 == 0:
        agent.epsilon = max(0.01, agent.epsilon * 0.95)

# Plot training progress
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(episode_rewards)
plt.title('Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

plt.subplot(1, 2, 2)
plt.plot(episode_lengths)
plt.title('Episode Lengths')
plt.xlabel('Episode')
plt.ylabel('Steps to Goal')
plt.tight_layout()
plt.show()

# Visualize learned policy
def visualize_policy(agent, env):
    policy_arrows = np.full((env.size, env.size), '', dtype=object)
    arrow_symbols = ['→', '←', '↑', '↓']
    
    for row in range(env.size):
        for col in range(env.size):
            if (row, col) == env.goal_state:
                policy_arrows[row, col] = 'G'
            else:
                best_action = np.argmax(agent.q_table[row, col])
                policy_arrows[row, col] = arrow_symbols[best_action]
    
    print("Learned Policy:")
    for row in policy_arrows:
        print(' '.join(f'{cell:>2}' for cell in row))

visualize_policy(agent, env)
```

---

## 👁️ Computer Vision Basics

### Image Processing Fundamentals

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load MNIST dataset (subset for efficiency)
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist.data[:1000], mnist.target[:1000].astype(int)

# Reshape for visualization
X_images = X.reshape(-1, 28, 28)

def show_images(images, labels, n_images=10):
    """Display a grid of images"""
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(min(n_images, len(images))):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(f'Label: {labels[i]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

show_images(X_images, y)

# Basic image operations
def apply_filters(image):
    """Apply basic image filters"""
    # Gaussian blur (simple approximation)
    from scipy.ndimage import gaussian_filter
    blurred = gaussian_filter(image, sigma=1.0)
    
    # Edge detection (Sobel filter approximation)
    def sobel_filter(img):
        # Simple edge detection using gradients
        grad_x = np.gradient(img, axis=1)
        grad_y = np.gradient(img, axis=0)
        edges = np.sqrt(grad_x**2 + grad_y**2)
        return edges
    
    edges = sobel_filter(image)
    
    return blurred, edges

# Apply filters to sample image
sample_image = X_images[0]
blurred, edges = apply_filters(sample_image)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(sample_image, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(blurred, cmap='gray')
plt.title('Blurred')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(edges, cmap='gray')
plt.title('Edges')
plt.axis('off')
plt.tight_layout()
plt.show()
```

### Feature Extraction and Classification

```python
# Extract simple features from images
def extract_features(images):
    """Extract simple statistical features from images"""
    features = []
    
    for image in images:
        # Statistical features
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)
        
        # Edge density (sum of gradients)
        grad_x = np.gradient(image, axis=1)
        grad_y = np.gradient(image, axis=0)
        edge_density = np.sum(np.sqrt(grad_x**2 + grad_y**2))
        
        # Symmetry (compare left and right halves)
        left_half = image[:, :14]
        right_half = np.fliplr(image[:, 14:])
        symmetry = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
        
        features.append([mean_intensity, std_intensity, edge_density, symmetry])
    
    return np.array(features)

# Extract features
image_features = extract_features(X_images)
print("Feature shape:", image_features.shape)

# Classify digits using extracted features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(
    image_features, y, test_size=0.2, random_state=42
)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy using hand-crafted features: {accuracy:.3f}")

# Compare with PCA features
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X)

X_train_pca, X_test_pca, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42
)

clf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
clf_pca.fit(X_train_pca, y_train)

y_pred_pca = clf_pca.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test, y_pred_pca)
print(f"Accuracy using PCA features: {accuracy_pca:.3f}")
```

---

## 🚀 Modern AI Applications and Trends

### Large Language Models (LLMs)

**Key Concepts:**
- **Transformers:** Attention-based architecture
- **Pre-training:** Learn language patterns from massive text
- **Fine-tuning:** Adapt to specific tasks
- **Prompt Engineering:** Crafting inputs for desired outputs

**Applications:**
- Text generation (GPT models)
- Translation (Google Translate)
- Code generation (GitHub Copilot)
- Conversational AI (ChatGPT, Claude)

### Generative AI

**Text-to-Image Models:**
- DALL-E, Midjourney, Stable Diffusion
- Learn to generate images from text descriptions

**Text-to-Video Models:**
- Sora, RunwayML
- Generate videos from text prompts

**Applications:**
- Creative content generation
- Rapid prototyping
- Personalized media

### Edge AI and IoT

**Trends:**
- Running AI models on mobile devices
- Real-time processing without cloud
- Privacy-preserving computation

**Examples:**
- Smartphone cameras (portrait mode, object detection)
- Smart home devices (voice recognition)
- Autonomous vehicles (sensor processing)

---

## 💼 Practical Challenges in AI Deployment

### Model Lifecycle Management

```python
# Simple example of model versioning and monitoring
import joblib
import json
from datetime import datetime

class ModelManager:
    def __init__(self, model_name):
        self.model_name = model_name
        self.models = {}
        self.performance_log = []
    
    def save_model(self, model, version, metrics):
        """Save model with version and metadata"""
        filename = f"{self.model_name}_v{version}.pkl"
        joblib.dump(model, filename)
        
        metadata = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'filename': filename
        }
        
        self.models[version] = metadata
        print(f"Model version {version} saved with accuracy: {metrics.get('accuracy', 'N/A')}")
    
    def load_model(self, version):
        """Load specific model version"""
        if version in self.models:
            filename = self.models[version]['filename']
            return joblib.load(filename)
        else:
            raise ValueError(f"Model version {version} not found")
    
    def get_best_model(self, metric='accuracy'):
        """Get the best performing model"""
        best_version = max(self.models.keys(), 
                          key=lambda v: self.models[v]['metrics'].get(metric, 0))
        return self.load_model(best_version), best_version
    
    def log_prediction_performance(self, version, predictions, actuals):
        """Log performance of model predictions"""
        accuracy = np.mean(predictions == actuals)
        
        log_entry = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy,
            'n_predictions': len(predictions)
        }
        
        self.performance_log.append(log_entry)
        return accuracy

# Example usage
manager = ModelManager("digit_classifier")

# Save different model versions
manager.save_model(clf, version=1, metrics={'accuracy': accuracy})
manager.save_model(clf_pca, version=2, metrics={'accuracy': accuracy_pca})

# Get best model
best_model, best_version = manager.get_best_model()
print(f"Best model is version {best_version}")
```

### Data Quality and Bias

```python
def analyze_data_quality(X, y):
    """Analyze potential data quality issues"""
    print("Data Quality Analysis:")
    print(f"Dataset shape: {X.shape}")
    print(f"Missing values: {np.isnan(X).sum()}")
    print(f"Duplicate rows: {X.shape[0] - len(np.unique(X, axis=0))}")
    
    # Check class balance
    from collections import Counter
    class_counts = Counter(y)
    print(f"Class distribution: {dict(class_counts)}")
    
    # Check for outliers (simple method)
    outlier_scores = np.sum(np.abs(X - np.mean(X, axis=0)) > 3 * np.std(X, axis=0), axis=1)
    n_outliers = np.sum(outlier_scores > 0)
    print(f"Potential outliers: {n_outliers} ({n_outliers/len(X)*100:.1f}%)")

analyze_data_quality(X, y)
```

---

## 🤔 Discussion Questions

1. **AI Applications:** Which AI application do you think will have the biggest impact on society in the next 5 years?

2. **Technical Challenges:** What are the biggest technical challenges in deploying AI systems in production?

3. **Interdisciplinary AI:** How do different fields (psychology, linguistics, neuroscience) contribute to AI development?

4. **Future Trends:** What emerging AI trends do you find most promising or concerning?

---

## 🔍 Looking Ahead

This week provided a taste of various AI applications and modern developments. Next week we'll address the critical ethical considerations in AI systems.

**Practical Assignment:**
1. Build a simple chatbot using rule-based NLP techniques
2. Implement a basic recommendation system using collaborative filtering
3. Create a computer vision pipeline for object detection in images
4. Research and present on a modern AI application of your choice