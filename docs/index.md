---
layout: home
title: "AI Learning Students - Course Overview"
permalink: /
---

# 🤖 Introduction to Artificial Intelligence
**Course Code:** CS 2250  
**Target Audience:** 3rd Semester Bachelor's students in Computer Science, Data Science, or related fields

Welcome to the **AI Learning Students Documentation**! This comprehensive educational resource provides a practical, hands-on introduction to Artificial Intelligence following a structured 14-week curriculum.

<div class="ai-showcase">
  <div class="showcase-item">
    <img src="{{ '/assets/images/ai-brain.svg' | relative_url }}" alt="AI Brain Visualization" class="showcase-image">
    <h3>🧠 Intelligent Systems</h3>
    <p>Explore how artificial minds process information and make decisions</p>
  </div>
  
  <div class="showcase-item">
    <img src="{{ '/assets/images/neural-network.svg' | relative_url }}" alt="Neural Network Architecture" class="showcase-image">
    <h3>🔗 Neural Networks</h3>
    <p>Understand the building blocks of modern AI and deep learning</p>
  </div>
  
  <div class="showcase-item">
    <img src="{{ '/assets/images/search-algorithm.svg' | relative_url }}" alt="Search Algorithms" class="showcase-image">
    <h3>🔍 Search Algorithms</h3>
    <p>Master problem-solving techniques that power intelligent agents</p>
  </div>
</div>

---

## 📖 Course Description

This course provides a broad introduction to the fundamental principles and techniques of Artificial Intelligence. Students will explore the core concepts of intelligent agents, problem-solving through search, game playing, knowledge representation, reasoning, and machine learning. The course combines theoretical foundations with practical implementation using Python programming.

---

## 🎯 Learning Objectives

<div class="learning-objectives">
Upon successful completion of this course, students will be able to:

<div class="objectives-grid">
<div class="objective-item">
  <div class="objective-icon">🎯</div>
  <div class="objective-content">
    <h4>Define AI</h4>
    <p>Understand artificial intelligence and describe the major subfields and applications</p>
  </div>
</div>

<div class="objective-item">
  <div class="objective-icon">🔍</div>
  <div class="objective-content">
    <h4>Formulate Problems</h4>
    <p>Transform real-world problems into search problems and implement algorithms to solve them</p>
  </div>
</div>

<div class="objective-item">
  <div class="objective-icon">🧠</div>
  <div class="objective-content">
    <h4>Knowledge Representation</h4>
    <p>Understand principles of knowledge representation and logical reasoning</p>
  </div>
</div>

<div class="objective-item">
  <div class="objective-icon">📊</div>
  <div class="objective-content">
    <h4>Probability Theory</h4>
    <p>Explain basic concepts of probability theory for uncertain reasoning</p>
  </div>
</div>

<div class="objective-item">
  <div class="objective-icon">🤖</div>
  <div class="objective-content">
    <h4>Machine Learning</h4>
    <p>Describe core ML concepts and implement basic supervised learning models</p>
  </div>
</div>

<div class="objective-item">
  <div class="objective-icon">⚖️</div>
  <div class="objective-content">
    <h4>AI Ethics</h4>
    <p>Recognize societal and ethical implications of AI technology</p>
  </div>
</div>
</div>
</div>

<style>
.objectives-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
  margin-top: 2rem;
}

.objective-item {
  display: flex;
  gap: 1rem;
  padding: 1.5rem;
  background: rgba(255, 255, 255, 0.7);
  border-radius: var(--border-radius);
  border-left: 4px solid var(--primary-color);
  transition: var(--transition);
}

.objective-item:hover {
  transform: translateX(4px);
  box-shadow: var(--shadow);
  background: rgba(255, 255, 255, 0.9);
}

.objective-icon {
  font-size: 2rem;
  min-width: 3rem;
  text-align: center;
}

.objective-content h4 {
  color: var(--primary-color);
  margin: 0 0 0.5rem 0;
  font-size: 1.1rem;
}

.objective-content p {
  margin: 0;
  font-size: 0.9rem;
  color: var(--text-secondary);
}

@media (max-width: 768px) {
  .objectives-grid {
    grid-template-columns: 1fr;
  }
}
</style>

---

## 📚 Weekly Course Navigation

<div class="course-nav">

### 🎯 Part I: Foundations and Problem-Solving
<div class="part-cards">
<div class="week-card">
  <div class="week-number">01</div>
  <div class="week-content">
    <h4><a href="week01.html">Introduction to AI and Intelligent Agents</a></h4>
    <p>What is AI? Intelligent agents and environments</p>
    <div class="week-tags">
      <span class="tag">Foundations</span>
      <span class="tag">Agents</span>
    </div>
  </div>
</div>

<div class="week-card">
  <div class="week-number">02</div>
  <div class="week-content">
    <h4><a href="week02.html">Problem-Solving by Search I</a></h4>
    <p>BFS, DFS, Uniform Cost Search</p>
    <div class="week-tags">
      <span class="tag">Search</span>
      <span class="tag">Algorithms</span>
    </div>
  </div>
</div>

<div class="week-card">
  <div class="week-number">03</div>
  <div class="week-content">
    <h4><a href="week03.html">Problem-Solving by Search II</a></h4>
    <p>A*, Heuristics, Greedy Search</p>
    <div class="week-tags">
      <span class="tag">Informed Search</span>
      <span class="tag">Heuristics</span>
    </div>
  </div>
</div>

<div class="week-card">
  <div class="week-number">04</div>
  <div class="week-content">
    <h4><a href="week04.html">Adversarial Search</a></h4>
    <p>Minimax, Alpha-Beta Pruning</p>
    <div class="week-tags">
      <span class="tag">Game Theory</span>
      <span class="tag">Minimax</span>
    </div>
  </div>
</div>
</div>

### 🧠 Part II: Knowledge, Reasoning, and Uncertainty
<div class="part-cards">
<div class="week-card">
  <div class="week-number">05</div>
  <div class="week-content">
    <h4><a href="week05.html">Knowledge Representation & Reasoning</a></h4>
    <p>Propositional and First-Order Logic</p>
    <div class="week-tags">
      <span class="tag">Logic</span>
      <span class="tag">Knowledge</span>
    </div>
  </div>
</div>

<div class="week-card">
  <div class="week-number">06</div>
  <div class="week-content">
    <h4><a href="week06.html">Reasoning under Uncertainty</a></h4>
    <p>Probability, Bayes' Rule, Bayesian Networks</p>
    <div class="week-tags">
      <span class="tag">Probability</span>
      <span class="tag">Bayes</span>
    </div>
  </div>
</div>
</div>

### 🤖 Part III: Introduction to Machine Learning
<div class="part-cards">
<div class="week-card">
  <div class="week-number">07</div>
  <div class="week-content">
    <h4><a href="week07.html">Machine Learning Fundamentals</a></h4>
    <p>Types of learning, training/validation/test sets</p>
    <div class="week-tags">
      <span class="tag">ML Basics</span>
      <span class="tag">Data</span>
    </div>
  </div>
</div>

<div class="week-card">
  <div class="week-number">08</div>
  <div class="week-content">
    <h4><a href="week08.html">Supervised Learning I</a></h4>
    <p>Linear models, decision trees</p>
    <div class="week-tags">
      <span class="tag">Supervised</span>
      <span class="tag">Regression</span>
    </div>
  </div>
</div>

<div class="week-card">
  <div class="week-number">09</div>
  <div class="week-content">
    <h4><a href="week09.html">Supervised Learning II</a></h4>
    <p>SVM, ensemble methods, neural networks</p>
    <div class="week-tags">
      <span class="tag">Advanced ML</span>
      <span class="tag">Neural Nets</span>
    </div>
  </div>
</div>

<div class="week-card">
  <div class="week-number">10</div>
  <div class="week-content">
    <h4><a href="week10.html">Introduction to Neural Networks</a></h4>
    <p>Perceptrons, backpropagation, CNNs</p>
    <div class="week-tags">
      <span class="tag">Deep Learning</span>
      <span class="tag">CNNs</span>
    </div>
  </div>
</div>

<div class="week-card">
  <div class="week-number">11</div>
  <div class="week-content">
    <h4><a href="week11.html">Unsupervised Learning</a></h4>
    <p>Clustering, dimensionality reduction</p>
    <div class="week-tags">
      <span class="tag">Clustering</span>
      <span class="tag">PCA</span>
    </div>
  </div>
</div>
</div>

### 🚀 Part IV: Advanced Topics and Applications
<div class="part-cards">
<div class="week-card">
  <div class="week-number">12</div>
  <div class="week-content">
    <h4><a href="week12.html">Reinforcement Learning</a></h4>
    <p>Q-learning, policy gradients</p>
    <div class="week-tags">
      <span class="tag">RL</span>
      <span class="tag">Agents</span>
    </div>
  </div>
</div>

<div class="week-card">
  <div class="week-number">13</div>
  <div class="week-content">
    <h4><a href="week13.html">Natural Language Processing</a></h4>
    <p>Text processing, language models</p>
    <div class="week-tags">
      <span class="tag">NLP</span>
      <span class="tag">Language</span>
    </div>
  </div>
</div>

<div class="week-card">
  <div class="week-number">14</div>
  <div class="week-content">
    <h4><a href="week14.html">AI Ethics & Future Directions</a></h4>
    <p>Ethics, bias, future of AI</p>
    <div class="week-tags">
      <span class="tag">Ethics</span>
      <span class="tag">Future</span>
    </div>
  </div>
</div>
</div>

</div>

<style>
.part-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.5rem;
  margin: 2rem 0 3rem 0;
}

.week-card {
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.7));
  backdrop-filter: blur(10px);
  border-radius: var(--border-radius-lg);
  padding: 1.5rem;
  box-shadow: var(--shadow);
  transition: var(--transition);
  position: relative;
  overflow: hidden;
  display: flex;
  gap: 1rem;
  border: 1px solid rgba(99, 102, 241, 0.1);
}

.week-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: var(--primary-gradient);
  transform: scaleX(0);
  transform-origin: left;
  transition: transform 0.4s ease;
}

.week-card:hover {
  transform: translateY(-4px);
  box-shadow: var(--shadow-xl);
  border-color: var(--primary-color);
}

.week-card:hover::before {
  transform: scaleX(1);
}

.week-number {
  font-size: 2rem;
  font-weight: 700;
  background: var(--primary-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  min-width: 3rem;
  text-align: center;
}

.week-content {
  flex: 1;
}

.week-content h4 {
  margin: 0 0 0.5rem 0;
  font-size: 1.1rem;
}

.week-content h4 a {
  color: var(--text-primary);
  text-decoration: none;
  transition: var(--transition);
}

.week-content h4 a:hover {
  color: var(--primary-color);
}

.week-content p {
  font-size: 0.9rem;
  color: var(--text-secondary);
  margin: 0 0 1rem 0;
}

.week-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.tag {
  background: rgba(99, 102, 241, 0.1);
  color: var(--primary-color);
  padding: 0.25rem 0.5rem;
  border-radius: 1rem;
  font-size: 0.75rem;
  font-weight: 500;
  border: 1px solid rgba(99, 102, 241, 0.2);
}

@media (max-width: 768px) {
  .part-cards {
    grid-template-columns: 1fr;
  }
}

/* AI Showcase Styles */
.ai-showcase {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 2rem;
  margin: 3rem 0;
  padding: 2rem 0;
}

.showcase-item {
  text-align: center;
  padding: 2rem;
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
  border-radius: var(--border-radius-lg);
  border: 1px solid rgba(255, 255, 255, 0.1);
  transition: var(--transition);
  backdrop-filter: blur(10px);
}

.showcase-item:hover {
  transform: translateY(-8px);
  box-shadow: var(--shadow-xl);
  border-color: var(--primary-color);
}

.showcase-image {
  width: 200px;
  height: 150px;
  object-fit: cover;
  border-radius: var(--border-radius);
  margin-bottom: 1.5rem;
  transition: transform 0.3s ease;
}

.showcase-item:hover .showcase-image {
  transform: scale(1.05);
}

.showcase-item h3 {
  color: var(--primary-color);
  margin-bottom: 1rem;
  font-size: 1.3rem;
}

.showcase-item p {
  color: rgba(255, 255, 255, 0.8);
  margin: 0;
  line-height: 1.6;
}

@media (max-width: 768px) {
  .ai-showcase {
    grid-template-columns: 1fr;
    gap: 1.5rem;
  }
  
  .showcase-item {
    padding: 1.5rem;
  }
  
  .showcase-image {
    width: 150px;
    height: 120px;
  }
}
</style>
- **[Week 8: Supervised Learning I - Regression & Classification](week08.html)** - Linear models, decision trees
- **[Week 9: Supervised Learning II - Advanced Models](week09.html)** - SVM, ensemble methods, neural networks
- **[Week 10: Introduction to Neural Networks & Deep Learning](week10.html)** - Perceptrons, backpropagation, CNNs
- **[Week 11: Unsupervised Learning](week11.html)** - Clustering, dimensionality reduction

### Part IV: Modern Topics and Conclusion
- **[Week 12: AI in Practice & Selected Topics](week12.html)** - NLP, Computer Vision, Real-world applications
- **[Week 13: Ethics in AI](week13.html)** - Bias, fairness, transparency, societal impact
- **[Week 14: Course Review, Project Presentations, and Future of AI](week14.html)** - Integration, projects, future directions

---

## 📋 Prerequisites

- **Programming Fundamentals** (preferably Python)
- **Introductory Calculus**
- **Basic Linear Algebra**
- **Introductory Statistics**

---

## 🛠️ Required Tools & Textbooks

### Programming Environment
- **Programming Language:** Python 3.x
- **Key Libraries:** NumPy, Pandas, Matplotlib, Scikit-learn
- **Optional Libraries:** TensorFlow/PyTorch for Neural Networks week
- **Development:** Jupyter Notebook/Lab recommended

### Required Textbooks
- **Primary:** [Artificial Intelligence: A Modern Approach (AIMA)](https://aima.cs.berkeley.edu/) by Stuart Russell and Peter Norvig (4th or 3rd Edition)
- **Supplementary:** [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) by Aurélien Géron

---

## 🚀 Getting Started

1. **Choose your week:** Click on any week above to start learning
2. **Follow the structure:** Each week contains clear explanations, examples, and exercises
3. **Practice coding:** Use the provided Python code snippets and exercises
4. **Explore resources:** Check out curated links and additional materials
5. **Join the community:** Contribute improvements via our [contribution guidelines](https://github.com/irfan-sec/ai-learning-students/blob/main/CONTRIBUTING.md)

---

## 📈 What Each Week Contains

- 📝 **Core Concepts** - Clear explanations of major topics with visual aids
- 🖼️ **Visual Learning** - Embedded images and diagrams for better understanding
- 💻 **Code Examples** - Python snippets and complete implementations
- 🔗 **Curated Resources** - Official docs, videos, papers, and interactive tools
- 💡 **Exercises** - Practice problems and discussion prompts
- 🧪 **Labs** - Hands-on programming assignments

---

**Happy Learning!** 🚀  
*"The best way to learn AI is to build, experiment, and ask questions."*

---

**Course Maintainers:** AI Learning Community  
**Last Updated:** 2024  
**License:** MIT License - feel free to use this content for educational purposes

---

### Navigation
**Next:** [Week 1: Introduction to AI and Intelligent Agents →](week01.html)