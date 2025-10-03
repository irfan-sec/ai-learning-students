# Week 6 Resources: Reasoning under Uncertainty

## 📖 Required Readings

### Primary Textbook: AIMA (Russell & Norvig)
- **Chapter 13:** Quantifying Uncertainty
  - Section 13.1: Acting under Uncertainty
  - Section 13.2: Basic Probability Notation
  - Section 13.3: Inference Using Full Joint Distributions
  - Section 13.4: Independence
  - Section 13.5: Bayes' Rule and Its Use
- **Chapter 14:** Probabilistic Reasoning
  - Section 14.1: Representing Knowledge in an Uncertain Domain
  - Section 14.2: The Semantics of Bayesian Networks
  - Section 14.3: Efficient Representation of Conditional Distributions
  - Section 14.4: Exact Inference in Bayesian Networks
  - Section 14.5: Approximate Inference (optional)

---

## 🎥 Video Resources

### Foundational Videos
1. **[Probability Theory Basics](https://www.youtube.com/watch?v=OyddY7DlV58)** - Khan Academy (10 min)
   - Clear explanation of probability fundamentals
   - Joint, marginal, and conditional probability

2. **[Bayes' Theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM)** - 3Blue1Brown (15 min)
   - Visual, intuitive explanation of Bayes' rule
   - Why it's so important in AI

3. **[Bayesian Networks](https://www.youtube.com/watch?v=TuGDMj43ehw)** - Bert Huang (12 min)
   - Introduction to probabilistic graphical models
   - Structure and inference

### Advanced Videos (Optional)
4. **[Probabilistic Graphical Models Course](https://www.youtube.com/playlist?list=PLoZgVqqHOumTY2CAQHL45tQp6kmDnDcqn)** - Stanford (Full course)
   - Comprehensive lecture series by Daphne Koller
   - Deep dive into theory and applications

5. **[Markov Chains Explained](https://www.youtube.com/watch?v=i3AkTO9HLXo)** - Normalized Nerd (15 min)
   - Understanding Markov models
   - Foundation for HMMs and other models

---

## 🌐 Interactive Resources

### Online Demos and Simulations
1. **[Seeing Theory - Probability](https://seeing-theory.brown.edu/basic-probability/index.html)**
   - Beautiful interactive probability visualizations
   - Explore conditional probability, Bayes' rule visually
   - Highly recommended for intuition building

2. **[Bayesian Network Editor](http://www.bnlearn.com/bnrepository/)**
   - Repository of Bayesian networks
   - Explore real-world network structures
   - Learn from examples

3. **[Medical Diagnosis Simulator](https://www.cse.unsw.edu.au/~se2011/Applets/medical.html)**
   - Interactive Bayesian inference demo
   - See how probabilities update with evidence
   - Classic AI application

### Visualization Tools
4. **[Probability Distribution Explorer](https://distribution-explorer.github.io/)**
   - Visualize different probability distributions
   - Understand parameters and properties

---

## 💻 Code Resources

### Official AIMA Code
1. **[AIMA Python Repository](https://github.com/aimacode/aima-python)**
   - Chapter 13: [`probability.py`](https://github.com/aimacode/aima-python/blob/master/probability.py)
   - Probability distributions, Bayes nets
   - Complete implementations ready to use

2. **[Probability Notebook](https://github.com/aimacode/aima-python/blob/master/probability.ipynb)**
   - Jupyter notebook with interactive examples
   - Bayes' theorem, inference examples

### Python Libraries
3. **pgmpy - Probabilistic Graphical Models**
   - `pip install pgmpy`
   - Complete Bayesian network library
   - [Documentation](https://pgmpy.org/)
   - [GitHub](https://github.com/pgmpy/pgmpy)

4. **pomegranate - Probabilistic Modeling**
   - `pip install pomegranate`
   - Bayesian networks, HMMs, and more
   - [Documentation](https://pomegranate.readthedocs.io/)

### Example Implementations
5. **[Bayesian Network Examples](https://github.com/pgmpy/pgmpy/tree/dev/examples)**
   - Real-world examples using pgmpy
   - Medical diagnosis, spam filtering, etc.

6. **[Probabilistic Programming in Python](https://github.com/pymc-devs/pymc)**
   - PyMC for Bayesian modeling
   - More advanced probabilistic programming

---

## 📚 Additional Reading Materials

### Academic Papers

1. **"Probabilistic Reasoning in Intelligent Systems"** - Judea Pearl (1988)
   - [Seminal work](https://dl.acm.org/doi/book/10.5555/52121) on Bayesian networks
   - Founded the field of probabilistic AI

2. **"Bayesian Networks Without Tears"** - Eugene Charniak (1991)
   - [Tutorial paper](https://www.aaai.org/ojs/index.php/aimagazine/article/view/918)
   - Accessible introduction to Bayesian networks

3. **"An Essay Towards Solving a Problem in the Doctrine of Chances"** - Thomas Bayes (1763)
   - [Original Bayes' theorem paper](https://royalsocietypublishing.org/doi/10.1098/rstl.1763.0053)
   - Historical significance

### Tutorials and Guides

4. **[Introduction to Bayesian Networks](https://towardsdatascience.com/introduction-to-bayesian-networks-81031eeed94e)**
   - Practical tutorial with examples
   - Code included

5. **[Bayes' Theorem Explained](https://betterexplained.com/articles/an-intuitive-and-short-explanation-of-bayes-theorem/)**
   - Intuitive explanation with examples
   - Great for building understanding

### Blog Posts and Articles

6. **[Probabilistic Programming & Bayesian Methods](https://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/)**
   - Free online book
   - Practical Bayesian methods with Python

7. **[Naive Bayes Explained](https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/)**
   - Simple yet powerful classifier
   - Real applications

---

## 📊 Datasets and Examples

### Classic Problems
1. **[Medical Diagnosis Networks](http://www.bnlearn.com/bnrepository/)**
   - Asia network (lung cancer diagnosis)
   - Heart disease, cancer, etc.
   - Real medical Bayesian networks

2. **[Weather Prediction Data](https://www.kaggle.com/datasets/budincsevity/szeged-weather)**
   - Build probabilistic weather models
   - Practice with real data

### Benchmark Networks
3. **[UAI Bayesian Network Repository](https://www.cs.huji.ac.il/project/PASCAL/)**
   - Standard test networks
   - Compare inference algorithms

4. **[Spam Email Dataset](https://archive.ics.uci.edu/ml/datasets/spambase)**
   - Build Naive Bayes spam filter
   - Classic application of probability

---

## 🔧 Tools and Software

### Python Libraries
1. **NumPy** - Numerical computing
   - `pip install numpy`
   - Essential for probability calculations
   - [Documentation](https://numpy.org/doc/)

2. **SciPy** - Scientific computing
   - `pip install scipy`
   - Statistical distributions: `scipy.stats`
   - [Documentation](https://docs.scipy.org/doc/scipy/)

3. **pgmpy** - Bayesian Networks
   - `pip install pgmpy`
   - Complete BN toolkit
   - Model creation, inference, learning

### Visualization
4. **Matplotlib/Seaborn** - Plotting
   - Visualize probability distributions
   - Plot network structures

5. **NetworkX** - Graph visualization
   - `pip install networkx`
   - Visualize Bayesian network structure
   - [Documentation](https://networkx.org/)

### Commercial Tools (Optional)
6. **[GeNIe Modeler](https://www.bayesfusion.com/genie/)**
   - Free academic Bayesian network tool
   - GUI for creating and testing networks

7. **[Netica](https://www.norsys.com/netica.html)**
   - Popular BN software
   - Free limited version available

---

## 🎯 Practical Applications

### Machine Learning
1. **Naive Bayes Classification**
   - Spam filtering
   - Sentiment analysis
   - Document classification

2. **Bayesian Optimization**
   - Hyperparameter tuning
   - Experiment design

### Medical Applications
3. **Diagnostic Systems**
   - Disease probability given symptoms
   - Treatment recommendations
   - Risk assessment

4. **Medical Imaging Analysis**
   - Probabilistic segmentation
   - Uncertainty quantification

### Natural Language Processing
5. **Language Models**
   - N-gram models
   - Text generation
   - Machine translation

6. **Speech Recognition**
   - Hidden Markov Models
   - Acoustic modeling

### Robotics
7. **Localization**
   - Particle filters
   - Probabilistic state estimation

8. **Sensor Fusion**
   - Combining multiple uncertain sensors
   - Kalman filtering

---

## 📖 Books (Supplementary)

1. **"Probabilistic Graphical Models"** - Daphne Koller & Nir Friedman
   - Comprehensive reference
   - Theory and practice

2. **"Machine Learning: A Probabilistic Perspective"** - Kevin Murphy
   - Modern ML through probabilistic lens
   - Extensive and detailed

3. **"Bayesian Reasoning and Machine Learning"** - David Barber
   - [Free PDF available](http://web4.cs.ucl.ac.uk/staff/D.Barber/pmwiki/pmwiki.php?n=Brml.HomePage)
   - Excellent algorithms and examples

4. **"Think Bayes"** - Allen B. Downey
   - [Free online](https://greenteapress.com/wp/think-bayes/)
   - Practical Bayesian statistics with Python

---

## 🏥 Real-World Case Studies

### Medical Diagnosis
1. **[QMR-DT Network](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2232994/)**
   - Quick Medical Reference - Decision Theoretic
   - 600+ diseases, 4000+ symptoms
   - Shows scale of real applications

2. **[ALARM Network](http://www.bnlearn.com/documentation/man/alarm.html)**
   - Anesthesia monitoring system
   - 37 variables, complex dependencies

### Spam Filtering
3. **[Naive Bayes for Email](https://www.paulgraham.com/spam.html)**
   - Paul Graham's influential essay
   - Practical implementation insights

### Recommendation Systems
4. **Netflix Prize**
   - Probabilistic matrix factorization
   - Handling uncertainty in preferences

---

## 🔬 Research Topics

### Inference Algorithms
1. **Variable Elimination**
   - Exact inference method
   - Dynamic programming approach

2. **Junction Tree Algorithm**
   - General exact inference
   - Complexity considerations

3. **Sampling Methods**
   - Monte Carlo methods
   - Gibbs sampling
   - Particle filtering

### Learning Bayesian Networks
4. **Structure Learning**
   - Learning network topology from data
   - Score-based and constraint-based methods

5. **Parameter Learning**
   - Maximum likelihood estimation
   - Bayesian parameter estimation

### Advanced Topics
6. **Dynamic Bayesian Networks**
   - Temporal reasoning
   - Time-series modeling

7. **Decision Networks (Influence Diagrams)**
   - Combining probability with utility
   - Decision making under uncertainty

---

## 💡 Study Tips and Best Practices

### Understanding Probability
- **Practice calculations:** Do many examples by hand
- **Visualize:** Draw Venn diagrams and probability trees
- **Check intuition:** Does the answer make sense?
- **Use real examples:** Connect to everyday situations

### Bayesian Networks
- **Start simple:** Begin with 2-3 variable networks
- **Draw it out:** Always sketch the network structure
- **Verify independence:** Check conditional independence assumptions
- **Test inference:** Calculate simple queries by hand first

### Common Mistakes to Avoid
- **Confusing P(A|B) with P(B|A):** Remember they're different!
- **Assuming independence:** Don't assume without justification
- **Ignoring normalization:** Probabilities must sum to 1
- **Wrong chain rule:** Get the conditioning right

### Debugging Probabilistic Code
- **Check probabilities:** All should be between 0 and 1
- **Verify normalization:** Distributions should sum to 1
- **Test simple cases:** Use networks you can solve by hand
- **Print intermediate values:** Track probability calculations

---

## 🔍 Related Topics to Explore

- **Hidden Markov Models (HMMs):** Sequential probabilistic models
- **Markov Decision Processes (MDPs):** Decision making over time
- **Gaussian Processes:** Continuous probabilistic models
- **Causal Inference:** Understanding cause and effect
- **Probabilistic Programming:** Modern approach to probabilistic modeling

---

## 🌟 Notable Researchers and Groups

1. **Judea Pearl** - [bayes.cs.ucla.edu](http://bayes.cs.ucla.edu/)
   - Pioneer of Bayesian networks
   - Turing Award winner

2. **Daphne Koller** - Stanford University
   - Probabilistic graphical models expert
   - Founded Coursera

3. **Michael Jordan** - UC Berkeley
   - Machine learning and probability
   - Influential researcher and teacher

---

## 🎓 Online Courses

1. **[Probabilistic Graphical Models](https://www.coursera.org/specializations/probabilistic-graphical-models)** - Coursera
   - By Daphne Koller
   - Comprehensive specialization

2. **[Introduction to Probability](https://www.edx.org/course/introduction-to-probability)** - edX/MIT
   - Solid probability foundations
   - Mathematical rigor

3. **[Bayesian Statistics](https://www.coursera.org/learn/bayesian-statistics)** - Coursera
   - Practical Bayesian methods
   - Applied focus

---

## 📱 Mobile Apps and Games

1. **[Probability Distributions](https://play.google.com/store/apps/details?id=com.snappy_app.probability)**
   - Interactive probability calculator
   - Learn distributions on the go

2. **[Bayes' Theorem Calculator](https://www.omnicalculator.com/statistics/bayes-theorem)**
   - Online calculator for Bayes' rule
   - Educational tool

---

**📌 Remember:** Probability is the language of uncertainty in AI. Bayesian networks provide a powerful framework for representing and reasoning with uncertain knowledge. Master the fundamentals of probability first, then build up to complex networks. Practice with real datasets to see the power of probabilistic reasoning!
