# Week 6 Exercises: Reasoning under Uncertainty

## 🎯 Learning Goals Review
After completing these exercises, you should be able to:
- Calculate joint, marginal, and conditional probabilities
- Apply Bayes' theorem to real-world problems
- Construct and analyze Bayesian networks
- Perform inference in Bayesian networks
- Design probabilistic models for uncertain domains

---

## 📝 Conceptual Questions

### Question 1: Probability Fundamentals
**Given the following probability table:**

| Weather | Sprinkler | Grass Wet | P(W,S,G) |
|---------|-----------|-----------|----------|
| Sunny   | On        | Yes       | 0.15     |
| Sunny   | On        | No        | 0.01     |
| Sunny   | Off       | Yes       | 0.02     |
| Sunny   | Off       | No        | 0.22     |
| Rainy   | On        | Yes       | 0.05     |
| Rainy   | On        | No        | 0.00     |
| Rainy   | Off       | Yes       | 0.35     |
| Rainy   | Off       | No        | 0.20     |

**Calculate:**
a) P(Grass Wet = Yes)
b) P(Rainy | Grass Wet = Yes)
c) P(Sprinkler = On | Sunny)
d) Are Weather and Sprinkler independent?
e) Are Sprinkler and Grass Wet conditionally independent given Weather?

---

### Question 2: Bayes' Theorem Application
**Medical Testing Scenario:**
- Disease prevalence: P(Disease) = 0.01 (1% of population)
- Test sensitivity: P(Positive | Disease) = 0.95 (true positive rate)
- Test specificity: P(Negative | ¬Disease) = 0.90 (true negative rate)

**Tasks:**
a) Calculate P(Positive), the probability of testing positive
b) Calculate P(Disease | Positive) using Bayes' theorem
c) If someone tests positive, what's the probability they actually have the disease?
d) Why is this probability so different from the test sensitivity?
e) What would happen if disease prevalence was 10% instead of 1%?

---

### Question 3: Conditional Independence
**For each scenario, determine if conditional independence holds:**

a) **Variables:** Season, Temperature, Ice Cream Sales
   - Are Temperature and Ice Cream Sales independent given Season?
   
b) **Variables:** Study Hours, Sleep, Exam Score
   - Are Study Hours and Sleep independent?
   - Are they conditionally independent given Exam Score?

c) **Variables:** Smoking, Yellow Fingers, Lung Cancer
   - Construct a Bayesian network showing dependencies
   - Identify all conditional independence relationships

---

## 🔍 Bayesian Network Analysis

### Exercise 1: Network Construction
**Burglar Alarm Problem:**
- Variables: Burglary (B), Earthquake (E), Alarm (A), JohnCalls (J), MaryCalls (M)
- Dependencies:
  - Alarm goes off if Burglary OR Earthquake
  - John and Mary call independently if they hear the Alarm

**Tasks:**
a) Draw the Bayesian network structure (DAG)
b) List all parent-child relationships
c) Write the factored form: P(B,E,A,J,M) = ?
d) Identify all conditional independence statements
e) How many parameters needed to specify this network?

**Given CPTs:**
```
P(B) = 0.001
P(E) = 0.002

P(A | B, E):
  B=T, E=T: 0.95
  B=T, E=F: 0.94
  B=F, E=T: 0.29
  B=F, E=F: 0.001

P(J | A):
  A=T: 0.90
  A=F: 0.05

P(M | A):
  A=T: 0.70
  A=F: 0.01
```

**Calculate:**
f) P(B=T | J=T, M=T)
g) P(A=T | J=T, M=F)

---

### Exercise 2: D-Separation
**Given this Bayesian network:**

```
    A
   / \
  B   C
   \ /
    D
    |
    E
```

**Determine if these pairs are d-separated:**
a) A and E (no evidence)
b) A and E given D
c) B and C (no evidence)
d) B and C given A
e) B and C given D

**Explain your reasoning for each case.**

---

### Exercise 3: Network Reasoning
**Student Network:**

```
Difficulty ──→ Grade ←── Intelligence
                ↓
              SAT ←── Intelligence
```

**Given:**
- P(Intelligence=high) = 0.3
- P(Difficulty=hard) = 0.4
- P(Grade | Intelligence, Difficulty)
- P(SAT | Intelligence)

**Questions:**
a) Are Intelligence and Difficulty independent?
b) Are Grade and SAT independent given Intelligence?
c) If we observe Grade=A, does this change our belief about Intelligence?
d) If we observe both Grade=A and SAT=high, what can we infer about Intelligence?
e) How does learning Difficulty=hard affect our belief about Grade?

---

## 💻 Programming Exercises

### Exercise 4: Implement Probability Calculations

```python
from typing import Dict, List
import itertools

class ProbabilityTable:
    """Represent a joint probability distribution."""
    
    def __init__(self, variables: List[str]):
        self.variables = variables
        self.table = {}  # Maps tuple of values to probability
    
    def set_probability(self, assignment: Dict[str, any], prob: float):
        """Set P(assignment) = prob."""
        key = tuple(assignment[var] for var in self.variables)
        self.table[key] = prob
    
    def get_probability(self, assignment: Dict[str, any]) -> float:
        """Get P(assignment)."""
        # TODO: Implement
        pass
    
    def marginalize(self, variable: str) -> 'ProbabilityTable':
        """
        Marginalize out a variable.
        P(X) = Σ_y P(X, Y=y)
        """
        # TODO: Implement
        # Create new table without this variable
        # Sum over all values of the variable
        pass
    
    def condition(self, evidence: Dict[str, any]) -> 'ProbabilityTable':
        """
        Condition on evidence.
        P(X | E=e) = P(X, E=e) / P(E=e)
        """
        # TODO: Implement
        pass
    
    def normalize(self):
        """Ensure probabilities sum to 1."""
        total = sum(self.table.values())
        for key in self.table:
            self.table[key] /= total

# Test with Weather/Sprinkler/Grass example
def test_probability_table():
    # Create joint distribution P(Weather, Sprinkler, GrassWet)
    pt = ProbabilityTable(['Weather', 'Sprinkler', 'GrassWet'])
    
    # Set probabilities (from Question 1)
    pt.set_probability({'Weather': 'Sunny', 'Sprinkler': 'On', 'GrassWet': 'Yes'}, 0.15)
    # ... set all other probabilities
    
    # Test marginalization
    p_grass = pt.marginalize('Weather').marginalize('Sprinkler')
    print(f"P(GrassWet=Yes) = {p_grass.get_probability({'GrassWet': 'Yes'})}")
    
    # Test conditioning
    p_weather_given_grass = pt.condition({'GrassWet': 'Yes'}).marginalize('Sprinkler')
    print(f"P(Weather=Rainy | GrassWet=Yes) = ...")
```

---

### Exercise 5: Implement Bayes' Theorem

```python
def bayes_theorem(prior: float, likelihood: float, 
                  evidence_prob: float) -> float:
    """
    Calculate posterior probability using Bayes' theorem.
    
    P(H | E) = P(E | H) * P(H) / P(E)
    
    Args:
        prior: P(H) - prior probability of hypothesis
        likelihood: P(E | H) - likelihood of evidence given hypothesis
        evidence_prob: P(E) - probability of evidence
    
    Returns:
        P(H | E) - posterior probability
    """
    # TODO: Implement
    pass

def calculate_evidence_prob(prior_h: float, likelihood_h: float,
                           prior_not_h: float, likelihood_not_h: float) -> float:
    """
    Calculate P(E) using law of total probability.
    P(E) = P(E|H)*P(H) + P(E|¬H)*P(¬H)
    """
    # TODO: Implement
    pass

# Test with medical diagnosis
def test_medical_diagnosis():
    """Test with disease testing scenario from Question 2."""
    p_disease = 0.01
    p_pos_given_disease = 0.95
    p_neg_given_healthy = 0.90
    
    # Calculate P(Positive)
    p_healthy = 1 - p_disease
    p_pos_given_healthy = 1 - p_neg_given_healthy
    p_positive = calculate_evidence_prob(
        p_disease, p_pos_given_disease,
        p_healthy, p_pos_given_healthy
    )
    
    # Calculate P(Disease | Positive)
    posterior = bayes_theorem(p_disease, p_pos_given_disease, p_positive)
    
    print(f"P(Disease | Positive) = {posterior:.4f}")
    print(f"Interpretation: {posterior*100:.2f}% chance of disease if test is positive")
```

---

### Exercise 6: Build a Bayesian Network

```python
from typing import Dict, Tuple, Callable
import numpy as np

class BayesianNetwork:
    """Simple Bayesian Network implementation."""
    
    def __init__(self):
        self.nodes = {}  # node_name -> Node
        self.edges = []  # (parent, child) tuples
    
    def add_node(self, name: str, domain: list, cpt: Dict = None):
        """
        Add a node to the network.
        
        Args:
            name: Variable name
            domain: Possible values
            cpt: Conditional probability table
        """
        self.nodes[name] = {
            'domain': domain,
            'cpt': cpt or {},
            'parents': []
        }
    
    def add_edge(self, parent: str, child: str):
        """Add directed edge from parent to child."""
        self.edges.append((parent, child))
        self.nodes[child]['parents'].append(parent)
    
    def get_probability(self, node: str, value: any, 
                       parent_values: Dict[str, any]) -> float:
        """
        Get P(node=value | parents=parent_values).
        """
        # TODO: Implement
        # Look up in CPT based on parent values
        pass
    
    def enumerate_all(self, query_var: str, evidence: Dict[str, any]) -> Dict[str, float]:
        """
        Exact inference by enumeration.
        Calculate P(query_var | evidence) for all values.
        """
        # TODO: Implement enumeration algorithm
        # 1. For each value of query variable
        # 2. Sum over all hidden variables
        # 3. Multiply probabilities according to network structure
        # 4. Normalize
        pass
    
    def sample(self, evidence: Dict[str, any] = None) -> Dict[str, any]:
        """
        Generate a sample from the network using ancestral sampling.
        """
        # TODO: Implement
        # Sample each variable in topological order
        # Condition on parent values
        pass

# Implement the Burglar Alarm network
def create_alarm_network():
    """Create the burglar alarm Bayesian network."""
    bn = BayesianNetwork()
    
    # Add nodes
    bn.add_node('Burglary', [True, False], {
        (True,): 0.001,
        (False,): 0.999
    })
    
    bn.add_node('Earthquake', [True, False], {
        (True,): 0.002,
        (False,): 0.998
    })
    
    # TODO: Add remaining nodes with CPTs
    
    # Add edges
    bn.add_edge('Burglary', 'Alarm')
    bn.add_edge('Earthquake', 'Alarm')
    # TODO: Add remaining edges
    
    return bn

# Test inference
def test_alarm_network():
    bn = create_alarm_network()
    
    # Query: P(Burglary | JohnCalls=True, MaryCalls=True)
    result = bn.enumerate_all('Burglary', {
        'JohnCalls': True,
        'MaryCalls': True
    })
    
    print(f"P(Burglary=True | John and Mary called) = {result[True]:.4f}")
```

---

### Exercise 7: Naive Bayes Classifier

Implement a Naive Bayes classifier for spam detection:

```python
from collections import defaultdict
import math

class NaiveBayesClassifier:
    """Naive Bayes classifier with Laplace smoothing."""
    
    def __init__(self, alpha=1.0):
        """
        Args:
            alpha: Laplace smoothing parameter
        """
        self.alpha = alpha
        self.class_priors = {}  # P(class)
        self.feature_probs = {}  # P(feature | class)
        self.classes = set()
        self.vocab = set()
    
    def fit(self, X: List[List[str]], y: List[str]):
        """
        Train the classifier.
        
        Args:
            X: List of documents (each document is list of words)
            y: List of class labels
        """
        # TODO: Implement training
        # 1. Calculate class priors P(class)
        # 2. Calculate P(word | class) for all words and classes
        # 3. Apply Laplace smoothing
        pass
    
    def predict_proba(self, document: List[str]) -> Dict[str, float]:
        """
        Predict class probabilities for a document.
        
        Returns:
            Dictionary mapping class -> probability
        """
        # TODO: Implement prediction
        # For each class:
        #   P(class | doc) ∝ P(class) * ∏ P(word | class)
        # Use log probabilities to avoid underflow
        pass
    
    def predict(self, document: List[str]) -> str:
        """Predict most likely class."""
        probs = self.predict_proba(document)
        return max(probs, key=probs.get)

# Test with spam classification
def test_spam_classifier():
    # Training data
    X_train = [
        ['buy', 'cheap', 'watches', 'now'],
        ['meeting', 'tomorrow', 'at', 'noon'],
        ['get', 'rich', 'quick'],
        ['lunch', 'plans', 'tomorrow'],
        # ... more examples
    ]
    
    y_train = ['spam', 'ham', 'spam', 'ham']  # labels
    
    # Train classifier
    nb = NaiveBayesClassifier()
    nb.fit(X_train, y_train)
    
    # Test
    test_doc = ['cheap', 'watches', 'on', 'sale']
    prediction = nb.predict(test_doc)
    probs = nb.predict_proba(test_doc)
    
    print(f"Document: {test_doc}")
    print(f"Prediction: {prediction}")
    print(f"P(spam | doc) = {probs['spam']:.4f}")
```

---

## 🧩 Problem-Solving Challenges

### Challenge 1: Medical Diagnosis System
**Build a Bayesian network for medical diagnosis:**

**Variables:**
- Diseases: Flu, COVID, Cold, Allergies
- Symptoms: Fever, Cough, Sore Throat, Runny Nose, Fatigue
- Risk Factors: Age, Vaccination Status, Exposure

**Requirements:**
1. Design the network structure (which variables depend on which)
2. Gather or estimate conditional probabilities
3. Implement the network
4. Test with various symptom combinations
5. Compare with doctor's diagnosis patterns

**Example queries:**
- P(Flu | Fever, Cough)
- P(COVID | Fever, Cough, Exposure)
- Most likely diagnosis given symptoms

---

### Challenge 2: Weather Prediction Model
**Create a probabilistic weather forecasting system:**

**Variables:**
- Season, Temperature, Humidity, Wind
- Cloud Cover, Precipitation
- Tomorrow's Weather

**Tasks:**
1. Collect historical weather data
2. Build Bayesian network structure
3. Learn parameters from data
4. Make predictions
5. Evaluate accuracy

**Advanced:**
- Dynamic Bayesian Network for time-series
- Compare with actual forecasts

---

### Challenge 3: Student Performance Predictor
**Model factors affecting student grades:**

**Variables:**
- Student: IQ, Prior Knowledge, Study Hours
- Course: Difficulty, Teaching Quality
- Environment: Class Size, Resources
- Outcome: Exam Score, Final Grade

**Build:**
1. Bayesian network structure
2. Parameter estimation from student data
3. Prediction system
4. Analysis of causal factors

**Use cases:**
- Predict student performance
- Identify at-risk students
- Recommend interventions

---

## 🔬 Advanced Exercises

### Exercise 8: Variable Elimination
**Implement variable elimination algorithm:**

```python
class Factor:
    """Represents a factor in variable elimination."""
    
    def __init__(self, variables: List[str], values: np.ndarray):
        self.variables = variables
        self.values = values  # Multi-dimensional array
    
    def multiply(self, other: 'Factor') -> 'Factor':
        """Multiply two factors."""
        # TODO: Implement factor multiplication
        pass
    
    def marginalize(self, variable: str) -> 'Factor':
        """Sum out a variable."""
        # TODO: Implement marginalization
        pass

def variable_elimination(bn: BayesianNetwork, 
                        query_var: str,
                        evidence: Dict[str, any],
                        elimination_order: List[str]) -> Dict:
    """
    Perform exact inference using variable elimination.
    
    Args:
        bn: Bayesian network
        query_var: Variable to query
        evidence: Observed variables
        elimination_order: Order to eliminate variables
    
    Returns:
        Probability distribution over query variable
    """
    # TODO: Implement variable elimination
    # 1. Create initial factors from CPTs
    # 2. Reduce factors based on evidence
    # 3. For each variable in elimination order:
    #    - Multiply factors containing that variable
    #    - Sum out the variable
    # 4. Multiply remaining factors
    # 5. Normalize
    pass
```

---

### Exercise 9: Gibbs Sampling
**Implement approximate inference using Gibbs sampling:**

```python
def gibbs_sampling(bn: BayesianNetwork,
                  query_var: str,
                  evidence: Dict[str, any],
                  num_samples: int = 10000) -> Dict:
    """
    Approximate inference using Gibbs sampling.
    
    Args:
        bn: Bayesian network
        query_var: Variable to query
        evidence: Observed variables
        num_samples: Number of samples to generate
    
    Returns:
        Approximate probability distribution
    """
    # TODO: Implement Gibbs sampling
    # 1. Initialize non-evidence variables randomly
    # 2. For each iteration:
    #    - For each non-evidence variable:
    #      - Sample from P(var | markov_blanket)
    # 3. Count samples for query variable
    # 4. Normalize counts to get probabilities
    pass

def markov_blanket(bn: BayesianNetwork, node: str) -> List[str]:
    """
    Get Markov blanket of a node:
    - Parents
    - Children  
    - Children's other parents
    """
    # TODO: Implement
    pass
```

---

### Exercise 10: Parameter Learning
**Learn Bayesian network parameters from data:**

```python
def maximum_likelihood_estimation(data: List[Dict[str, any]],
                                 bn_structure: BayesianNetwork) -> BayesianNetwork:
    """
    Learn CPT parameters using maximum likelihood estimation.
    
    Args:
        data: List of complete observations
        bn_structure: Network structure (edges known, CPTs unknown)
    
    Returns:
        Network with learned CPTs
    """
    # TODO: Implement MLE
    # For each node:
    #   For each parent configuration:
    #     Count occurrences in data
    #     Calculate conditional probabilities
    pass

def em_algorithm(data: List[Dict[str, any]],
                bn_structure: BayesianNetwork,
                max_iterations: int = 100) -> BayesianNetwork:
    """
    Learn parameters with missing data using EM algorithm.
    
    Args:
        data: List of observations (may have missing values)
        bn_structure: Network structure
        max_iterations: Maximum EM iterations
    
    Returns:
        Network with learned CPTs
    """
    # TODO: Implement EM
    # E-step: Infer missing values given current parameters
    # M-step: Update parameters given completed data
    # Repeat until convergence
    pass
```

---

## 📊 Analysis Questions

### Question 4: Network Complexity
**For a Bayesian network with n binary variables:**

a) What's the worst-case number of parameters needed?
b) If the network is a chain (A→B→C→...→Z), how many parameters?
c) If the network is a tree, how many parameters?
d) Why are Bayesian networks more efficient than full joint distributions?

---

### Question 5: Inference Complexity
**Analyze computational complexity:**

a) Exact inference by enumeration: Time complexity?
b) Variable elimination: Best and worst case?
c) When would you use approximate inference instead?
d) Compare sampling methods: Rejection, Likelihood weighting, Gibbs

---

## 📝 Submission Guidelines

### What to Submit:
1. **Probability calculations** with clear work shown
2. **Bayesian network implementations** (complete working code)
3. **Real application** (medical diagnosis, spam filter, or custom)
4. **Performance analysis** comparing exact and approximate inference
5. **Report** (750-1000 words) covering:
   - Network design decisions
   - Implementation challenges
   - Inference results
   - Practical insights
   - Real-world applications

### Evaluation Criteria:
- **Correctness:** Probability calculations and inference are accurate
- **Implementation:** Clean, efficient, well-documented code
- **Design:** Well-motivated network structure and parameters
- **Analysis:** Deep understanding of probabilistic reasoning
- **Application:** Practical, useful real-world example

### Bonus Points:
- Learning network structure from data
- Dynamic Bayesian Networks
- Large-scale application
- Novel inference algorithms
- Visualization of inference process

### Due Date: End of Week 6

---

## 💡 Study Tips

### Mastering Probability
- **Practice, practice, practice:** Do many calculation problems
- **Check your work:** Probabilities must sum to 1
- **Use trees:** Draw probability trees for complex problems
- **Think through problems:** Does the answer make intuitive sense?

### Understanding Bayesian Networks
- **Draw networks:** Always sketch the DAG
- **Test independence:** Use d-separation rules
- **Start simple:** Master small networks first
- **Verify by hand:** Calculate simple queries manually

### Common Mistakes
- **P(A|B) vs P(B|A):** These are NOT the same!
- **Forgetting normalization:** Always normalize probability distributions
- **Wrong independence assumptions:** Test them carefully
- **Numerical underflow:** Use log probabilities for products

### Advanced Topics
- **Dynamic Bayesian Networks:** For temporal reasoning
- **Decision Networks:** Adding utility and decisions
- **Causal reasoning:** Going beyond correlation
- **Deep learning:** Modern probabilistic models

---

**🎯 Key Takeaway:** Probabilistic reasoning is essential for handling uncertainty in AI. Bayesian networks provide an elegant framework for representing and reasoning with uncertain knowledge. Master the basics of probability theory, then learn to build and use Bayesian networks for real applications!
