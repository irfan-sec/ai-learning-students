# Week 6: Reasoning under Uncertainty

## 📚 Learning Objectives
By the end of this week, students will be able to:
- Understand why probabilistic reasoning is essential in AI
- Apply basic probability theory including joint, conditional, and marginal probability
- Use Bayes' theorem for inference and belief updating
- Understand the structure and inference in simple Bayesian networks
- Implement basic probabilistic reasoning algorithms in Python

---

## 🎲 Why Probability in AI?

The real world is **uncertain**. Perfect logical knowledge is rare:

### Limitations of Pure Logic
- **Incomplete Information:** We don't know everything about the world
- **Unreliable Sensors:** Measurements contain noise and errors  
- **Stochastic Processes:** Many phenomena are inherently random
- **Computational Limits:** Can't consider all possibilities

### Examples of Uncertainty in AI
- **Medical Diagnosis:** Symptoms don't guarantee specific diseases
- **Speech Recognition:** Audio signals are noisy and ambiguous
- **Autonomous Driving:** Sensor readings and behavior predictions are probabilistic
- **Game Playing:** Opponent's moves and hidden information

**Solution:** Use **probability theory** to quantify and reason with uncertainty.

---

## 📊 Probability Theory Foundations

### Random Variables
A **random variable** represents an uncertain quantity:
- **Boolean:** `Rain` ∈ {true, false}
- **Discrete:** `Weather` ∈ {sunny, cloudy, rainy, snowy}
- **Continuous:** `Temperature` ∈ ℝ

### Probability Distributions
**Probability Distribution** P(X) specifies the likelihood of each possible value:

**Example - Weather:**
- P(Weather = sunny) = 0.6
- P(Weather = cloudy) = 0.2  
- P(Weather = rainy) = 0.15
- P(Weather = snowy) = 0.05

**Properties:**
- 0 ≤ P(X = x) ≤ 1 for all x
- Σ P(X = x) = 1 (probabilities sum to 1)

### Joint Probability
**Joint probability** P(X, Y) gives probability of multiple variables:

**Example:**
| Weather | Umbrella | P(Weather, Umbrella) |
|---------|----------|---------------------|
| sunny   | true     | 0.1                |
| sunny   | false    | 0.5                |
| rainy   | true     | 0.12               |
| rainy   | false    | 0.03               |
| cloudy  | true     | 0.08               |
| cloudy  | false    | 0.12               |
| snowy   | true     | 0.03               |
| snowy   | false    | 0.02               |

### Marginal Probability
**Marginal probability** is obtained by summing over other variables:

P(Weather = rainy) = P(rainy, true) + P(rainy, false) = 0.12 + 0.03 = 0.15

**General Rule:**
P(X) = Σ_y P(X, Y = y)

### Conditional Probability
**Conditional probability** P(X|Y) represents probability of X given that Y is observed:

**Definition:** P(X|Y) = P(X, Y) / P(Y)

**Example:**
P(Umbrella = true | Weather = rainy) = P(rainy, true) / P(rainy) = 0.12 / 0.15 = 0.8

**Interpretation:** 80% chance of carrying umbrella when it's raining.

---

## 🧮 Bayes' Theorem

### The Most Important Formula in AI

**Bayes' Theorem:**
P(H|E) = P(E|H) × P(H) / P(E)

Where:
- **H** = Hypothesis (what we want to know)
- **E** = Evidence (what we observe)
- **P(H|E)** = Posterior probability (updated belief)
- **P(E|H)** = Likelihood (how likely evidence given hypothesis)
- **P(H)** = Prior probability (initial belief)
- **P(E)** = Marginal likelihood (normalizing constant)

### Medical Diagnosis Example

**Problem:** A patient tests positive for a rare disease. What's the probability they actually have it?

**Given:**
- Disease prevalence: P(Disease) = 0.001 (0.1% of population)
- Test sensitivity: P(Positive|Disease) = 0.99 (99% detection rate)
- Test false positive rate: P(Positive|¬Disease) = 0.05 (5% false positives)

**Question:** P(Disease|Positive) = ?

**Solution using Bayes' Theorem:**

1. **Prior:** P(Disease) = 0.001
2. **Likelihood:** P(Positive|Disease) = 0.99
3. **Marginal likelihood:** 
   P(Positive) = P(Positive|Disease)×P(Disease) + P(Positive|¬Disease)×P(¬Disease)
   P(Positive) = 0.99×0.001 + 0.05×0.999 = 0.00099 + 0.04995 ≈ 0.051

4. **Posterior:**
   P(Disease|Positive) = (0.99 × 0.001) / 0.051 ≈ 0.019

**Result:** Only 1.9% chance of actually having the disease!

**Key Insight:** Base rates matter enormously in diagnosis.

---

## 🕸️ Bayesian Networks

### Representing Complex Dependencies

A **Bayesian Network** is a directed acyclic graph where:
- **Nodes** represent random variables
- **Edges** represent direct dependencies  
- **Conditional Probability Tables (CPTs)** quantify relationships

### Example: Burglar Alarm Network

```
    Burglary    Earthquake
        ↓           ↓
         ← Alarm →
        ↙       ↘
  JohnCalls   MaryCalls
```

**Variables:**
- Burglary, Earthquake: {true, false}
- Alarm: {true, false}  
- JohnCalls, MaryCalls: {true, false}

**Conditional Probability Tables:**

**P(Burglary):**
| Burglary | P |
|----------|---|
| true     | 0.001 |
| false    | 0.999 |

**P(Earthquake):**
| Earthquake | P |
|------------|---|
| true       | 0.002 |
| false      | 0.998 |

**P(Alarm | Burglary, Earthquake):**
| Burglary | Earthquake | P(Alarm=true) |
|----------|------------|---------------|
| true     | true       | 0.95          |
| true     | false      | 0.94          |
| false    | true       | 0.29          |
| false    | false      | 0.001         |

**P(JohnCalls | Alarm):**
| Alarm | P(JohnCalls=true) |
|-------|-------------------|
| true  | 0.90              |
| false | 0.05              |

**P(MaryCalls | Alarm):**
| Alarm | P(MaryCalls=true) |
|-------|-------------------|
| true  | 0.70              |
| false | 0.01              |

### Independence and Conditional Independence

**Key Advantage:** Bayesian networks exploit independence to reduce the number of parameters needed.

**Full joint distribution** would require 2^5 - 1 = 31 parameters.
**Bayesian network** requires only 1 + 1 + 4 + 2 + 2 = 10 parameters.

**Conditional Independence:** 
JohnCalls is conditionally independent of Burglary given Alarm:
P(JohnCalls | Alarm, Burglary) = P(JohnCalls | Alarm)

---

## 🔍 Inference in Bayesian Networks

### Types of Inference
1. **Prior Probability:** P(Burglary) = 0.001
2. **Posterior Probability:** P(Burglary | JohnCalls = true)
3. **Most Likely Explanation:** Which combination of variables is most probable?

### Enumeration Algorithm

**Algorithm:** Sum over all possible values of unobserved variables

**Example:** P(Burglary | JohnCalls = true, MaryCalls = true)

```python
def enumeration_ask(X, e, bn):
    """
    Compute P(X|e) using enumeration algorithm
    X: query variable
    e: evidence (observed variables)
    bn: Bayesian network
    """
    Q = {}
    for x in X.domain:
        e_extended = e.copy()
        e_extended[X] = x
        Q[x] = enumerate_all(bn.variables, e_extended, bn)
    
    return normalize(Q)

def enumerate_all(variables, e, bn):
    if not variables:
        return 1.0
    
    Y = variables[0]
    rest = variables[1:]
    
    if Y in e:  # Y is observed
        return bn.p(Y, e[Y], parents(Y, bn)) * enumerate_all(rest, e, bn)
    else:       # Y is unobserved, sum over all values
        total = 0
        for y in Y.domain:
            e_extended = e.copy()
            e_extended[Y] = y
            total += bn.p(Y, y, parents(Y, bn)) * enumerate_all(rest, e_extended, bn)
        return total
```

---

## 💻 Practical Implementation

### Simple Bayesian Network in Python

```python
import numpy as np
from itertools import product

class BayesianNetwork:
    def __init__(self):
        self.variables = {}
        self.parents = {}
        self.cpts = {}
    
    def add_variable(self, name, domain):
        self.variables[name] = domain
        self.parents[name] = []
        
    def add_edge(self, parent, child):
        if child not in self.parents:
            self.parents[child] = []
        self.parents[child].append(parent)
    
    def set_cpt(self, variable, cpt):
        """Set conditional probability table"""
        self.cpts[variable] = cpt
    
    def get_probability(self, variable, value, evidence):
        """Get P(variable=value | parents=evidence)"""
        parent_values = tuple(evidence.get(p, None) for p in self.parents[variable])
        return self.cpts[variable].get((value,) + parent_values, 0.0)

# Example: Simple weather model
bn = BayesianNetwork()
bn.add_variable('Season', ['summer', 'winter'])
bn.add_variable('Rain', [True, False])
bn.add_edge('Season', 'Rain')

# Set probability tables
bn.set_cpt('Season', {
    ('summer',): 0.6,
    ('winter',): 0.4
})

bn.set_cpt('Rain', {
    (True, 'summer'): 0.2,
    (False, 'summer'): 0.8,
    (True, 'winter'): 0.7,
    (False, 'winter'): 0.3
})
```

---

## 🤔 Discussion Questions

1. **Base Rate Neglect:** Why do people often ignore base rates when making probabilistic judgments? How can AI systems help?

2. **Computational Complexity:** Exact inference in Bayesian networks is NP-hard in general. What approximation methods might be useful?

3. **Learning vs. Knowledge:** When should we learn probabilities from data vs. specify them from expert knowledge?

4. **Uncertainty vs. Ignorance:** What's the difference between uncertainty (randomness) and ignorance (lack of information)?

---

## 🔍 Looking Ahead

Understanding probability sets the foundation for machine learning! Next week we'll see how probabilistic principles underlie learning algorithms.

**Key Connections:**
- Week 7-11: Machine learning builds on probabilistic foundations
- Week 10: Neural networks can be viewed as probabilistic models
- Week 13: Ethical issues arise from uncertain AI decisions

**Practical Assignment:** Implement a medical diagnosis system using Bayesian networks to reason about symptoms and diseases with uncertainty.