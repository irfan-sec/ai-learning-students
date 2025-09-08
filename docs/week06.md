---
layout: default
title: "Week 6: Reasoning under Uncertainty"
permalink: /week06.html
---

### Navigation
[🏠 Home](index.html) | [← Previous: Week 5](week05.html) | [Next: Week 7 →](week07.html)

---

# Week 6: Reasoning under Uncertainty

## 📚 Learning Objectives
By the end of this week, students will be able to:
- Understand probability theory fundamentals for AI
- Apply Bayes' rule for probabilistic inference
- Work with Bayesian networks for complex reasoning
- Handle uncertainty in AI decision-making
- Implement basic probabilistic reasoning algorithms

---

## 🎲 Why Uncertainty Matters in AI

Real-world AI systems must deal with:
- **Incomplete information** - Missing data or observations
- **Noisy sensors** - Measurement errors and uncertainty
- **Stochastic processes** - Inherently random phenomena
- **Complex dependencies** - Uncertain relationships between variables

![Uncertainty in AI](images/uncertainty-in-ai.png)

### Examples of Uncertainty
- **Medical diagnosis:** Symptoms don't always indicate specific diseases
- **Speech recognition:** Audio signals are noisy and ambiguous  
- **Computer vision:** Lighting and occlusion create ambiguity
- **Natural language:** Words and sentences can have multiple meanings

---

## 📊 Probability Theory Foundations

### Basic Probability Concepts

```python
import random
import numpy as np
from collections import Counter

class ProbabilityBasics:
    """Demonstrate basic probability concepts"""
    
    @staticmethod
    def coin_flip_simulation(n_flips=1000):
        """Simulate coin flips to demonstrate probability"""
        results = [random.choice(['H', 'T']) for _ in range(n_flips)]
        prob_heads = results.count('H') / n_flips
        return prob_heads, results
    
    @staticmethod
    def conditional_probability_example():
        """Demonstrate conditional probability with card drawing"""
        # P(King | Face Card) = P(King and Face Card) / P(Face Card)
        # Face cards: J, Q, K (12 total), Kings: 4 total
        return 4/12  # 1/3
    
    @staticmethod
    def bayes_rule(prior, likelihood, evidence):
        """Apply Bayes' rule: P(H|E) = P(E|H) * P(H) / P(E)"""
        posterior = (likelihood * prior) / evidence
        return posterior

# Example usage
prob_calc = ProbabilityBasics()
heads_prob, _ = prob_calc.coin_flip_simulation()
print(f"Empirical probability of heads: {heads_prob:.3f}")
```

### Joint and Conditional Probability

```python
class ProbabilityDistributions:
    """Work with probability distributions"""
    
    def __init__(self):
        # Weather example: P(Weather, Temperature)
        self.joint_prob = {
            ('sunny', 'hot'): 0.3,
            ('sunny', 'mild'): 0.2,
            ('cloudy', 'hot'): 0.1,
            ('cloudy', 'mild'): 0.2,
            ('rainy', 'hot'): 0.05,
            ('rainy', 'mild'): 0.15
        }
    
    def marginal_probability(self, variable, value):
        """Calculate P(variable = value)"""
        if variable == 'weather':
            return sum(prob for (weather, temp), prob in self.joint_prob.items() 
                      if weather == value)
        elif variable == 'temperature':
            return sum(prob for (weather, temp), prob in self.joint_prob.items() 
                      if temp == value)
    
    def conditional_probability(self, var1, val1, var2, val2):
        """Calculate P(var1=val1 | var2=val2)"""
        joint = self.joint_prob.get((val1, val2), 0) if var1 == 'weather' else \
                self.joint_prob.get((val2, val1), 0)
        marginal = self.marginal_probability(var2, val2)
        return joint / marginal if marginal > 0 else 0

# Example
weather_model = ProbabilityDistributions()
prob_sunny = weather_model.marginal_probability('weather', 'sunny')
prob_sunny_given_hot = weather_model.conditional_probability('weather', 'sunny', 'temperature', 'hot')
print(f"P(sunny) = {prob_sunny}")
print(f"P(sunny | hot) = {prob_sunny_given_hot:.3f}")
```

---

## 🧮 Bayes' Rule and Inference

### The Foundation of Probabilistic Reasoning

Bayes' Rule: P(H|E) = P(E|H) × P(H) / P(E)

- **P(H|E):** Posterior probability (what we want to know)
- **P(E|H):** Likelihood (how well hypothesis explains evidence)
- **P(H):** Prior probability (what we believed before)
- **P(E):** Evidence probability (normalizing constant)

### Medical Diagnosis Example

```python
class MedicalDiagnosis:
    """Bayesian inference for medical diagnosis"""
    
    def __init__(self):
        # Disease probabilities
        self.disease_prior = 0.01  # 1% of population has disease
        
        # Test characteristics
        self.test_sensitivity = 0.95  # P(positive | disease)
        self.test_specificity = 0.90  # P(negative | no disease)
    
    def diagnose(self, test_result):
        """Use Bayes' rule to diagnose based on test result"""
        if test_result == 'positive':
            # P(disease | positive test)
            likelihood = self.test_sensitivity
            prior = self.disease_prior
            
            # P(positive test) = P(pos|disease)*P(disease) + P(pos|no disease)*P(no disease)
            evidence = (self.test_sensitivity * self.disease_prior + 
                       (1 - self.test_specificity) * (1 - self.disease_prior))
            
            posterior = (likelihood * prior) / evidence
            return posterior
        
        else:  # negative test
            # P(disease | negative test)
            likelihood = 1 - self.test_sensitivity  # P(negative | disease)
            prior = self.disease_prior
            
            evidence = ((1 - self.test_sensitivity) * self.disease_prior + 
                       self.test_specificity * (1 - self.disease_prior))
            
            posterior = (likelihood * prior) / evidence
            return posterior

# Example diagnosis
doctor = MedicalDiagnosis()
prob_disease_given_positive = doctor.diagnose('positive')
prob_disease_given_negative = doctor.diagnose('negative')

print(f"P(disease | positive test) = {prob_disease_given_positive:.3f}")
print(f"P(disease | negative test) = {prob_disease_given_negative:.6f}")
```

---

## 🕸️ Bayesian Networks

Bayesian networks provide a compact representation of joint probability distributions using conditional independence.

### Simple Bayesian Network Implementation

```python
class SimpleBayesianNetwork:
    """Basic Bayesian Network for the classic Alarm example"""
    
    def __init__(self):
        # Network structure: Burglary -> Alarm <- Earthquake
        #                              |
        #                         JohnCalls  MaryCalls
        
        # Prior probabilities
        self.p_burglary = 0.001
        self.p_earthquake = 0.002
        
        # Conditional probability tables
        self.p_alarm = {
            (True, True): 0.95,    # P(Alarm | Burglary, Earthquake)
            (True, False): 0.94,   # P(Alarm | Burglary, ¬Earthquake)
            (False, True): 0.29,   # P(Alarm | ¬Burglary, Earthquake)
            (False, False): 0.001  # P(Alarm | ¬Burglary, ¬Earthquake)
        }
        
        self.p_john_calls_given_alarm = 0.90
        self.p_john_calls_given_no_alarm = 0.05
        
        self.p_mary_calls_given_alarm = 0.70
        self.p_mary_calls_given_no_alarm = 0.01
    
    def sample_network(self, n_samples=10000):
        """Generate samples from the network"""
        samples = []
        
        for _ in range(n_samples):
            # Sample from priors
            burglary = random.random() < self.p_burglary
            earthquake = random.random() < self.p_earthquake
            
            # Sample alarm given parents
            alarm_prob = self.p_alarm[(burglary, earthquake)]
            alarm = random.random() < alarm_prob
            
            # Sample calls given alarm
            john_calls = random.random() < (self.p_john_calls_given_alarm if alarm 
                                          else self.p_john_calls_given_no_alarm)
            mary_calls = random.random() < (self.p_mary_calls_given_alarm if alarm 
                                          else self.p_mary_calls_given_no_alarm)
            
            samples.append({
                'burglary': burglary,
                'earthquake': earthquake, 
                'alarm': alarm,
                'john_calls': john_calls,
                'mary_calls': mary_calls
            })
        
        return samples
    
    def query_given_evidence(self, samples, query_var, evidence):
        """Estimate P(query_var | evidence) from samples"""
        matching_samples = []
        
        for sample in samples:
            # Check if sample matches evidence
            match = all(sample[var] == val for var, val in evidence.items())
            if match:
                matching_samples.append(sample)
        
        if not matching_samples:
            return 0.0
        
        # Calculate probability of query variable
        query_true = sum(1 for sample in matching_samples if sample[query_var])
        return query_true / len(matching_samples)

# Example usage
bn = SimpleBayesianNetwork()
samples = bn.sample_network(50000)

# Query: P(Burglary | JohnCalls=True, MaryCalls=True)
evidence = {'john_calls': True, 'mary_calls': True}
prob_burglary = bn.query_given_evidence(samples, 'burglary', evidence)
print(f"P(Burglary | Both call) ≈ {prob_burglary:.3f}")

# Query: P(Earthquake | JohnCalls=True, MaryCalls=False)  
evidence = {'john_calls': True, 'mary_calls': False}
prob_earthquake = bn.query_given_evidence(samples, 'earthquake', evidence)
print(f"P(Earthquake | John calls, Mary doesn't) ≈ {prob_earthquake:.3f}")
```

---

## 🎯 Probabilistic Inference Algorithms

### Variable Elimination

```python
class VariableElimination:
    """Simplified variable elimination for small networks"""
    
    def __init__(self, network):
        self.network = network
        self.factors = []
    
    def eliminate_variable(self, var, factors):
        """Eliminate a variable by summing out"""
        # This is a simplified version - real implementation would be more complex
        relevant_factors = [f for f in factors if var in f.variables]
        other_factors = [f for f in factors if var not in f.variables]
        
        # Join relevant factors and sum out variable
        if relevant_factors:
            joined = self.join_factors(relevant_factors)
            summed_out = self.sum_out_variable(joined, var)
            return other_factors + [summed_out]
        else:
            return factors
    
    def join_factors(self, factors):
        """Join multiple factors into one"""
        # Implementation would multiply probability tables
        pass
    
    def sum_out_variable(self, factor, variable):
        """Sum out a variable from a factor"""
        # Implementation would marginalize over variable
        pass
```

### Markov Chain Monte Carlo (MCMC)

```python
class MCMCSampler:
    """MCMC sampling for Bayesian networks"""
    
    def __init__(self, network):
        self.network = network
    
    def gibbs_sampling(self, evidence, n_samples=10000, burn_in=1000):
        """Gibbs sampling for posterior inference"""
        # Initialize all variables randomly
        state = {var: random.choice([True, False]) 
                for var in self.network.variables}
        
        # Set evidence variables
        for var, val in evidence.items():
            state[var] = val
        
        samples = []
        
        for i in range(n_samples + burn_in):
            # Sample each non-evidence variable given others
            for var in self.network.variables:
                if var not in evidence:
                    state[var] = self.sample_variable_given_others(var, state)
            
            # Collect samples after burn-in
            if i >= burn_in:
                samples.append(state.copy())
        
        return samples
    
    def sample_variable_given_others(self, var, state):
        """Sample variable given current state of others"""
        # Calculate P(var | markov_blanket)
        # This would use the network structure and CPTs
        prob_true = self.calculate_conditional_prob(var, True, state)
        return random.random() < prob_true
    
    def calculate_conditional_prob(self, var, value, state):
        """Calculate P(var=value | current_state)"""
        # Implementation would use Markov blanket
        return 0.5  # Placeholder
```

---

## 💻 Hands-On Exercise: Spam Filter with Naive Bayes

```python
import math
from collections import defaultdict, Counter

class NaiveBayesSpamFilter:
    """Naive Bayes classifier for spam detection"""
    
    def __init__(self):
        self.word_counts = {'spam': defaultdict(int), 'ham': defaultdict(int)}
        self.class_counts = {'spam': 0, 'ham': 0}
        self.vocabulary = set()
    
    def train(self, emails, labels):
        """Train the classifier on labeled emails"""
        for email, label in zip(emails, labels):
            self.class_counts[label] += 1
            words = email.lower().split()
            
            for word in words:
                self.word_counts[label][word] += 1
                self.vocabulary.add(word)
    
    def classify(self, email):
        """Classify an email as spam or ham"""
        words = email.lower().split()
        
        # Calculate log probabilities to avoid underflow
        log_prob_spam = math.log(self.class_counts['spam'] / 
                                sum(self.class_counts.values()))
        log_prob_ham = math.log(self.class_counts['ham'] / 
                               sum(self.class_counts.values()))
        
        # Add log-likelihood for each word
        for word in words:
            if word in self.vocabulary:
                # Laplace smoothing
                prob_word_given_spam = ((self.word_counts['spam'][word] + 1) / 
                                       (sum(self.word_counts['spam'].values()) + 
                                        len(self.vocabulary)))
                prob_word_given_ham = ((self.word_counts['ham'][word] + 1) / 
                                      (sum(self.word_counts['ham'].values()) + 
                                       len(self.vocabulary)))
                
                log_prob_spam += math.log(prob_word_given_spam)
                log_prob_ham += math.log(prob_word_given_ham)
        
        # Return class with higher probability
        return 'spam' if log_prob_spam > log_prob_ham else 'ham'
    
    def get_word_probabilities(self, word):
        """Get P(word | class) for both classes"""
        total_spam = sum(self.word_counts['spam'].values())
        total_ham = sum(self.word_counts['ham'].values())
        vocab_size = len(self.vocabulary)
        
        prob_spam = (self.word_counts['spam'][word] + 1) / (total_spam + vocab_size)
        prob_ham = (self.word_counts['ham'][word] + 1) / (total_ham + vocab_size)
        
        return prob_spam, prob_ham

# Example usage
spam_filter = NaiveBayesSpamFilter()

# Training data
emails = [
    "buy now cheap viagra discount",
    "meeting tomorrow at 3pm conference room",
    "free money click here now",
    "project deadline next week please review",
    "congratulations you won million dollars",
    "lunch plans for friday anyone interested"
]

labels = ['spam', 'ham', 'spam', 'ham', 'spam', 'ham']

spam_filter.train(emails, labels)

# Test classification
test_email = "free discount offer click now"
prediction = spam_filter.classify(test_email)
print(f"Email: '{test_email}' -> {prediction}")

# Analyze word probabilities
word = "free"
prob_spam, prob_ham = spam_filter.get_word_probabilities(word)
print(f"P('{word}' | spam) = {prob_spam:.3f}")
print(f"P('{word}' | ham) = {prob_ham:.3f}")
```

---

## 🔗 Curated Resources

### Essential Reading
- [AIMA Chapter 13-14](https://aima.cs.berkeley.edu/) - Uncertainty and Probabilistic Reasoning
- [Think Bayes](https://greenteapress.com/wp/think-bayes/) - Free book on Bayesian statistics

### Video Resources  
- [Bayes' Theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM) - 3Blue1Brown (15 min)
- [Bayesian Networks](https://www.youtube.com/watch?v=TuGDMj43ehw) - Intro to probabilistic graphical models
- [Naive Bayes](https://www.youtube.com/watch?v=O2L2Uv9pdDA) - Machine learning explanation

### Interactive Learning
- [Bayes' Rule Visualization](https://seeing-theory.brown.edu/bayesian-inference/index.html)
- [Probabilistic Reasoning](https://aispace.org/bayes/) - Interactive Bayesian network tool
- [Medical Diagnosis Simulator](https://www.ncrg.aston.ac.uk/netlab/demos/medical/medical.html)

---

## 🎯 Key Concepts Summary

### Probability Fundamentals
- **Joint, marginal, conditional probabilities**
- **Bayes' rule for updating beliefs**
- **Independence and conditional independence**

### Bayesian Networks
- **Compact representation of joint distributions**
- **Exploit conditional independence**
- **Enable efficient probabilistic inference**

### Practical Applications
- **Medical diagnosis systems**
- **Spam filtering and text classification**
- **Sensor fusion and robotics**
- **Decision support systems**

---

## 🤔 Discussion Questions

1. How does probabilistic reasoning differ from logical reasoning?
2. When might you prefer frequentist vs. Bayesian approaches?
3. What are the assumptions of Naive Bayes and when do they break down?
4. How do you handle missing data in probabilistic models?

---

## 🔍 Looking Ahead

Next week, we'll dive into **machine learning fundamentals** - the foundation for building systems that can learn patterns from data and make predictions about new, unseen examples.

---

### Navigation
[🏠 Home](index.html) | [← Previous: Week 5](week05.html) | [Next: Week 7 →](week07.html)

---

*"Probability is not about the odds of something happening, but about what we know of those odds."* - Nate Silver