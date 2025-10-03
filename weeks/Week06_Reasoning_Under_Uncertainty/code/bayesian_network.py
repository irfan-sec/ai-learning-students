"""
Bayesian Network Implementation
Week 6: Reasoning under Uncertainty

This module implements a simple Bayesian Network with inference capabilities.
"""

from typing import Dict, List, Tuple, Any
from itertools import product
import random


class BayesianNetwork:
    """
    A simple Bayesian Network implementation supporting:
    - Network construction
    - Conditional probability table (CPT) specification
    - Exact inference by enumeration
    - Sampling
    """
    
    def __init__(self):
        self.variables = {}  # variable_name -> domain
        self.parents = {}    # variable_name -> list of parent names
        self.cpts = {}       # variable_name -> CPT dict
    
    def add_variable(self, name: str, domain: List[Any]):
        """Add a variable to the network."""
        self.variables[name] = domain
        self.parents[name] = []
        self.cpts[name] = {}
    
    def add_edge(self, parent: str, child: str):
        """Add a directed edge from parent to child."""
        if child not in self.parents:
            self.parents[child] = []
        self.parents[child].append(parent)
    
    def set_cpt(self, variable: str, cpt: Dict):
        """
        Set the conditional probability table for a variable.
        
        For root nodes (no parents):
            cpt = {value: probability}
        
        For nodes with parents:
            cpt = {(value, parent1_val, parent2_val, ...): probability}
        """
        self.cpts[variable] = cpt
    
    def get_probability(self, variable: str, value: Any, parent_values: Dict[str, Any]) -> float:
        """Get P(variable=value | parents=parent_values)."""
        if not self.parents[variable]:
            # Root node - no parents
            return self.cpts[variable].get(value, 0.0)
        else:
            # Build key from parent values in order
            key = tuple([value] + [parent_values[p] for p in self.parents[variable]])
            return self.cpts[variable].get(key, 0.0)
    
    def enumerate_all(self, query_var: str, evidence: Dict[str, Any]) -> Dict[Any, float]:
        """
        Compute P(query_var | evidence) by enumeration.
        
        Returns a dictionary mapping each value of query_var to its probability.
        """
        result = {}
        
        # For each value of the query variable
        for query_value in self.variables[query_var]:
            # Create extended evidence
            extended_evidence = evidence.copy()
            extended_evidence[query_var] = query_value
            
            # Sum over all hidden variables
            prob = self._enumerate_all_recursive(
                list(self.variables.keys()),
                extended_evidence
            )
            result[query_value] = prob
        
        # Normalize
        total = sum(result.values())
        if total > 0:
            for key in result:
                result[key] /= total
        
        return result
    
    def _enumerate_all_recursive(self, variables: List[str], evidence: Dict[str, Any]) -> float:
        """Helper for enumeration - recursively sum over hidden variables."""
        if not variables:
            # Base case: all variables assigned, return product of probabilities
            prob = 1.0
            for var in self.variables:
                value = evidence[var]
                parent_vals = {p: evidence[p] for p in self.parents[var]}
                prob *= self.get_probability(var, value, parent_vals)
            return prob
        
        # Get first variable
        var = variables[0]
        rest = variables[1:]
        
        if var in evidence:
            # Variable is observed
            return self._enumerate_all_recursive(rest, evidence)
        else:
            # Variable is hidden - sum over all values
            total = 0.0
            for value in self.variables[var]:
                extended = evidence.copy()
                extended[var] = value
                total += self._enumerate_all_recursive(rest, extended)
            return total
    
    def sample(self, evidence: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a sample using ancestral sampling.
        Samples each variable in topological order conditioning on parent values.
        """
        if evidence is None:
            evidence = {}
        
        sample = {}
        
        # Need to sample in topological order
        # Simple approach: keep trying to sample variables whose parents are sampled
        remaining = set(self.variables.keys()) - set(evidence.keys())
        
        while remaining:
            for var in list(remaining):
                # Check if all parents are sampled
                if all(p in sample or p in evidence for p in self.parents[var]):
                    # Can sample this variable
                    parent_values = {}
                    for p in self.parents[var]:
                        parent_values[p] = evidence.get(p, sample.get(p))
                    
                    # Sample according to P(var | parents)
                    value = self._sample_from_distribution(var, parent_values)
                    sample[var] = value
                    remaining.remove(var)
                    break
        
        # Add evidence to sample
        sample.update(evidence)
        return sample
    
    def _sample_from_distribution(self, variable: str, parent_values: Dict[str, Any]) -> Any:
        """Sample a value for variable given parent values."""
        # Get distribution
        probs = []
        values = []
        for value in self.variables[variable]:
            prob = self.get_probability(variable, value, parent_values)
            probs.append(prob)
            values.append(value)
        
        # Sample
        r = random.random()
        cumulative = 0.0
        for prob, value in zip(probs, values):
            cumulative += prob
            if r <= cumulative:
                return value
        
        return values[-1]  # Fallback


def create_alarm_network():
    """
    Create the classic Burglar Alarm Bayesian Network.
    
    Network structure:
        Burglary → Alarm ← Earthquake
                     ↓
          JohnCalls  MaryCalls
    
    Story: You have a burglar alarm at home. It can be triggered by a burglary
    or an earthquake. If the alarm goes off, your neighbors John and Mary
    might call you (independently).
    """
    bn = BayesianNetwork()
    
    # Add variables
    bn.add_variable('Burglary', [True, False])
    bn.add_variable('Earthquake', [True, False])
    bn.add_variable('Alarm', [True, False])
    bn.add_variable('JohnCalls', [True, False])
    bn.add_variable('MaryCalls', [True, False])
    
    # Add edges
    bn.add_edge('Burglary', 'Alarm')
    bn.add_edge('Earthquake', 'Alarm')
    bn.add_edge('Alarm', 'JohnCalls')
    bn.add_edge('Alarm', 'MaryCalls')
    
    # Set CPTs
    # P(Burglary)
    bn.set_cpt('Burglary', {
        True: 0.001,
        False: 0.999
    })
    
    # P(Earthquake)
    bn.set_cpt('Earthquake', {
        True: 0.002,
        False: 0.998
    })
    
    # P(Alarm | Burglary, Earthquake)
    bn.set_cpt('Alarm', {
        (True, True, True): 0.95,    # Alarm=T, Burglary=T, Earthquake=T
        (False, True, True): 0.05,
        (True, True, False): 0.94,   # Alarm=T, Burglary=T, Earthquake=F
        (False, True, False): 0.06,
        (True, False, True): 0.29,   # Alarm=T, Burglary=F, Earthquake=T
        (False, False, True): 0.71,
        (True, False, False): 0.001, # Alarm=T, Burglary=F, Earthquake=F
        (False, False, False): 0.999
    })
    
    # P(JohnCalls | Alarm)
    bn.set_cpt('JohnCalls', {
        (True, True): 0.90,    # JohnCalls=T, Alarm=T
        (False, True): 0.10,
        (True, False): 0.05,   # JohnCalls=T, Alarm=F
        (False, False): 0.95
    })
    
    # P(MaryCalls | Alarm)
    bn.set_cpt('MaryCalls', {
        (True, True): 0.70,    # MaryCalls=T, Alarm=T
        (False, True): 0.30,
        (True, False): 0.01,   # MaryCalls=T, Alarm=F
        (False, False): 0.99
    })
    
    return bn


def demonstrate_inference():
    """Demonstrate Bayesian network inference."""
    print("=" * 60)
    print("Burglar Alarm Bayesian Network")
    print("=" * 60)
    
    bn = create_alarm_network()
    
    print("\nNetwork Structure:")
    print("  Burglary → Alarm ← Earthquake")
    print("               ↓")
    print("      JohnCalls   MaryCalls")
    
    # Example 1: Prior probability of burglary
    print("\n" + "=" * 60)
    print("Query 1: What's the probability of a burglary (no evidence)?")
    result = bn.enumerate_all('Burglary', {})
    print(f"  P(Burglary=True) = {result[True]:.6f}")
    print(f"  P(Burglary=False) = {result[False]:.6f}")
    
    # Example 2: Alarm went off
    print("\n" + "=" * 60)
    print("Query 2: Alarm went off. Is it a burglary?")
    result = bn.enumerate_all('Burglary', {'Alarm': True})
    print(f"  P(Burglary=True | Alarm=True) = {result[True]:.6f}")
    print(f"  P(Burglary=False | Alarm=True) = {result[False]:.6f}")
    print("  Interpretation: Alarm slightly increases burglary probability")
    
    # Example 3: John called
    print("\n" + "=" * 60)
    print("Query 3: John called. Is it a burglary?")
    result = bn.enumerate_all('Burglary', {'JohnCalls': True})
    print(f"  P(Burglary=True | JohnCalls=True) = {result[True]:.6f}")
    print(f"  P(Burglary=False | JohnCalls=True) = {result[False]:.6f}")
    
    # Example 4: Both John and Mary called
    print("\n" + "=" * 60)
    print("Query 4: Both John and Mary called. Is it a burglary?")
    result = bn.enumerate_all('Burglary', {'JohnCalls': True, 'MaryCalls': True})
    print(f"  P(Burglary=True | JohnCalls=True, MaryCalls=True) = {result[True]:.6f}")
    print(f"  P(Burglary=False | JohnCalls=True, MaryCalls=True) = {result[False]:.6f}")
    print("  Interpretation: Both calling significantly increases probability!")
    
    # Example 5: John called, but we know there was an earthquake
    print("\n" + "=" * 60)
    print("Query 5: John called, but there was an earthquake. Burglary?")
    result = bn.enumerate_all('Burglary', {'JohnCalls': True, 'Earthquake': True})
    print(f"  P(Burglary=True | JohnCalls=True, Earthquake=True) = {result[True]:.6f}")
    print(f"  P(Burglary=False | JohnCalls=True, Earthquake=True) = {result[False]:.6f}")
    print("  Interpretation: Earthquake explains the alarm, reducing burglary probability")
    
    # Demonstrate sampling
    print("\n" + "=" * 60)
    print("Sampling from the network (10 samples):")
    print("=" * 60)
    for i in range(10):
        sample = bn.sample()
        print(f"Sample {i+1}: {sample}")


def bayes_theorem_demo():
    """Demonstrate Bayes' Theorem with medical diagnosis example."""
    print("\n\n" + "=" * 60)
    print("Bayes' Theorem: Medical Diagnosis Example")
    print("=" * 60)
    
    # Disease testing scenario
    p_disease = 0.01        # 1% of population has disease
    p_pos_disease = 0.95    # 95% true positive rate
    p_neg_healthy = 0.90    # 90% true negative rate
    
    print(f"\nGiven:")
    print(f"  P(Disease) = {p_disease} (1% prevalence)")
    print(f"  P(Positive | Disease) = {p_pos_disease} (95% sensitivity)")
    print(f"  P(Negative | Healthy) = {p_neg_healthy} (90% specificity)")
    
    # Calculate P(Positive)
    p_healthy = 1 - p_disease
    p_pos_healthy = 1 - p_neg_healthy
    p_positive = p_pos_disease * p_disease + p_pos_healthy * p_healthy
    
    print(f"\nCalculate P(Positive) using law of total probability:")
    print(f"  P(Positive) = P(Pos|Disease)·P(Disease) + P(Pos|Healthy)·P(Healthy)")
    print(f"  P(Positive) = {p_pos_disease}·{p_disease} + {p_pos_healthy}·{p_healthy}")
    print(f"  P(Positive) = {p_positive:.4f}")
    
    # Apply Bayes' theorem
    p_disease_pos = (p_pos_disease * p_disease) / p_positive
    
    print(f"\nApply Bayes' Theorem:")
    print(f"  P(Disease | Positive) = P(Pos|Disease)·P(Disease) / P(Positive)")
    print(f"  P(Disease | Positive) = {p_pos_disease}·{p_disease} / {p_positive:.4f}")
    print(f"  P(Disease | Positive) = {p_disease_pos:.4f} ({p_disease_pos*100:.2f}%)")
    
    print(f"\nInterpretation:")
    print(f"  Even with a positive test, only {p_disease_pos*100:.1f}% chance of disease!")
    print(f"  This is because the disease is rare (low prior).")
    print(f"  Most positive tests are false positives from healthy people.")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_inference()
    bayes_theorem_demo()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("Bayesian networks allow us to:")
    print("  1. Represent uncertain knowledge compactly")
    print("  2. Reason about probabilities given evidence")
    print("  3. Make predictions under uncertainty")
    print("  4. Update beliefs as new information arrives")
    print("\nKey concepts demonstrated:")
    print("  - Conditional independence")
    print("  - Bayes' theorem")
    print("  - Exact inference by enumeration")
    print("  - Prior and posterior probabilities")
