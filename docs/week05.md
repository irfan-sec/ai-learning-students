---
layout: default
title: "Week 5: Knowledge Representation & Reasoning I (Logic)"
permalink: /week05.html
---

### Navigation
[🏠 Home](index.html) | [← Previous: Week 4](week04.html) | [Next: Week 6 →](week06.html)

---

# Week 5: Knowledge Representation & Reasoning I (Logic)

## 📚 Learning Objectives
By the end of this week, students will be able to:
- Understand the principles of knowledge representation in AI
- Work with propositional logic and first-order logic
- Implement basic inference algorithms
- Apply logical reasoning to solve AI problems
- Design knowledge bases for expert systems

---

## 🧠 What is Knowledge Representation?

Knowledge representation is a fundamental area of AI that deals with how to formally encode information about the world in a form that AI systems can use to make decisions and solve problems.

![Knowledge Representation](images/knowledge-representation.png)

### Key Components
- **Syntax:** How to write statements in the language
- **Semantics:** What those statements mean in the world
- **Inference:** How to derive new knowledge from existing knowledge

---

## 🔤 Propositional Logic

### Basic Concepts
- **Propositions:** Statements that are either true or false
- **Logical Connectives:** AND (∧), OR (∨), NOT (¬), IMPLIES (→), BICONDITIONAL (↔)
- **Truth Tables:** Define meaning of logical connectives

### Example Implementation

```python
class PropositionalLogic:
    def __init__(self):
        self.knowledge_base = set()
    
    def add_rule(self, rule):
        """Add a logical rule to the knowledge base"""
        self.knowledge_base.add(rule)
    
    def query(self, proposition):
        """Query whether a proposition is true given the KB"""
        # Implementation would involve inference algorithms
        pass
```

---

## 🎯 First-Order Logic (Predicate Logic)

First-order logic extends propositional logic by introducing:
- **Objects:** Things in the world (people, places, concepts)
- **Relations:** Relationships between objects
- **Functions:** Mappings from objects to objects
- **Quantifiers:** Universal (∀) and existential (∃)

### Example: Family Relationships

```python
# Facts about family relationships
facts = [
    "Parent(John, Mary)",
    "Parent(John, Tom)", 
    "Parent(Mary, Alice)",
    "Male(John)",
    "Male(Tom)",
    "Female(Mary)",
    "Female(Alice)"
]

# Rules for deriving new relationships
rules = [
    "∀x,y Parent(x,y) → Ancestor(x,y)",
    "∀x,y,z Parent(x,y) ∧ Ancestor(y,z) → Ancestor(x,z)",
    "∀x,y Parent(x,y) ∧ Male(x) → Father(x,y)",
    "∀x,y Parent(x,y) ∧ Female(x) → Mother(x,y)"
]
```

---

## 🔍 Inference Algorithms

### Forward Chaining
- Start with facts and apply rules to derive new facts
- Continue until no more facts can be derived or goal is reached

### Backward Chaining
- Start with goal and work backwards to find supporting facts
- Used in many expert systems

### Resolution Theorem Proving
- Convert statements to clause form
- Apply resolution rule to derive contradictions
- Used in automated theorem provers

---

## 💻 Hands-On Exercise: Simple Expert System

```python
class SimpleExpertSystem:
    """A basic expert system using propositional logic"""
    
    def __init__(self):
        self.facts = set()
        self.rules = []
    
    def add_fact(self, fact):
        """Add a known fact"""
        self.facts.add(fact)
    
    def add_rule(self, condition, conclusion):
        """Add an if-then rule"""
        self.rules.append((condition, conclusion))
    
    def infer(self):
        """Apply forward chaining to infer new facts"""
        changed = True
        while changed:
            changed = False
            for condition, conclusion in self.rules:
                if self.evaluate_condition(condition) and conclusion not in self.facts:
                    self.facts.add(conclusion)
                    changed = True
    
    def evaluate_condition(self, condition):
        """Evaluate if a condition is satisfied by current facts"""
        # Simplified evaluation - in practice would parse logical expressions
        return condition in self.facts
    
    def query(self, fact):
        """Query if a fact can be inferred"""
        self.infer()
        return fact in self.facts

# Example: Medical diagnosis system
expert_system = SimpleExpertSystem()

# Add symptoms as facts
expert_system.add_fact("fever")
expert_system.add_fact("cough")
expert_system.add_fact("headache")

# Add diagnostic rules
expert_system.add_rule("fever", "possible_infection")
expert_system.add_rule("possible_infection", "recommend_rest")
expert_system.add_rule("cough", "respiratory_symptoms")

# Query the system
if expert_system.query("recommend_rest"):
    print("System recommends rest")
```

---

## 🔗 Curated Resources

### Essential Reading
- [AIMA Chapter 7-9](https://aima.cs.berkeley.edu/) - Logical Agents and Knowledge Representation
- [Logic in AI - Stanford](https://plato.stanford.edu/entries/logic-ai/) - Philosophical foundations

### Video Resources
- [Propositional Logic](https://www.youtube.com/watch?v=1bWry5mx2C8) - Introduction to formal logic
- [First-Order Logic](https://www.youtube.com/watch?v=gyoqX0W-NH4) - Predicate logic explained
- [Expert Systems](https://www.youtube.com/watch?v=taxfmj0pNwE) - Historical perspective

### Interactive Learning
- [Logic Puzzle Games](https://www.logicpuzzles.org/) - Practice logical reasoning
- [Prolog Online](https://swish.swi-prolog.org/) - Try logic programming
- [Logic Grid Puzzles](https://www.puzzle-bridges.com/) - Applied reasoning

---

## 🎯 Key Concepts Summary

### Logic Fundamentals
- **Syntax and Semantics:** Formal representation of knowledge
- **Inference:** Deriving new knowledge from existing knowledge
- **Soundness and Completeness:** Properties of inference systems

### Applications
- **Expert Systems:** Codify human expertise
- **Automated Theorem Proving:** Verify mathematical proofs
- **Semantic Web:** Machine-readable web content

---

## 🤔 Discussion Questions

1. What are the advantages and limitations of logical representation?
2. How do expert systems differ from modern machine learning approaches?
3. When is first-order logic necessary vs. propositional logic?
4. What role does logic play in modern AI systems?

---

## 🔍 Looking Ahead

Next week, we'll explore **reasoning under uncertainty** using probability theory and Bayesian networks. We'll learn how to handle incomplete and uncertain information - a crucial skill for real-world AI applications.

---

### Navigation
[🏠 Home](index.html) | [← Previous: Week 4](week04.html) | [Next: Week 6 →](week06.html)

---

*"Logic is the beginning of wisdom, not the end."* - Leonard Nimoy