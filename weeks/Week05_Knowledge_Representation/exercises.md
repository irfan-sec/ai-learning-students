# Week 5 Exercises: Knowledge Representation & Reasoning I (Logic)

## 🎯 Learning Goals Review
After completing these exercises, you should be able to:
- Convert natural language statements to propositional and first-order logic
- Apply logical inference rules and model checking
- Build simple knowledge bases and reason with them
- Understand the strengths and limitations of logical representation

---

## 📝 Propositional Logic Exercises

### Exercise 1: Logical Translation

Translate the following English statements into propositional logic. Define your propositions clearly.

**Scenario:** A smart home system with sensors and devices.

**Propositions to define:**
- `R` = Rain is detected
- `W` = Windows are open  
- `H` = Heating is on
- `C` = Air conditioning is on
- `M` = Motion is detected
- `L` = Lights are on

**Translate these statements:**

1. "If it's raining and the windows are open, then close the windows."
   - **Your answer:** ________________________________

2. "The heating and air conditioning are never on at the same time."
   - **Your answer:** ________________________________

3. "If motion is detected, turn on the lights, unless it's daytime."
   - **Your answer:** ________________________________ (add `D` = Daytime)

4. "The system is in eco-mode if and only if no motion is detected and it's daytime."
   - **Your answer:** ________________________________ (add `E` = Eco-mode)

### Exercise 2: Truth Tables

Create truth tables for the following formulas:

**Formula A:** `(P → Q) ↔ (¬P ∨ Q)`

| P | Q | P→Q | ¬P | ¬P∨Q | (P→Q)↔(¬P∨Q) |
|---|---|-----|----|----- |---------------|
| T | T |     |    |      |               |
| T | F |     |    |      |               |
| F | T |     |    |      |               |
| F | F |     |    |      |               |

**Formula B:** `¬(P ∧ Q) ↔ (¬P ∨ ¬Q)`

| P | Q | P∧Q | ¬(P∧Q) | ¬P | ¬Q | ¬P∨¬Q | ¬(P∧Q)↔(¬P∨¬Q) |
|---|---|-----|--------|----|----|-------|-----------------|
| T | T |     |        |    |    |       |                 |
| T | F |     |        |    |    |       |                 |
| F | T |     |        |    |    |       |                 |
| F | F |     |        |    |    |       |                 |

**Question:** What do these truth tables tell you about logical equivalence?

### Exercise 3: Model Checking Practice

Given this knowledge base about a university:
```
KB = {
    StudentTakesCS ∨ StudentTakesMath,
    StudentTakesCS → GoodWithComputers, 
    StudentTakesMath → GoodWithNumbers,
    ¬(GoodWithComputers ∧ GoodWithNumbers)
}
```

**Questions:**
1. Does KB ⊨ StudentTakesCS? (Show your work by considering all models)
2. Does KB ⊨ ¬StudentTakesMath?
3. Is the KB consistent (satisfiable)?

**Your analysis:**
- **Models where KB is true:** ________________________________
- **Answer to Q1:** ________________________________
- **Answer to Q2:** ________________________________
- **Answer to Q3:** ________________________________

---

## 🌍 First-Order Logic Exercises

### Exercise 4: FOL Translation

Translate these English statements into First-Order Logic:

**Domain:** University with students, professors, and courses

**Predicates:**
- `Student(x)`, `Professor(x)`, `Course(x)`
- `Enrolled(x,y)` - x is enrolled in course y
- `Teaches(x,y)` - x teaches course y
- `Prerequisite(x,y)` - x is prerequisite for y
- `Completed(x,y)` - x has completed course y

**Translate:**

1. "All students are enrolled in at least one course."
   - **Your answer:** ________________________________

2. "No student teaches a course."
   - **Your answer:** ________________________________

3. "Every course is taught by exactly one professor."
   - **Your answer:** ________________________________

4. "A student can only enroll in a course if they have completed all prerequisites."
   - **Your answer:** ________________________________

5. "There exists a student who has completed all courses."
   - **Your answer:** ________________________________

### Exercise 5: Quantifier Scope

For each formula, identify the scope of each quantifier and determine which variables are bound vs. free:

1. `∀x (Student(x) → ∃y (Course(y) ∧ Enrolled(x,y)))`
   - **Scope of ∀x:** ________________________________
   - **Scope of ∃y:** ________________________________
   - **Bound variables:** ________________________________
   - **Free variables:** ________________________________

2. `∀x ∃y (Teaches(x,y) → Professor(x)) ∧ Student(z)`
   - **Scope of ∀x:** ________________________________
   - **Scope of ∃y:** ________________________________
   - **Bound variables:** ________________________________
   - **Free variables:** ________________________________

---

## 💻 Programming Exercises

### Exercise 6: Simple Model Checker

Implement a basic model checker in Python:

```python
def evaluate_formula(formula, model):
    """
    Evaluate a propositional logic formula given a model.
    
    Args:
        formula: String representation (e.g., "P and Q")
        model: Dictionary mapping variables to boolean values
    
    Returns:
        Boolean value of the formula
    """
    # TODO: Implement this function
    pass

def model_check(kb_formulas, query_formula):
    """
    Check if KB entails query using model checking.
    
    Args:
        kb_formulas: List of formula strings representing KB
        query_formula: String representing the query
    
    Returns:
        True if KB entails query, False otherwise
    """
    # TODO: Implement this function
    # Hint: Generate all possible models and check each one
    pass

# Test your implementation
kb = ["P or Q", "P implies R", "Q implies R"]
query = "R"
result = model_check(kb, query)
print(f"Does KB entail {query}? {result}")
```

**Your implementation:** Complete the functions above.

### Exercise 7: Knowledge Base Builder

Create a simple knowledge base for the Wumpus World:

```python
class WumpusKB:
    def __init__(self):
        self.facts = set()
    
    def tell(self, fact):
        """Add a fact to the knowledge base"""
        # TODO: Implement
        pass
    
    def ask(self, query):
        """Query the knowledge base"""
        # TODO: Implement basic inference
        pass
    
    def add_breeze_rule(self, x, y):
        """Add rule: Breeze at (x,y) iff pit in adjacent cell"""
        # TODO: Implement
        pass

# Example usage
kb = WumpusKB()
kb.tell("not Pit_1_1")  # Agent starts safely
kb.tell("not Breeze_1_1")  # No breeze observed
kb.add_breeze_rule(1, 1)  # Add the breeze rule

# What can we infer?
print(kb.ask("not Pit_1_2"))  # Should we avoid (1,2)?
```

---

## 🧩 Real-World Applications

### Exercise 8: Smart Home Logic

Design a logical knowledge base for a smart home system.

**Requirements:**
1. Define at least 10 propositions about sensors, devices, and conditions
2. Write at least 8 logical rules that govern system behavior
3. Show how the system would reason about a specific scenario

**Your Design:**

**Propositions:**
1. ________________________________
2. ________________________________
3. ________________________________
... (continue for 10 total)

**Rules:**
1. ________________________________
2. ________________________________
3. ________________________________
... (continue for 8 total)

**Scenario Analysis:**
Pick a specific scenario (e.g., "user comes home on a rainy evening") and trace through the logical inference to show what actions the system would take.

### Exercise 9: Medical Diagnosis

Create a simple diagnostic system using propositional logic:

**Symptoms:** `Fever`, `Cough`, `Headache`, `Fatigue`, `Nausea`
**Conditions:** `Cold`, `Flu`, `Migraine`, `FoodPoisoning`

**Your task:**
1. Write logical rules connecting symptoms to conditions
2. Given a set of observed symptoms, what can you conclude?
3. What are the limitations of this approach?

**Your rules:**
1. ________________________________
2. ________________________________
3. ________________________________
... (add more as needed)

**Test case:** Patient has `Fever`, `Headache`, and `Fatigue`. What's your diagnosis?

---

## 🤝 Discussion Questions

### Question 1: Limitations of Logic
List three real-world reasoning scenarios where pure propositional logic would be inadequate. For each, explain why logic falls short and what additional capabilities would be needed.

### Question 2: FOL vs. Propositional Logic
Compare first-order logic to propositional logic. In what situations would you choose each? Give specific examples.

### Question 3: Computational Complexity
Model checking is NP-complete for propositional logic. How does this limitation affect the practical use of logic in AI systems? What strategies can help manage this complexity?

### Question 4: Integration with ML
How might logical reasoning complement machine learning approaches in modern AI systems? Provide a concrete example.

---

## 🏆 Challenge Problems (Optional)

### Challenge 1: Logic Puzzle Solver
Implement a solver for logic grid puzzles (like Einstein's riddle). Your solver should:
- Parse natural language clues into logical constraints
- Use systematic reasoning to find the unique solution
- Explain the reasoning steps taken

### Challenge 2: Resolution Theorem Prover
Implement a basic resolution theorem prover for propositional logic:
- Convert formulas to Conjunctive Normal Form (CNF)
- Apply the resolution rule systematically
- Determine satisfiability or prove unsatisfiability

### Challenge 3: Natural Language Interface
Create a system that can:
- Parse simple English statements into logical formulas
- Build a knowledge base from these statements
- Answer questions by logical inference
- Explain its reasoning in natural language

---

## 📝 Submission Guidelines

### What to Submit:
1. **Completed exercises** with clear solutions and explanations
2. **Python code** for programming exercises (well-commented)
3. **Design document** for real-world application exercise
4. **Reflection essay** (300-500 words) on the strengths and limitations of logical reasoning in AI

### Evaluation Criteria:
- **Correctness:** Are logical formulas and inferences correct?
- **Clarity:** Are explanations clear and well-reasoned?
- **Creativity:** Do applications show understanding beyond basic examples?
- **Code Quality:** Is code clean, documented, and working?

### Due Date: End of Week 5

---

**💡 Study Tips:**
- Practice converting between natural language and logic daily
- Use truth tables to verify your logical formulas
- Think about how logical reasoning applies to systems you use every day
- Don't just memorize rules - understand the underlying principles!