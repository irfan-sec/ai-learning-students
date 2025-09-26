# Week 5: Knowledge Representation & Reasoning I (Logic)

## 📚 Learning Objectives
By the end of this week, students will be able to:
- Understand the role of knowledge representation in AI systems
- Master propositional logic syntax, semantics, and inference methods
- Apply model checking and logical inference to solve problems
- Understand the basics of first-order logic (FOL) including syntax and quantifiers
- Implement simple logical reasoning systems in Python

---

## 🧠 The Role of Knowledge in AI

**Knowledge Representation** is fundamental to building intelligent systems that can reason about the world.

### Why Do We Need Knowledge Representation?

1. **Explicit Reasoning:** Unlike neural networks, symbolic AI allows us to see and understand the reasoning process
2. **Interpretability:** We can trace through logical steps and explain decisions
3. **Generalization:** Rules learned in one domain can often apply to similar situations
4. **Integration:** Combine learned facts with expert knowledge

### Knowledge-Based Agents

A **knowledge-based agent** maintains an internal knowledge base (KB) and uses logical inference to decide what actions to take.

**Agent Architecture:**
```
Sensors → Knowledge Base ← Inference Engine
                ↓
            Decision Making
                ↓
            Actuators
```

---

## 🔣 Propositional Logic

### Syntax of Propositional Logic

**Atomic Sentences (Propositions):**
- `P`, `Q`, `R`, `Rain`, `Sunny`, `WumpusAlive`
- Each represents a fact that is either true or false

**Logical Connectives:**
- **Negation:** `¬P` (not P)
- **Conjunction:** `P ∧ Q` (P and Q)
- **Disjunction:** `P ∨ Q` (P or Q)  
- **Implication:** `P → Q` (if P then Q)
- **Biconditional:** `P ↔ Q` (P if and only if Q)

**Complex Sentences:**
- `(Rain ∧ ¬Umbrella) → Wet`
- `(Smoke → Fire) ∧ (Fire → Alarm)`

### Semantics: Truth Tables

Truth values determine the meaning of logical sentences:

| P | Q | ¬P | P∧Q | P∨Q | P→Q | P↔Q |
|---|---|----|----|----|----|-----|
| T | T | F  | T  | T  | T  | T   |
| T | F | F  | F  | T  | F  | F   |
| F | T | T  | F  | T  | T  | F   |
| F | F | T  | F  | F  | T  | T   |

**Key Insight:** `P → Q` is equivalent to `¬P ∨ Q`

---

## 🔍 Logical Inference

### Model Checking

A **model** is an assignment of truth values to all propositional symbols.

**Entailment:** KB ⊨ α (KB entails α)
- α is true in all models where KB is true
- If we know KB is true, then α must also be true

**Model Checking Algorithm:**
```python
def model_check(kb, query):
    """Check if KB entails query using brute force"""
    symbols = get_symbols(kb, query)
    return check_all(kb, query, symbols, {})

def check_all(kb, query, symbols, model):
    if not symbols:  # Base case
        if is_true(kb, model):
            return is_true(query, model)
        else:
            return True  # KB is false, so anything follows
    
    # Recursive case
    symbol = symbols[0]
    rest = symbols[1:]
    
    # Try symbol = True
    model_true = model.copy()
    model_true[symbol] = True
    result_true = check_all(kb, query, rest, model_true)
    
    # Try symbol = False  
    model_false = model.copy()
    model_false[symbol] = False
    result_false = check_all(kb, query, rest, model_false)
    
    return result_true and result_false
```

### Logical Equivalences

Important equivalences for simplifying formulas:

- **Double Negation:** `¬¬P ≡ P`
- **De Morgan's Laws:** 
  - `¬(P ∧ Q) ≡ (¬P ∨ ¬Q)`
  - `¬(P ∨ Q) ≡ (¬P ∧ ¬Q)`
- **Distributivity:**
  - `P ∧ (Q ∨ R) ≡ (P ∧ Q) ∨ (P ∧ R)`
  - `P ∨ (Q ∧ R) ≡ (P ∨ Q) ∧ (P ∨ R)`

---

## 🌍 Introduction to First-Order Logic (FOL)

Propositional logic is limited - it can't express relationships or generalize over objects.

### FOL Syntax

**Objects:** Constants like `John`, `Mary`, `Stanford`

**Predicates:** Relations like `Student(x)`, `Teaches(x,y)`, `Loves(x,y)`

**Functions:** `Father(x)`, `Age(x)`, `ColorOf(x)`

**Quantifiers:**
- **Universal:** `∀x P(x)` (for all x, P(x))
- **Existential:** `∃x P(x)` (there exists an x such that P(x))

**Example Sentences:**
- `∀x Student(x) → Person(x)` (All students are persons)
- `∃x Student(x) ∧ Smart(x)` (There exists a smart student)
- `∀x,y Father(x,y) → Male(x)` (All fathers are male)

### Scope and Binding

Variables can be **bound** (by quantifiers) or **free**:
- `∀x Loves(x, Mary)` - x is bound
- `Loves(John, y)` - y is free

---

## 💻 Practical Example: Wumpus World Logic

Consider the classic Wumpus World where an agent uses logic to navigate safely:

**Knowledge Base:**
```
1. ¬P₁,₁                    (No pit in [1,1])
2. B₁,₁ ↔ (P₁,₂ ∨ P₂,₁)     (Breeze iff adjacent pit)
3. B₂,₁ ↔ (P₁,₁ ∨ P₂,₂ ∨ P₃,₁) (Breeze rules for [2,1])
4. ¬B₁,₁                    (No breeze observed in [1,1])
5. B₂,₁                     (Breeze observed in [2,1])
```

**Inference:**
- From 2 and 4: `¬P₁,₂ ∧ ¬P₂,₁` (No pits adjacent to [1,1])
- From 3 and 5: `P₂,₂ ∨ P₃,₁` (Pit in [2,2] or [3,1])

---

## 🤔 Discussion Questions

1. **Expressiveness:** What types of knowledge are difficult to represent in propositional logic?

2. **Computational Complexity:** Model checking is exponential in the number of variables. How might we make inference more efficient?

3. **Real-world Applications:** Where do you see logical reasoning being used in modern AI systems?

4. **Limitations:** What are the main limitations of purely logical approaches to AI?

---

## 🔍 Looking Ahead

Next week, we'll explore how to handle **uncertainty** in reasoning - the real world is rarely as black and white as pure logic assumes!

**Key Connections:**
- Week 6: From certain logic to probabilistic reasoning
- Week 8-11: How machine learning complements symbolic reasoning
- Week 13: Ethical implications of automated reasoning systems