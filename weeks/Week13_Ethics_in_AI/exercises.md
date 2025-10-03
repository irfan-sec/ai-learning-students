# Week 13 Exercises: Ethics in AI

## 🎯 Learning Goals
- Identify ethical issues in AI
- Understand bias in ML
- Design fair systems
- Consider societal impacts

## 📝 Discussion Questions

### Case Study 1: Hiring Algorithm
An AI system screens job applications but discriminates against women.

**Discuss:**
a) How could this bias arise?
b) How to detect it?
c) Mitigation strategies?
d) Is removing gender enough?

### Case Study 2: Autonomous Vehicles
The trolley problem: choosing between unavoidable accidents.

**Discuss:**
a) Who decides ethical rules?
b) How to prioritize different lives?
c) Passenger safety vs total harm?

### Case Study 3: Facial Recognition
Police want to use facial recognition for public safety.

**Discuss:**
a) Benefits and risks?
b) Accuracy across demographics?
c) Regulation needed?
d) Privacy vs security?

## 💻 Practical Exercises

### Exercise 1: Audit for Bias
```python
def audit_model_fairness(model, X_test, y_test, sensitive_attr):
    """Analyze performance across different groups."""
    # TODO: Calculate metrics per group
    # TODO: Identify disparities
    pass
```

### Exercise 2: Model Interpretability
```python
from sklearn.inspection import permutation_importance

def explain_predictions(model, X, feature_names):
    # TODO: Feature importance
    # TODO: Generate explanations
    pass
```

## 📝 Essay Assignment

**Write 1000 words on ONE topic:**
1. "Should AI systems provide explanations?"
2. "Ensuring AI benefits all humanity"
3. "Regulations for AI development"
4. "AI and inequality"

---
**Due Date: End of Week 13**
