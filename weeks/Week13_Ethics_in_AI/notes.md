# Week 13: Ethics in AI

## 📚 Learning Objectives
By the end of this week, students will be able to:
- Understand the ethical implications of AI systems in society
- Identify sources of bias in AI algorithms and data
- Explain concepts of fairness, accountability, and transparency in AI
- Analyze real-world case studies of AI ethical dilemmas
- Discuss approaches to responsible AI development and deployment

---

## ⚖️ Why AI Ethics Matters

As AI systems become increasingly prevalent in society, they impact human lives in profound ways. From hiring decisions to criminal justice, from healthcare to financial services, AI is no longer just a technical challenge—it's a societal responsibility.

![AI Ethics Overview](https://miro.medium.com/max/1400/1*9jQD7JG_zJ6tFGZ8kl7M6g.png)

### The Stakes Are High
- **Individual Impact:** AI decisions affect careers, health, freedom, and opportunities
- **Societal Impact:** AI can perpetuate or amplify existing inequalities  
- **Global Impact:** AI systems shape information, democracy, and economic structures
- **Future Impact:** Today's AI decisions set precedents for tomorrow's more powerful systems

---

## 🔍 Core Ethical Principles in AI

### 1. **Fairness and Non-Discrimination**
AI systems should not unfairly discriminate against individuals or groups.

**Key Questions:**
- Does the system treat all groups equitably?
- Are outcomes distributed fairly across demographics?
- Does the system perpetuate historical biases?

**Example:** A hiring algorithm that systematically rejects qualified women candidates because it was trained on historical data from male-dominated industries.

### 2. **Transparency and Explainability**
People should understand how AI systems make decisions that affect them.

**Levels of Transparency:**
- **Algorithmic:** How does the system work internally?
- **Procedural:** What process led to this decision?
- **Outcome:** Why was this particular decision made?

**Challenge:** Complex models (deep learning) often lack interpretability.

### 3. **Accountability and Responsibility**
There must be clear lines of responsibility for AI system outcomes.

**Key Questions:**
- Who is responsible when AI makes a mistake?
- How do we ensure proper oversight?
- What recourse do affected individuals have?

### 4. **Privacy and Data Protection**
AI systems must respect individual privacy and data rights.

**Privacy Concerns:**
- **Data collection:** What data is gathered and how?
- **Data use:** How is personal data used in training and inference?
- **Data sharing:** Who has access to personal information?
- **Re-identification:** Can anonymized data be traced back to individuals?

### 5. **Human Autonomy and Agency**
AI should augment rather than replace human decision-making in critical areas.

**Considerations:**
- Maintaining human oversight in important decisions
- Preserving human skills and capabilities
- Ensuring humans can override AI recommendations

---

## 📊 Sources of Bias in AI Systems

### 1. **Historical Bias in Training Data**
AI systems learn from historical data, which often reflects past discrimination.

**Examples:**
- **Resume screening:** Historical hiring practices favor certain demographics
- **Credit scoring:** Past lending discrimination embedded in training data
- **Criminal justice:** Historical arrest patterns reflect biased policing

```python
# Example: Biased hiring data
historical_hires = {
    'software_engineer': {'male': 90, 'female': 10},  # Historical bias
    'nurse': {'male': 15, 'female': 85},
    'teacher': {'male': 25, 'female': 75}
}

# AI trained on this data will perpetuate these patterns
```

### 2. **Representation Bias**
Training data may not represent all groups equally.

**Examples:**
- **Facial recognition:** Poor performance on darker skin tones
- **Voice recognition:** Biased toward certain accents or dialects
- **Medical AI:** Trained primarily on data from one demographic group

### 3. **Measurement Bias**
Different groups may be measured or evaluated differently.

**Examples:**
- **Standardized tests:** May not measure intelligence equally across cultures
- **Performance reviews:** Subjective evaluations may contain unconscious bias
- **Health indicators:** May have different meanings across populations

### 4. **Evaluation Bias**
Using inappropriate benchmarks or metrics for different groups.

**Example:** A model optimized for overall accuracy might perform poorly on minority groups if they're underrepresented in the dataset.

---

## 🎯 Algorithmic Fairness

### Defining Fairness

**Statistical Parity (Demographic Parity)**
```
P(Ŷ = 1 | A = 0) = P(Ŷ = 1 | A = 1)
```
Equal probability of positive outcome across groups.

**Equal Opportunity**
```
P(Ŷ = 1 | A = 0, Y = 1) = P(Ŷ = 1 | A = 1, Y = 1)
```
Equal true positive rates across groups.

**Equalized Odds**
```
P(Ŷ = 1 | A = 0, Y = y) = P(Ŷ = 1 | A = 1, Y = y) for y ∈ {0,1}
```
Equal true positive and false positive rates across groups.

### The Impossibility of Fairness
**Important:** These fairness criteria often conflict with each other! Perfect fairness across all metrics simultaneously is often mathematically impossible.

---

## 📱 Real-World Case Studies

### Case Study 1: Amazon's Hiring Algorithm (2018)

**Background:** Amazon developed an AI system to rank job candidates' resumes.

**The Problem:**
- System was trained on 10 years of historical hiring data
- Historical data reflected male-dominated tech industry
- Algorithm learned to penalize resumes mentioning "women" (e.g., "women's chess club captain")

**Ethical Issues:**
- **Bias amplification:** System amplified existing gender bias
- **Lack of diversity:** Reinforced homogeneous hiring patterns
- **Transparency:** Candidates unaware of algorithmic evaluation

**Outcome:** Amazon scrapped the system after discovering the bias.

**Lessons:**
- Historical data can perpetuate discrimination
- Bias testing is crucial before deployment
- Diverse development teams help identify problems

### Case Study 2: COMPAS Recidivism Prediction

**Background:** COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) predicts likelihood of reoffending.

**The Controversy:**
- System showed higher false positive rates for Black defendants
- Black defendants more likely to be incorrectly labeled high-risk
- White defendants more likely to be incorrectly labeled low-risk

**Ethical Dilemma:**
- **Predictive accuracy:** System was statistically accurate overall
- **Disparate impact:** But outcomes differed significantly by race
- **Lack of transparency:** Algorithm details were proprietary

**Ongoing Debate:**
- Should we use accurate but biased systems?
- How do we balance efficiency with fairness?
- What level of transparency is required for criminal justice AI?

### Case Study 3: Facial Recognition and Surveillance

**Background:** Widespread deployment of facial recognition technology.

**Ethical Concerns:**

**Bias Issues:**
- Higher error rates for women and people of color
- Led to false arrests and wrongful detentions

**Privacy Concerns:**
- Constant surveillance without consent
- Potential for government overreach
- Data security and misuse risks

**Societal Impact:**
- Chilling effect on freedom of assembly
- Disproportionate impact on marginalized communities
- Normalization of surveillance

**Responses:**
- Some cities banned facial recognition by government agencies
- Companies restricted sale of facial recognition technology
- Calls for regulation and oversight

---

## 🛡️ Approaches to Responsible AI

### 1. **Diverse and Inclusive Development Teams**

**Why It Matters:**
- Different perspectives identify different problems
- Inclusive teams build more inclusive systems
- Representation matters at all levels

**Best Practices:**
- Diverse hiring in AI roles
- Include social scientists, ethicists, domain experts
- Community engagement and stakeholder input

### 2. **Bias Detection and Mitigation**

**Pre-processing Approaches:**
```python
# Example: Data balancing
def balance_dataset(X, y, sensitive_attribute):
    # Ensure equal representation across groups
    # Oversample underrepresented groups
    # Remove biased features
    pass
```

**In-processing Approaches:**
```python
# Example: Fairness constraints
def train_fair_classifier(X, y, sensitive_attribute):
    # Add fairness constraints to optimization objective
    # Regularize for equal opportunity
    # Multi-objective optimization (accuracy + fairness)
    pass
```

**Post-processing Approaches:**
```python
# Example: Threshold adjustment
def adjust_thresholds(predictions, sensitive_attribute):
    # Set different decision thresholds for different groups
    # Calibrate to achieve equal opportunity
    # Trade off some accuracy for fairness
    pass
```

### 3. **Explainable AI (XAI)**

**Techniques:**
- **Local explanations:** Why this particular decision?
- **Global explanations:** How does the system work overall?
- **Feature importance:** Which factors mattered most?
- **Counterfactual explanations:** What would change the outcome?

**Example: LIME (Local Interpretable Model-agnostic Explanations)**
```python
import lime
from lime import lime_text

# Explain individual text classification
explainer = lime_text.LimeTextExplainer()
explanation = explainer.explain_instance(
    text_instance, 
    classifier.predict_proba,
    num_features=10
)
```

### 4. **AI Governance and Oversight**

**Internal Governance:**
- Ethics review boards
- Regular bias audits
- Impact assessments
- Clear policies and procedures

**External Oversight:**
- Regulatory frameworks
- Industry standards
- Third-party audits
- Public accountability

### 5. **Human-in-the-Loop Systems**

**Design Principles:**
- Humans maintain ultimate decision authority
- AI provides recommendations, not final decisions
- Clear handoff procedures between AI and humans
- Continuous human oversight and monitoring

---

## 🌍 Global Perspectives on AI Ethics

### European Union: GDPR and AI Act

**GDPR (General Data Protection Regulation):**
- Right to explanation for automated decision-making
- Data protection by design and by default
- Individual consent and control over data

**Proposed AI Act:**
- Risk-based regulation of AI systems
- Prohibited AI practices (social scoring, subliminal manipulation)
- High-risk system requirements (healthcare, criminal justice)

### United States: Sectoral Approach

**NIST AI Risk Management Framework:**
- Voluntary guidelines for AI development
- Focus on risk management and mitigation
- Emphasis on trustworthy AI characteristics

**Agency-Specific Regulations:**
- FDA for medical AI
- EEOC for employment AI
- FTC for consumer protection

### China: National AI Strategy

**Focus Areas:**
- AI leadership and innovation
- Data governance and security
- Social stability and control

**Unique Approaches:**
- Social credit systems
- AI governance through party structure
- Emphasis on collective benefits

---

## 🤔 Ethical Dilemmas in AI

### Dilemma 1: Accuracy vs. Fairness
**Scenario:** A medical AI system is highly accurate overall but performs worse for certain ethnic groups.

**Questions:**
- Is it ethical to use a system that could save many lives but disadvantages some groups?
- How do we balance overall benefit against individual fairness?
- What if improving fairness reduces accuracy for everyone?

### Dilemma 2: Privacy vs. Public Health
**Scenario:** Contact tracing apps during COVID-19 pandemic.

**Questions:**
- Is privacy sacrifice justified for public health?
- Who controls the data and for how long?
- How do we prevent mission creep?

### Dilemma 3: Automation vs. Employment
**Scenario:** AI systems can replace human workers in many industries.

**Questions:**
- Do we have a moral obligation to preserve jobs?
- How do we manage the transition for displaced workers?
- What is society's responsibility for those affected by automation?

---

## 💼 Practical Frameworks for Ethical AI

### 1. **Ethics by Design Checklist**

**Before Development:**
- [ ] Define clear purpose and scope
- [ ] Identify potential stakeholders and impacts
- [ ] Assess data sources for bias
- [ ] Plan for transparency and explainability

**During Development:**
- [ ] Test for bias across different groups
- [ ] Implement fairness metrics
- [ ] Document design decisions
- [ ] Include diverse perspectives in review

**Before Deployment:**
- [ ] Conduct impact assessment
- [ ] Plan monitoring and evaluation
- [ ] Establish feedback mechanisms
- [ ] Train users on ethical considerations

**After Deployment:**
- [ ] Monitor performance across groups
- [ ] Address identified problems promptly
- [ ] Regular bias audits
- [ ] Update as needed

### 2. **Stakeholder Impact Analysis**

**Identify Stakeholders:**
- Direct users of the system
- People affected by system decisions
- Organizations implementing the system
- Society at large

**Assess Impacts:**
- Benefits and harms for each group
- Short-term and long-term effects
- Intended and unintended consequences
- Distribution of costs and benefits

---

## 🎯 Key Principles for Students

### As Future AI Practitioners

1. **Stay Informed:** Keep up with ethical developments in AI
2. **Think Critically:** Question assumptions and potential biases
3. **Collaborate:** Work with diverse teams and stakeholders
4. **Be Transparent:** Document decisions and trade-offs
5. **Take Responsibility:** Consider the broader impact of your work

### Questions to Ask

- **Who benefits** from this AI system?
- **Who might be harmed** by this system?
- **What biases** might exist in the data or algorithm?
- **How transparent** is the decision-making process?
- **What safeguards** are in place?
- **How will we know** if the system is working as intended?

---

## 🔮 Future Challenges

### Emerging Concerns

1. **Artificial General Intelligence:** What happens when AI surpasses human capability?
2. **Deepfakes and Misinformation:** How do we maintain truth in an age of synthetic media?
3. **AI Warfare:** What are the ethics of autonomous weapons?
4. **Economic Disruption:** How do we manage large-scale job displacement?
5. **Algorithmic Governance:** When should AI make societal decisions?

### Preparing for the Future

- Develop ethical frameworks that can adapt to new technologies
- Foster international cooperation on AI governance
- Invest in education and reskilling programs
- Maintain human agency in critical decisions
- Balance innovation with responsible development

---

## 🤝 Discussion Questions

1. Can AI systems ever be truly "objective" or are they always reflections of human biases?

2. Should there be a "right to explanation" for all AI decisions that affect individuals?

3. How do we balance the benefits of AI surveillance (crime prevention, public health) with privacy rights?

4. What responsibility do tech companies have for the societal impacts of their AI systems?

5. How can we ensure that AI benefits everyone, not just those with access to technology and data?

6. What role should the public play in governing AI development and deployment?

---

## 📚 Required Actions for Students

### Individual Reflection (Required)
Write a 2-page reflection on:
- A specific AI ethical issue that concerns you most
- How you would address this issue as an AI practitioner
- What policies or practices you would recommend

### Group Project (Optional)
Choose a current AI system and conduct an ethical analysis:
- Identify potential biases and their sources
- Assess fairness across different groups
- Propose improvements and safeguards
- Present findings to the class

---

*"With great power comes great responsibility. As AI becomes more powerful, our responsibility to develop and deploy it ethically becomes even greater."*

---

**Remember:** Ethics in AI isn't about having all the answers—it's about asking the right questions and being committed to addressing the challenges we discover.