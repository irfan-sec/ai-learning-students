---
layout: default
title: "Week 13: Ethics in AI"
permalink: /week13.html
---

### Navigation
[🏠 Home](index.html) | [← Previous: Week 12](week12.html) | [Next: Week 14 →](week14.html)

---

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

![AI Ethics Overview](images/ai-ethics-overview.png)

### The Stakes Are High
- **Individual Impact:** AI decisions affect careers, health, freedom, and opportunities
- **Societal Impact:** AI can perpetuate or amplify existing inequalities  
- **Global Impact:** AI systems shape information, democracy, and economic structures
- **Future Impact:** Today's AI decisions set precedents for tomorrow's more powerful systems

### Why This Matters to You
As future AI practitioners, you will be responsible for building systems that affect millions of people. Understanding these ethical challenges isn't just academic—it's essential for professional practice.

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

```python
# Example: Biased hiring data
historical_hires = {
    'software_engineer': {'male': 90, 'female': 10},  # Historical bias
    'nurse': {'male': 15, 'female': 85},
    'teacher': {'male': 25, 'female': 75}
}

# Problem: AI trained on this data will perpetuate these patterns
def analyze_hiring_bias(job_data):
    """Detect potential gender bias in hiring data"""
    for job, demographics in job_data.items():
        total = sum(demographics.values())
        male_ratio = demographics['male'] / total
        female_ratio = demographics['female'] / total
        
        # Flag potential bias (threshold: 70-30 split)
        if male_ratio > 0.7 or female_ratio > 0.7:
            print(f"⚠️ Potential bias in {job}: {male_ratio:.0%} male, {female_ratio:.0%} female")

analyze_hiring_bias(historical_hires)
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

### Defining Fairness Mathematically

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

```python
import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_fairness_metrics(y_true, y_pred, sensitive_attr):
    """Calculate various fairness metrics"""
    # Split by sensitive attribute
    group_0 = sensitive_attr == 0
    group_1 = sensitive_attr == 1
    
    # Statistical Parity
    positive_rate_0 = np.mean(y_pred[group_0])
    positive_rate_1 = np.mean(y_pred[group_1])
    statistical_parity = abs(positive_rate_0 - positive_rate_1)
    
    # Equal Opportunity (True Positive Rate equality)
    tp_0 = np.sum((y_true[group_0] == 1) & (y_pred[group_0] == 1))
    fn_0 = np.sum((y_true[group_0] == 1) & (y_pred[group_0] == 0))
    tpr_0 = tp_0 / (tp_0 + fn_0) if (tp_0 + fn_0) > 0 else 0
    
    tp_1 = np.sum((y_true[group_1] == 1) & (y_pred[group_1] == 1))
    fn_1 = np.sum((y_true[group_1] == 1) & (y_pred[group_1] == 0))
    tpr_1 = tp_1 / (tp_1 + fn_1) if (tp_1 + fn_1) > 0 else 0
    
    equal_opportunity = abs(tpr_0 - tpr_1)
    
    return {
        'statistical_parity_difference': statistical_parity,
        'equal_opportunity_difference': equal_opportunity,
        'tpr_group_0': tpr_0,
        'tpr_group_1': tpr_1
    }

# Example usage
np.random.seed(42)
y_true = np.random.binomial(1, 0.3, 1000)
y_pred = np.random.binomial(1, 0.4, 1000)  # Biased predictions
sensitive_attr = np.random.binomial(1, 0.5, 1000)

metrics = calculate_fairness_metrics(y_true, y_pred, sensitive_attr)
print("Fairness Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.3f}")
```

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
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def preprocess_for_fairness(X, y, sensitive_feature):
    """Pre-processing to reduce bias"""
    
    # Remove sensitive features from training
    X_processed = X.drop(columns=[sensitive_feature])
    
    # Balance dataset across sensitive groups
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_processed, y)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_balanced)
    
    return X_scaled, y_balanced, scaler
```

**In-processing Approaches:**
```python
def fairness_regularized_loss(y_true, y_pred, sensitive_attr, lambda_fair=0.1):
    """Loss function with fairness regularization"""
    
    # Standard prediction loss (e.g., cross-entropy)
    prediction_loss = binary_crossentropy(y_true, y_pred)
    
    # Fairness penalty - discourage disparate impact
    group_0_pred = y_pred[sensitive_attr == 0]
    group_1_pred = y_pred[sensitive_attr == 1]
    
    fairness_penalty = tf.abs(tf.reduce_mean(group_0_pred) - tf.reduce_mean(group_1_pred))
    
    # Combined loss
    total_loss = prediction_loss + lambda_fair * fairness_penalty
    
    return total_loss
```

**Post-processing Approaches:**
```python
def calibrate_for_fairness(predictions, sensitive_attr, method='equal_opportunity'):
    """Adjust predictions to achieve fairness"""
    
    if method == 'equal_opportunity':
        # Adjust thresholds to equalize true positive rates
        group_0_threshold = find_threshold_for_tpr(predictions[sensitive_attr == 0], target_tpr=0.8)
        group_1_threshold = find_threshold_for_tpr(predictions[sensitive_attr == 1], target_tpr=0.8)
        
        # Apply different thresholds
        calibrated_predictions = predictions.copy()
        calibrated_predictions[sensitive_attr == 0] = (predictions[sensitive_attr == 0] > group_0_threshold).astype(int)
        calibrated_predictions[sensitive_attr == 1] = (predictions[sensitive_attr == 1] > group_1_threshold).astype(int)
        
    return calibrated_predictions
```

### 3. **Explainable AI (XAI)**

**Techniques:**
- **Local explanations:** Why this particular decision?
- **Global explanations:** How does the system work overall?
- **Feature importance:** Which factors mattered most?
- **Counterfactual explanations:** What would change the outcome?

```python
import shap
from lime import lime_text

def explain_model_decision(model, X_train, X_test, instance_idx):
    """Provide multiple explanations for a model decision"""
    
    # SHAP explanation (global and local)
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    
    # Feature importance for specific instance
    instance_explanation = shap_values[instance_idx]
    
    # LIME explanation for comparison
    lime_explainer = lime_text.LimeTextExplainer()
    lime_explanation = lime_explainer.explain_instance(
        X_test[instance_idx], 
        model.predict_proba,
        num_features=10
    )
    
    return {
        'shap_values': instance_explanation,
        'lime_explanation': lime_explanation,
        'feature_importance': dict(zip(X_train.columns, shap_values.values[instance_idx]))
    }
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

```python
class HumanInTheLoopClassifier:
    """AI system with human oversight"""
    
    def __init__(self, ai_model, confidence_threshold=0.8):
        self.ai_model = ai_model
        self.confidence_threshold = confidence_threshold
        self.human_review_queue = []
    
    def predict(self, X):
        """Make predictions with human fallback"""
        predictions = self.ai_model.predict_proba(X)
        confident_predictions = []
        
        for i, pred_proba in enumerate(predictions):
            max_confidence = max(pred_proba)
            
            if max_confidence >= self.confidence_threshold:
                # AI is confident - use AI prediction
                confident_predictions.append(pred_proba.argmax())
            else:
                # Low confidence - queue for human review
                self.human_review_queue.append({
                    'instance_id': i,
                    'features': X[i],
                    'ai_prediction': pred_proba.argmax(),
                    'confidence': max_confidence
                })
                confident_predictions.append(None)  # Pending human review
        
        return confident_predictions
    
    def get_human_review_queue(self):
        """Get cases that need human review"""
        return self.human_review_queue
```

---

## 💻 Hands-On Exercise: Bias Detection Tool

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

class BiasDetectionTool:
    """Comprehensive bias detection and analysis tool"""
    
    def __init__(self, model, sensitive_attributes):
        self.model = model
        self.sensitive_attributes = sensitive_attributes
        self.bias_report = {}
    
    def analyze_dataset_bias(self, X, y):
        """Analyze bias in the dataset"""
        dataset_bias = {}
        
        for attr in self.sensitive_attributes:
            if attr in X.columns:
                # Distribution analysis
                attr_distribution = X[attr].value_counts(normalize=True)
                outcome_by_attr = X.groupby(attr)[y.name].mean()
                
                dataset_bias[attr] = {
                    'distribution': attr_distribution.to_dict(),
                    'outcome_rates': outcome_by_attr.to_dict()
                }
        
        return dataset_bias
    
    def analyze_model_bias(self, X_test, y_test):
        """Analyze bias in model predictions"""
        predictions = self.model.predict(X_test)
        pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        model_bias = {}
        
        for attr in self.sensitive_attributes:
            if attr in X_test.columns:
                # Group-specific performance
                groups = X_test[attr].unique()
                group_metrics = {}
                
                for group in groups:
                    group_mask = X_test[attr] == group
                    group_y_true = y_test[group_mask]
                    group_y_pred = predictions[group_mask]
                    
                    if len(group_y_true) > 0:
                        # Calculate metrics for this group
                        from sklearn.metrics import accuracy_score, precision_score, recall_score
                        
                        group_metrics[group] = {
                            'accuracy': accuracy_score(group_y_true, group_y_pred),
                            'precision': precision_score(group_y_true, group_y_pred, average='weighted', zero_division=0),
                            'recall': recall_score(group_y_true, group_y_pred, average='weighted', zero_division=0),
                            'positive_prediction_rate': np.mean(group_y_pred),
                            'size': len(group_y_true)
                        }
                
                model_bias[attr] = group_metrics
        
        return model_bias
    
    def visualize_bias(self, X_test, y_test, predictions):
        """Create visualizations of bias"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for i, attr in enumerate(self.sensitive_attributes[:4]):  # Limit to 4 attributes
            if attr in X_test.columns and i < 4:
                ax = axes[i//2, i%2]
                
                # Prediction rates by group
                results_df = pd.DataFrame({
                    'Sensitive_Attribute': X_test[attr],
                    'True_Label': y_test,
                    'Predicted_Label': predictions
                })
                
                prediction_rates = results_df.groupby('Sensitive_Attribute')['Predicted_Label'].mean()
                
                prediction_rates.plot(kind='bar', ax=ax)
                ax.set_title(f'Positive Prediction Rate by {attr}')
                ax.set_ylabel('Positive Prediction Rate')
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def generate_bias_report(self, X_train, y_train, X_test, y_test):
        """Generate comprehensive bias report"""
        print("=" * 50)
        print("BIAS DETECTION REPORT")
        print("=" * 50)
        
        # Dataset bias analysis
        print("\n1. DATASET BIAS ANALYSIS")
        dataset_bias = self.analyze_dataset_bias(X_train, y_train)
        
        for attr, analysis in dataset_bias.items():
            print(f"\nAttribute: {attr}")
            print("Distribution:")
            for value, prop in analysis['distribution'].items():
                print(f"  {value}: {prop:.2%}")
            
            print("Outcome rates by group:")
            for value, rate in analysis['outcome_rates'].items():
                print(f"  {value}: {rate:.2%}")
        
        # Model bias analysis
        print("\n2. MODEL BIAS ANALYSIS")
        predictions = self.model.predict(X_test)
        model_bias = self.analyze_model_bias(X_test, y_test)
        
        for attr, groups in model_bias.items():
            print(f"\nAttribute: {attr}")
            print("Performance by group:")
            
            for group, metrics in groups.items():
                print(f"  Group {group} (n={metrics['size']}):")
                print(f"    Accuracy: {metrics['accuracy']:.3f}")
                print(f"    Precision: {metrics['precision']:.3f}")
                print(f"    Recall: {metrics['recall']:.3f}")
                print(f"    Positive Prediction Rate: {metrics['positive_prediction_rate']:.3f}")
        
        # Visualizations
        print("\n3. BIAS VISUALIZATIONS")
        self.visualize_bias(X_test, y_test, predictions)
        
        # Recommendations
        print("\n4. RECOMMENDATIONS")
        self._generate_recommendations(dataset_bias, model_bias)
    
    def _generate_recommendations(self, dataset_bias, model_bias):
        """Generate actionable recommendations"""
        recommendations = []
        
        # Check for representation bias
        for attr, analysis in dataset_bias.items():
            distribution = analysis['distribution']
            min_representation = min(distribution.values())
            if min_representation < 0.1:  # Less than 10% representation
                recommendations.append(f"⚠️ Low representation for some groups in {attr} (min: {min_representation:.1%})")
                recommendations.append(f"   → Consider oversampling or collecting more data")
        
        # Check for performance disparities
        for attr, groups in model_bias.items():
            accuracies = [m['accuracy'] for m in groups.values()]
            if len(accuracies) > 1:
                accuracy_gap = max(accuracies) - min(accuracies)
                if accuracy_gap > 0.05:  # More than 5% accuracy gap
                    recommendations.append(f"⚠️ Performance gap in {attr}: {accuracy_gap:.1%}")
                    recommendations.append(f"   → Consider fairness-aware training or post-processing")
        
        if not recommendations:
            recommendations.append("✅ No major bias issues detected")
            recommendations.append("   → Continue monitoring and regular bias audits")
        
        for rec in recommendations:
            print(rec)

# Example usage
def demonstrate_bias_detection():
    """Demonstrate the bias detection tool"""
    
    # Create synthetic biased dataset
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    age = np.random.normal(35, 10, n_samples)
    gender = np.random.choice(['M', 'F'], n_samples, p=[0.7, 0.3])  # Gender imbalance
    education = np.random.choice(['High School', 'College', 'Graduate'], n_samples)
    
    # Create biased target (loan approval)
    # Bias: Lower approval rates for women
    base_approval_prob = 0.6
    gender_bias = -0.2 if 'F' else 0  # This creates bias
    
    approval_probs = []
    for i in range(n_samples):
        prob = base_approval_prob
        if gender[i] == 'F':
            prob += -0.2  # Bias against women
        if education[i] == 'Graduate':
            prob += 0.3
        elif education[i] == 'College':
            prob += 0.1
            
        approval_probs.append(max(0.1, min(0.9, prob)))  # Clamp between 0.1 and 0.9
    
    approved = np.random.binomial(1, approval_probs)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'gender': gender,
        'education': education,
        'approved': approved
    })
    
    # Encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['gender', 'education'])
    
    # Split data
    X = df_encoded.drop('approved', axis=1)
    y = df_encoded['approved']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Create bias detection tool
    # Note: Using original categorical columns for analysis
    X_train_orig = df.iloc[X_train.index].drop('approved', axis=1)
    X_test_orig = df.iloc[X_test.index].drop('approved', axis=1)
    
    bias_detector = BiasDetectionTool(model, ['gender', 'education'])
    bias_detector.generate_bias_report(X_train_orig, y_train, X_test_orig, y_test)

# Uncomment to run the demonstration
# demonstrate_bias_detection()
```

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

## 💼 Practical Frameworks for Ethical AI

### Ethics by Design Checklist

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

### Stakeholder Impact Analysis

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

## 🔗 Curated Resources

### Essential Reading
- [Partnership on AI Principles](https://partnershiponai.org/) - Industry best practices
- [AI Ethics Guidelines Global Inventory](https://ai-ethics-guidelines-inventory.com/) - Compare different frameworks
- [Algorithmic Accountability Act](https://www.congress.gov/bill/117th-congress/house-bill/2231) - Proposed US legislation

### Academic Papers
- ["Fairness and Abstraction in Sociotechnical Systems"](https://dl.acm.org/doi/10.1145/3287560.3287598) - ACM FAccT
- ["Man is to Computer Programmer as Woman is to Homemaker?"](https://arxiv.org/abs/1607.06520) - Word embedding bias
- ["Equalizing Odds: Fairness Criteria for Machine Learning"](https://arxiv.org/abs/1610.02413) - Fairness definitions

### Video Resources
- [AI Ethics Course - MIT](https://www.youtube.com/playlist?list=PLUl4u3cNGP62uI5y_2wOJvKz3dn9kZYZf)
- [Algorithmic Justice League](https://www.ajl.org/) - Joy Buolamwini's work on bias
- [Weapons of Math Destruction](https://www.youtube.com/watch?v=TQHs8SA1qpk) - Cathy O'Neil's talk

### Tools and Frameworks
- [IBM AI Fairness 360](https://aif360.mybluemix.net/) - Open source bias detection toolkit
- [Google What-If Tool](https://pair-code.github.io/what-if-tool/) - Visual bias analysis
- [Microsoft Fairlearn](https://fairlearn.org/) - Python library for fairness assessment

---

## 🎯 Key Principles for Students

### As Future AI Practitioners

1. **Stay Informed:** Keep up with ethical developments in AI
2. **Think Critically:** Question assumptions and potential biases
3. **Collaborate:** Work with diverse teams and stakeholders
4. **Be Transparent:** Document decisions and trade-offs
5. **Take Responsibility:** Consider the broader impact of your work

### Questions to Always Ask

- **Who benefits** from this AI system?
- **Who might be harmed** by this system?
- **What biases** might exist in the data or algorithm?
- **How transparent** is the decision-making process?
- **What safeguards** are in place?
- **How will we know** if the system is working as intended?

---

## 🤔 Discussion Questions

1. Can AI systems ever be truly "objective" or are they always reflections of human biases?

2. Should there be a "right to explanation" for all AI decisions that affect individuals?

3. How do we balance the benefits of AI surveillance (crime prevention, public health) with privacy rights?

4. What responsibility do tech companies have for the societal impacts of their AI systems?

5. How can we ensure that AI benefits everyone, not just those with access to technology and data?

6. What role should the public play in governing AI development and deployment?

---

## 📚 Assignment: Ethical AI Analysis

### Individual Reflection (Required)
Write a 2-page reflection addressing:

1. **Issue Identification:** Choose a specific AI ethical issue that concerns you most and explain why
2. **Personal Impact:** How might this issue affect you or your community?
3. **Professional Response:** How would you address this issue as an AI practitioner?
4. **Policy Recommendations:** What policies or practices would you recommend?
5. **Future Considerations:** How might this issue evolve as AI technology advances?

### Group Project (Optional)
Choose a current AI system and conduct an ethical analysis:

1. **System Overview:** Describe the AI system and its purpose
2. **Stakeholder Analysis:** Identify all affected parties
3. **Bias Assessment:** Identify potential biases and their sources
4. **Fairness Evaluation:** Assess fairness across different groups
5. **Recommendations:** Propose improvements and safeguards
6. **Presentation:** Present findings to the class (10 minutes + Q&A)

**Suggested Systems to Analyze:**
- Hiring algorithms (LinkedIn, ZipRecruiter)
- Credit scoring systems (FICO, alternative credit)
- Content moderation (Facebook, YouTube)
- Predictive policing (PredPol, CompStat)
- Healthcare AI (diagnostic imaging, drug discovery)

---

## 🔮 Looking Ahead

Next week, we'll conclude our AI journey with **course review, project presentations, and exploring the future of AI**. We'll synthesize everything we've learned and discuss emerging trends and challenges.

**Preview of Week 14:**
- Course review and concept integration
- Student project presentations
- Emerging AI technologies and trends
- Career paths in AI
- The future of artificial intelligence

---

### Navigation
[🏠 Home](index.html) | [← Previous: Week 12](week12.html) | [Next: Week 14 →](week14.html)

---

*"With great power comes great responsibility. As AI becomes more powerful, our responsibility to develop and deploy it ethically becomes even greater."*

**Remember:** Ethics in AI isn't about having all the answers—it's about asking the right questions and being committed to addressing the challenges we discover.