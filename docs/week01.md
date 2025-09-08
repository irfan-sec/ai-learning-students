---
layout: week
title: "Week 1: Introduction to AI and Intelligent Agents"
permalink: /week01.html
---

### Navigation
[🏠 Home](index.html) | Previous | [Next: Week 2 →](week02.html)

---

# Week 1: Introduction to AI and Intelligent Agents

## 📚 Learning Objectives
By the end of this week, students will be able to:
- Define artificial intelligence and understand its historical development
- Distinguish between narrow and general AI
- Understand the concept of intelligent agents and the PEAS framework
- Identify different types of agent environments
- Recognize key AI applications in modern technology

---

## 🧠 What is Artificial Intelligence?

![AI Timeline](images/ai-timeline.png)

**Artificial Intelligence (AI)** is the science of making machines act intelligently. But what does "acting intelligently" mean?

### Historical Perspectives on AI

#### 1. **Thinking Humanly** (Cognitive Science Approach)
- Model human thought processes
- Understand how the human mind works
- Example: Cognitive architectures that simulate human reasoning

#### 2. **Acting Humanly** (Turing Test Approach)  
- Pass the Turing Test - convince a human interrogator that the machine is human
- Focus on external behavior rather than internal processes
- **Alan Turing (1950):** "Can machines think?"

#### 3. **Thinking Rationally** (Laws of Thought Approach)
- Use logical reasoning and formal logic
- Aristotelian syllogisms and mathematical proof
- Problem: Not all intelligent behavior is logical

#### 4. **Acting Rationally** (Rational Agent Approach) ⭐
- **Current dominant paradigm**
- Do the "right thing" given available information
- Maximize expected performance based on goals

---

## 🤖 Intelligent Agents

### What is an Agent?

An **agent** is anything that can be viewed as:
- **Perceiving** its environment through **sensors**
- **Acting** upon the environment through **actuators**

![Agent Environment Diagram](images/intelligent-agent-diagram.png)

### The PEAS Framework

To design any AI system, we must specify:

| Component | Description | Example: Autonomous Car |
|-----------|-------------|-------------------------|
| **P**erformance | What constitutes success? | Safe, fast, legal driving; passenger comfort |
| **E**nvironment | What is the agent's world? | Roads, traffic, weather, pedestrians |
| **A**ctuators | How can the agent act? | Steering, brakes, accelerator, horn, signals |
| **S**ensors | What can the agent perceive? | Camera, GPS, speedometer, engine sensors |

### Programming Example: Simple Reflex Agent

```python
class SimpleReflexAgent:
    def __init__(self):
        self.rules = {
            'dirty': 'suck',
            'clean': 'move'
        }
    
    def perceive(self, environment):
        """Get current state from sensors"""
        return environment.get_current_state()
    
    def act(self, percept):
        """Choose action based on current percept"""
        if percept in self.rules:
            return self.rules[percept]
        return 'no_action'

# Example usage
agent = SimpleReflexAgent()
current_state = 'dirty'
action = agent.act(current_state)
print(f"State: {current_state}, Action: {action}")  # Output: State: dirty, Action: suck
```

### Types of Agent Environments

#### 1. **Observable vs. Partially Observable**
- **Fully Observable:** Chess (can see entire board)
- **Partially Observable:** Poker (can't see opponents' cards)

#### 2. **Deterministic vs. Stochastic**
- **Deterministic:** Chess (outcomes are certain given actions)
- **Stochastic:** Backgammon (dice introduces randomness)

#### 3. **Episodic vs. Sequential**
- **Episodic:** Email classification (each email independent)
- **Sequential:** Chess (current move affects future positions)

#### 4. **Static vs. Dynamic**
- **Static:** Crossword puzzle (doesn't change while thinking)
- **Dynamic:** Taxi driving (world changes continuously)

#### 5. **Discrete vs. Continuous**
- **Discrete:** Chess (finite number of moves)
- **Continuous:** Taxi driving (infinite steering angles)

#### 6. **Single Agent vs. Multi-agent**
- **Single:** Puzzle solving
- **Multi-agent:** Chess, autonomous driving

---

## 🔧 Types of AI

### Narrow AI (Weak AI) vs. General AI (Strong AI)

| Aspect | Narrow AI | General AI |
|--------|-----------|------------|
| **Scope** | Specific tasks | Any intellectual task |
| **Examples** | Chess programs, image recognition, Siri | Human-level intelligence (theoretical) |
| **Current Status** | Widely deployed | Not yet achieved |
| **Intelligence** | Specialized | General-purpose |

### Current AI Applications

![AI Applications](https://upload.wikimedia.org/wikipedia/commons/thumb/d/de/ArtificialFictionBrain.png/300px-ArtificialFictionBrain.png)

1. **Computer Vision**
   - Facial recognition
   - Medical image analysis
   - Autonomous vehicles

2. **Natural Language Processing**
   - Machine translation
   - Chatbots and virtual assistants
   - Sentiment analysis

3. **Robotics**
   - Manufacturing automation
   - Service robots
   - Exploration robots

4. **Game Playing**
   - Chess (Deep Blue)
   - Go (AlphaGo)
   - Video games (OpenAI Five)

5. **Machine Learning Applications**
   - Recommendation systems
   - Fraud detection
   - Predictive maintenance

---

## 🎯 Key Concepts Summary

### Important Terms
- **Agent:** Entity that perceives and acts
- **Rationality:** Acting to maximize expected performance
- **Autonomy:** Ability to operate without human intervention
- **PEAS:** Framework for describing agent design problems

### The Rational Agent Approach
- **Goal:** Create agents that act rationally
- **Rationality ≠ Perfection:** Work with available information
- **Performance:** Measured by external criteria, not internal satisfaction

---

## 💻 Hands-On Exercise: Design Your Own Agent

### Exercise 1: PEAS Analysis
For each scenario below, define the PEAS framework:

**Scenario A: Smart Home Thermostat**
- **Performance:** _______________
- **Environment:** _______________
- **Actuators:** _______________
- **Sensors:** _______________

**Scenario B: Chess Playing Program**
- **Performance:** _______________
- **Environment:** _______________
- **Actuators:** _______________
- **Sensors:** _______________

### Exercise 2: Environment Classification
Classify these environments along all six dimensions:
1. Web search engine
2. Medical diagnosis system
3. Autonomous drone delivery

---

## 🔗 Curated Resources

### Essential Reading
- [AIMA Chapter 1 & 2](https://aima.cs.berkeley.edu/) - Introduction to AI and Intelligent Agents
- [AI: A Guide for Thinking Humans](https://www.basicbooks.com/titles/melanie-mitchell/artificial-intelligence/9780374715236/) - Melanie Mitchell

### Video Resources
- [What is Artificial Intelligence?](https://www.youtube.com/watch?v=ad79nYk2keg) - Crash Course (11 min)
- [The Turing Test](https://www.youtube.com/watch?v=3wLqsRLvV-c) - Computerphile (8 min)
- [AI vs Machine Learning vs Deep Learning](https://www.youtube.com/watch?v=k2P_pHQDlp0) - IBM Technology (7 min)

### Interactive Learning
- [AI for People](https://www.elementsofai.com/) - University of Helsinki course
- [CS50's Introduction to AI](https://cs50.harvard.edu/ai/2020/) - Harvard online course

### Historical Context
- [Dartmouth Conference 1956](https://www.aaai.org/ojs/index.php/aimagazine/article/view/1904) - Where AI began
- [Turing's Original Paper](https://www.csee.umbc.edu/courses/471/papers/turing.pdf) - Computing Machinery and Intelligence

---

## 🤔 Discussion Questions

1. Is a thermostat an intelligent agent? Why or why not?
2. How would you define the PEAS framework for a web search engine?
3. What are the challenges in creating truly general AI?
4. Should AI systems be transparent in their decision-making? When and why?
5. How might the definition of "intelligence" change as AI systems become more sophisticated?

---

## 🔍 Looking Ahead

Next week, we'll dive into **problem-solving as search**, where we'll learn how agents can find solutions to complex problems by exploring possible actions and states. This forms the foundation for much of classical AI!

**Preview of Week 2:**
- Formulating problems as search problems
- Breadth-First Search (BFS) and Depth-First Search (DFS)
- Uniform Cost Search and search complexity
- Real-world applications like GPS navigation and puzzle solving

---

### Navigation
[🏠 Home](index.html) | Previous | [Next: Week 2 →](week02.html)

---

*"AI is not about replacing human intelligence, but about amplifying it."* - Fei-Fei Li