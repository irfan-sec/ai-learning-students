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

![AI Timeline](https://qbi.uq.edu.au/files/40697/The-Brain-Intelligent-Machines-AI-timeline.jpg)

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

![Agent Environment Diagram](https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/IntelligentAgent-SimpleReflex.png/400px-IntelligentAgent-SimpleReflex.png)

### The PEAS Framework

To design any AI system, we must specify:

| Component | Description | Example: Autonomous Car |
|-----------|-------------|-------------------------|
| **P**erformance | What constitutes success? | Safe, fast, legal driving; passenger comfort |
| **E**nvironment | What is the agent's world? | Roads, traffic, weather, pedestrians |
| **A**ctuators | How can the agent act? | Steering, brakes, accelerator, horn, signals |
| **S**ensors | What can the agent perceive? | Camera, GPS, speedometer, engine sensors |

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

## 🤔 Discussion Questions

1. Is a thermostat an intelligent agent? Why or why not?
2. How would you define the PEAS framework for a web search engine?
3. What are the challenges in creating truly general AI?
4. Should AI systems be transparent in their decision-making? When and why?

---

## 🔍 Looking Ahead

Next week, we'll dive into **problem-solving as search**, where we'll learn how agents can find solutions to complex problems by exploring possible actions and states. This forms the foundation for much of classical AI!

---

*"AI is not about replacing human intelligence, but about amplifying it."* - Fei-Fei Li
