---
layout: week
title: "Week 2: Problem-Solving by Search I (Uninformed Search)"
subtitle: "Fundamental Search Algorithms"
description: "Learn about problem formulation and uninformed search strategies including breadth-first search, depth-first search, and uniform cost search."
week_number: 2
total_weeks: 14
github_folder: "Week02_Uninformed_Search"
notes_link: "https://github.com/irfan-sec/ai-learning-students/blob/main/weeks/Week02_Uninformed_Search/notes.md"
resources_link: "https://github.com/irfan-sec/ai-learning-students/blob/main/weeks/Week02_Uninformed_Search/resources.md"
exercises_link: "https://github.com/irfan-sec/ai-learning-students/blob/main/weeks/Week02_Uninformed_Search/exercises.md"
code_link: "https://github.com/irfan-sec/ai-learning-students/tree/main/weeks/Week02_Uninformed_Search/code"
prev_week:
  title: "Introduction to AI"
  url: "/week01.html"
next_week:
  title: "Informed Search"
  url: "/week03.html"
objectives:
  - "Formulate problems as state-space search"
  - "Understand and implement breadth-first search (BFS)"
  - "Understand and implement depth-first search (DFS)"
  - "Learn uniform cost search and its applications"
  - "Analyze the completeness and optimality of search algorithms"
  - "Apply search algorithms to solve practical problems"
key_concepts:
  - "State Space"
  - "Search Trees vs. Graphs"
  - "Breadth-First Search (BFS)"
  - "Depth-First Search (DFS)"
  - "Uniform Cost Search (UCS)"
  - "Completeness and Optimality"
  - "Time and Space Complexity"
---

## 🗺️ Navigation Through Problem Spaces

Welcome to the world of search algorithms! This week, we'll learn how intelligent agents can systematically explore problem spaces to find solutions. Search is one of the most fundamental techniques in AI, forming the backbone of everything from GPS navigation to game playing.

## 📚 What You'll Learn

### Core Topics

1. **Problem Formulation**
   - State space representation
   - Initial state, goal test, and actions
   - Path cost and solution quality

2. **Search Strategies Overview**
   - Tree search vs. graph search
   - Frontier management
   - Explored set maintenance

3. **Uninformed Search Algorithms**
   - **Breadth-First Search (BFS):** Complete and optimal for unit costs
   - **Depth-First Search (DFS):** Memory efficient but not optimal
   - **Uniform Cost Search (UCS):** Optimal for any step cost
   - **Depth-Limited Search:** DFS with depth bounds
   - **Iterative Deepening:** Combines benefits of BFS and DFS

## 🎯 Learning Activities

### Required Reading
- **AIMA Chapter 3:** Solving Problems by Searching (sections 3.1-3.4)

### Interactive Learning
Use the online visualizations linked in the resources to:
- Watch BFS explore level by level
- See how DFS dives deep into the search space
- Compare algorithm performance on different problems

### Coding Practice
- Implement BFS and DFS from scratch
- Solve classic problems like the 8-puzzle
- Create visualizations of search tree exploration

## 🧩 Classic Problems

### The 8-Puzzle
```
Initial State:    Goal State:
1 2 3            1 2 3
4   6    →       4 5 6
7 5 8            7 8
```

### Pathfinding
Finding optimal routes through mazes, grids, and graphs - the foundation of GPS systems and robotics.

### The Water Jug Problem
Given two jugs of different capacities, how can you measure a specific amount of water?

## 🔍 Algorithm Comparison

| Algorithm | Complete? | Optimal? | Time Complexity | Space Complexity |
|-----------|-----------|----------|-----------------|------------------|
| BFS       | Yes*      | Yes*     | O(b^d)         | O(b^d)          |
| DFS       | No        | No       | O(b^m)         | O(bm)           |
| UCS       | Yes*      | Yes      | O(b^⌈C*/ε⌉)   | O(b^⌈C*/ε⌉)    |

*when the branching factor is finite

## 🚀 Practical Applications

### GPS Navigation
How does your phone find the shortest route? UCS with road distances as costs!

### Robotics
Mobile robots use search to plan paths while avoiding obstacles.

### Game AI
Even complex games start with basic search principles for move generation.

### Puzzle Solving
From Sudoku to Rubik's cubes, search algorithms can solve structured problems.

## 💡 Key Insights

- **Trade-offs:** Every algorithm has strengths and weaknesses
- **Problem-dependent:** The best algorithm depends on your specific problem
- **Foundation:** These "simple" algorithms form the basis for advanced AI techniques

## 🧠 Critical Thinking Questions

- When would you choose DFS over BFS?
- How do real-world constraints affect algorithm choice?
- What happens when the search space is infinite?
- How can we make search more efficient?

## 🔗 Connections

- **Previous Week:** Agents need search to find action sequences
- **Next Week:** Informed search uses domain knowledge to be more efficient
- **Later Weeks:** Search principles appear in machine learning and planning

---

*Ready to start searching? Dive into the detailed materials and start implementing your first AI algorithms!*