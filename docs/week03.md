---
layout: week
title: "Week 3: Problem-Solving by Search II (Informed Search)"
subtitle: "Heuristic Search Strategies"
description: "Explore informed search algorithms like A* that use domain knowledge to search more efficiently. Learn about heuristics, admissibility, and optimization techniques."
week_number: 3
total_weeks: 14
github_folder: "Week03_Informed_Search"
notes_link: "https://github.com/irfan-sec/ai-learning-students/blob/main/weeks/Week03_Informed_Search/notes.md"
resources_link: "https://github.com/irfan-sec/ai-learning-students/blob/main/weeks/Week03_Informed_Search/resources.md"
exercises_link: "https://github.com/irfan-sec/ai-learning-students/blob/main/weeks/Week03_Informed_Search/exercises.md"
code_link: "https://github.com/irfan-sec/ai-learning-students/tree/main/weeks/Week03_Informed_Search/code"
prev_week:
  title: "Uninformed Search"
  url: "/week02.html"
next_week:
  title: "Adversarial Search"
  url: "/week04.html"
objectives:
  - "Understand the concept of heuristic functions"
  - "Learn and implement the A* search algorithm"
  - "Explore greedy best-first search and its properties"
  - "Understand admissibility and consistency in heuristics"
  - "Design effective heuristics for specific problems"
  - "Compare informed vs. uninformed search performance"
key_concepts:
  - "Heuristic Functions"
  - "A* Search Algorithm"
  - "Greedy Best-First Search"
  - "Admissible Heuristics"
  - "Consistent Heuristics"
  - "Manhattan Distance"
  - "Euclidean Distance"
  - "Hill Climbing"
---

## 🧭 Smart Search with Heuristics

This week, we'll supercharge our search algorithms with domain knowledge! Instead of blindly exploring the search space, we'll use **heuristics** - informed guesses about how close we are to the goal. This transforms search from a systematic but inefficient process into an intelligent, goal-directed exploration.

## 📚 What You'll Learn

### Core Topics

1. **Heuristic Functions**
   - What makes a good heuristic?
   - Admissibility and consistency
   - Designing domain-specific heuristics

2. **Informed Search Algorithms**
   - **Greedy Best-First Search:** Fast but not optimal
   - **A* Search:** Optimal and complete with admissible heuristics
   - **Weighted A*:** Trading optimality for speed

3. **Advanced Topics**
   - Memory-bounded search (IDA*, SMA*)
   - Bidirectional search
   - Heuristic generation and learning

## 🎯 The A* Algorithm

A* combines the best of both worlds:
- **g(n):** Cost from start to current node (like UCS)
- **h(n):** Heuristic estimate from current node to goal
- **f(n) = g(n) + h(n):** Total estimated cost of solution through n

```python
def a_star_search(problem, heuristic):
    frontier = PriorityQueue()
    frontier.put((0, start_node))
    explored = set()
    
    while not frontier.empty():
        current = frontier.get()
        
        if problem.goal_test(current.state):
            return current.solution()
            
        explored.add(current.state)
        
        for action in problem.actions(current.state):
            child = child_node(problem, current, action)
            if child.state not in explored:
                f_cost = child.path_cost + heuristic(child.state)
                frontier.put((f_cost, child))
```

## 🗺️ Classic Heuristics

### For Grid-Based Pathfinding

**Manhattan Distance (L1)**
```
h(n) = |x₁ - x₂| + |y₁ - y₂|
```
Perfect for grid worlds where you can only move in four directions.

**Euclidean Distance (L2)**
```
h(n) = √[(x₁ - x₂)² + (y₁ - y₂)²]
```
Ideal when diagonal movement is allowed.

### For the 8-Puzzle

**Misplaced Tiles**
```
h(n) = number of tiles not in their goal position
```

**Manhattan Distance**
```
h(n) = sum of distances of tiles from their goal positions
```

## 🏃‍♀️ Algorithm Performance

| Algorithm | Optimality | Time Complexity | Space Complexity | Notes |
|-----------|------------|-----------------|------------------|-------|
| Greedy Best-First | No | O(b^m) | O(b^m) | Fast but risky |
| A* | Yes* | O(b^d) | O(b^d) | *with admissible h(n) |
| IDA* | Yes* | O(b^d) | O(bd) | Memory efficient |

## 🎮 Interactive Examples

### The 8-Puzzle Revisited
Watch how A* with Manhattan distance heuristic solves the puzzle much faster than BFS:
- **BFS:** Explores ~100,000 states
- **A* with good heuristic:** Explores ~100 states

### Pathfinding in Games
See how game characters navigate complex environments using A* with custom heuristics that account for terrain, enemies, and objectives.

## 🧠 Heuristic Design Principles

### Properties of Good Heuristics

1. **Admissible:** Never overestimate the true cost
2. **Consistent:** h(n) ≤ c(n,a,n') + h(n') for all n,n'
3. **Informative:** Closer estimates lead to better performance
4. **Efficient:** Quick to compute

### Common Heuristic Patterns

- **Relaxed Problems:** Remove constraints from the original problem
- **Pattern Databases:** Precompute costs for subproblems
- **Landmark Heuristics:** Identify must-visit intermediate goals

## 🚀 Real-World Applications

### GPS and Navigation
Your smartphone uses variants of A* to find optimal routes, considering:
- Distance (g-cost)
- Estimated time to destination (h-cost)
- Traffic conditions
- Road quality

### Video Games
From Pac-Man to modern RPGs, A* powers intelligent NPC movement.

### Robotics
Path planning for autonomous robots in warehouses, hospitals, and homes.

### Logistics
Optimizing delivery routes and resource allocation.

## 💡 Advanced Concepts

### When A* Struggles
- **Memory limitations:** Large search spaces
- **Dynamic environments:** Changing costs and obstacles
- **Multiple objectives:** More than one goal to optimize

### Solutions and Variations
- **Anytime algorithms:** Improve solution quality over time
- **Hierarchical search:** Solve at multiple abstraction levels
- **Learning heuristics:** Improve estimates through experience

## 🔗 Connections

- **Previous:** Building on uninformed search foundations
- **Next:** Adversarial search uses similar evaluation functions
- **Future:** Heuristic principles appear in optimization and machine learning

---

*Ready to make your search intelligent? Explore the materials and implement A* to see the power of informed search!*