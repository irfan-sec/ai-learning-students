# Week 2 Exercises: Problem-Solving by Search I (Uninformed Search)

## 🎯 Learning Goals Review
After completing these exercises, you should be able to:
- Formulate real-world problems as search problems
- Implement and trace through uninformed search algorithms  
- Analyze and compare search algorithms on completeness, optimality, and complexity
- Apply appropriate search strategies to different types of problems

---

## 📝 Problem Formulation Exercises

### Exercise 1: Formulate Search Problems

For each scenario below, clearly define all five components of the search problem:

#### Scenario A: Robot Navigation
A cleaning robot needs to navigate from its charging station to a specific room in a house to clean up a spill.

**Complete this formulation:**
- **Initial State:** ________________________________
- **Actions:** ____________________________________
- **Transition Model:** ___________________________
- **Goal Test:** __________________________________  
- **Path Cost:** __________________________________

#### Scenario B: Course Scheduling
A student needs to create a schedule for 5 courses, with each course offered at multiple time slots, avoiding conflicts.

**Your formulation:**
- **Initial State:** ________________________________
- **Actions:** ____________________________________
- **Transition Model:** ___________________________
- **Goal Test:** __________________________________
- **Path Cost:** __________________________________

#### Scenario C: Rubik's Cube
Solve a scrambled 3×3×3 Rubik's cube.

**Your formulation:**
- **Initial State:** ________________________________
- **Actions:** ____________________________________
- **Transition Model:** ___________________________
- **Goal Test:** __________________________________
- **Path Cost:** __________________________________

**Analysis Questions:**
1. Which of these problems has the largest state space? Estimate the size.
2. Which problem would be most suitable for BFS? Why?
3. For the robot navigation, what additional constraints might you add?

---

## 🌳 Search Tree Analysis

### Exercise 2: Search Tree Construction

Consider this simple graph representing cities and roads:

```
    A --- B --- D
    |     |     |
    C --- E --- F
```

Starting from city A, wanting to reach city F, with each road having cost 1:

**Tasks:**
1. **Draw the search tree** for BFS (show order of node expansion)
2. **Draw the search tree** for DFS (assume alphabetical ordering: B before C)
3. **Show the path** found by each algorithm
4. **Count total nodes expanded** by each algorithm

**BFS Tree:**
```
[Draw your tree here - use text format like:]
Level 0: A
Level 1: B, C
Level 2: ...
```

**DFS Tree:**
```
[Draw your tree here]
```

**Analysis:**
- Which algorithm found the shorter path?
- Which algorithm expanded fewer nodes? 
- Would the results change if F was at a different location?

---

## 🔍 Algorithm Tracing

### Exercise 3: Step-by-Step Algorithm Execution

Given this graph with edge costs:

```
      2
  A ------> B
  |         |
 3|         |1  
  |         |
  v    4    v
  C ------> D
```

Goal: Find path from A to D using Uniform-Cost Search.

**Complete this trace table:**

| Step | Frontier (node:cost) | Current | Goal? | Expanded? | New Children |
|------|---------------------|---------|-------|-----------|-------------|
| 0 | A:0 | - | - | - | - |
| 1 | | A | No | Yes | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |

**Questions:**
1. What path did UCS find?
2. What was the total cost?
3. Why didn't UCS take the direct A→C→D path?

---

## ⚡ Algorithm Comparison

### Exercise 4: Performance Analysis

For the 8-puzzle problem with this initial state:
```
Initial:    Goal:
1 2 3      1 2 3
4 _ 6      4 5 6  
7 5 8      7 8 _
```

**Theoretical Analysis:**
1. **State Space Size:** How many possible states are there? (Hint: 9! positions, but not all reachable)

2. **Branching Factor:** What's the average branching factor for the 8-puzzle?

3. **Algorithm Comparison:** Fill in this table for finding the optimal solution:

| Algorithm | Complete? | Optimal? | Time Complexity | Space Complexity | 
|-----------|-----------|----------|-----------------|------------------|
| BFS | | | | |
| DFS | | | | |
| IDS | | | | |
| UCS | | | | |

**Practical Questions:**
4. Which algorithm would you choose if you had unlimited memory?
5. Which algorithm would you choose if memory was severely limited?
6. Why might IDS be better than both BFS and DFS for this problem?

---

## 💻 Implementation Exercises

### Exercise 5: Code Implementation

**Task:** Implement breadth-first search for a simple grid world.

**Grid Layout (0 = free, 1 = obstacle):**
```python
grid = [
    [0, 0, 1, 0, 0],
    [0, 1, 1, 0, 0], 
    [0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0]
]
start = (0, 0)  # top-left
goal = (4, 4)   # bottom-right
```

**Your Implementation:**
```python
from collections import deque

def bfs_grid_search(grid, start, goal):
    """
    Find shortest path in grid using BFS.
    
    Args:
        grid: 2D list where 0=free, 1=obstacle
        start: (row, col) starting position  
        goal: (row, col) goal position
        
    Returns:
        List of (row, col) positions from start to goal,
        or None if no path exists
    """
    # TODO: Implement BFS here
    # Remember to:
    # 1. Use a queue for the frontier
    # 2. Keep track of visited cells
    # 3. Store parent pointers to reconstruct path
    # 4. Check bounds and obstacles
    
    pass  # Replace with your implementation

# Test your implementation
path = bfs_grid_search(grid, start, goal)
print(f"Path found: {path}")
print(f"Path length: {len(path) if path else 'No path'}")
```

**Extension Questions:**
1. Modify your code to also implement DFS - what differences do you observe?
2. Add a function to visualize the path on the grid
3. What happens if you change the start or goal positions?

---

## 🧩 Puzzle Solving

### Exercise 6: 8-Puzzle by Hand

**Initial State:**
```
2 8 3
1 6 4  
7 _ 5
```

**Goal State:**
```
1 2 3
8 _ 4
7 6 5  
```

**Tasks:**
1. **Solve by hand** - find any solution sequence (list the moves)
2. **Apply BFS approach** - what would be the first few states explored?
3. **Estimate nodes expanded** - how many states would BFS need to explore?

**Your Solution Sequence:**
```
Move 1: ____________________
Move 2: ____________________  
Move 3: ____________________
...
```

**BFS First States (show first 3 levels):**
```
Level 0: [Initial state]
Level 1: [States after 1 move]
Level 2: [States after 2 moves]  
```

---

## 🎮 Game-Based Problems

### Exercise 7: Pathfinding in Games

**Scenario:** In a simple dungeon crawler game, a player character needs to navigate from the entrance to the treasure while avoiding monsters.

**Map (E=entrance, T=treasure, M=monster, .=empty):**
```
E . . M .
. M . . .
. . M . .  
M . . . T
```

**Questions:**
1. **Formulate as search problem:** Define states, actions, goal test
2. **Safety vs. Speed:** How would you modify the path cost to balance avoiding monsters vs. finding short paths?
3. **Dynamic monsters:** If monsters move, how does this change the problem type?
4. **Algorithm choice:** Which search algorithm would be best for real-time pathfinding?

**Your Analysis:**
- **Problem formulation:** ________________________________
- **Modified path cost:** ________________________________  
- **Dynamic environment impact:** ________________________________
- **Algorithm recommendation:** ________________________________

---

## 🔍 Research and Analysis

### Exercise 8: Real-World Applications

**Choose one of these applications and analyze how search algorithms are used:**

1. **GPS Navigation Systems** (Google Maps, Apple Maps)
2. **Social Media Friend Suggestions** (Facebook, LinkedIn)  
3. **Video Game AI** (Pathfinding in strategy games)
4. **Web Crawling** (Search engine indexing)

**Research Questions:**
1. **What is being searched?** (Define the state space)
2. **What search algorithm is most likely used?** Why?
3. **What constraints or optimizations are needed?** (real-time, memory, etc.)
4. **How is this different from textbook examples?**

**Your Analysis:**
**Application chosen:** ________________________________

**State space:** ________________________________

**Algorithm used:** ________________________________

**Constraints:** ________________________________

**Differences from textbook:** ________________________________

---

## 🤝 Discussion Questions

### Group Discussion 1: Search Strategy Selection
**Scenario:** You're designing an AI system for each situation below. Discuss which search algorithm you'd choose and why:

1. **Automated theorem proving** - need to find any valid proof
2. **Robot arm motion planning** - minimize energy consumption  
3. **Puzzle solving game** - entertain users with step-by-step solution
4. **Emergency route planning** - find quickest path during disasters

**Discussion Points:**
- What are the priorities in each case?
- How do real-world constraints affect algorithm choice?
- When might you combine multiple search strategies?

### Group Discussion 2: Limitations of Uninformed Search
**Questions for discussion:**
1. What types of problems are poorly suited for uninformed search?
2. How do you know when the search space is too large?
3. What makes a good state representation for search?
4. When should you give up searching and try a different approach?

---

## 🧮 Mathematical Analysis

### Exercise 9: Complexity Calculations

**Given a search problem with:**
- Branching factor (b) = 3
- Solution depth (d) = 4  
- Maximum depth (m) = 6

**Calculate for each algorithm:**

1. **BFS:**
   - Time complexity: ____________________
   - Space complexity: ____________________
   - Actual values: ____________________

2. **DFS:**
   - Time complexity (worst case): ____________________  
   - Space complexity: ____________________
   - Actual values: ____________________

3. **Iterative Deepening:**
   - Time complexity: ____________________
   - Space complexity: ____________________  
   - Actual values: ____________________

**Analysis Questions:**
- At what point does BFS become impractical due to memory?
- How does increasing the branching factor affect each algorithm?
- What if the solution was at depth 10 instead of 4?

---

## ⚗️ Experimental Problems

### Exercise 10: Algorithm Racing

**Design an experiment to compare search algorithms:**

**Experimental Setup:**
1. **Create test problems** of varying difficulty (different grid sizes, obstacle densities)
2. **Implement timing code** to measure execution time
3. **Track memory usage** (number of nodes in frontier and explored set)  
4. **Run algorithms** on the same problems and collect data

**Code Framework:**
```python
import time
import tracemalloc
from collections import deque

def run_experiment(algorithm, problem, iterations=5):
    """
    Run algorithm on problem multiple times and collect statistics.
    """
    times = []
    memory_peaks = []
    
    for i in range(iterations):
        tracemalloc.start()
        start_time = time.time()
        
        result = algorithm(problem)
        
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        times.append(end_time - start_time)  
        memory_peaks.append(peak)
    
    return {
        'avg_time': sum(times) / len(times),
        'avg_memory': sum(memory_peaks) / len(memory_peaks),
        'solution_length': len(result) if result else None
    }

# TODO: Design your experiments here
```

**Questions to Investigate:**
1. How does performance scale with problem size?
2. Which algorithm finds shorter solutions on average?
3. At what problem size does each algorithm become impractical?

---

## 🏆 Challenge Problems

### Challenge 1: Optimal Coffee Shop Route
**Problem:** You need to visit 5 specific coffee shops in a city, starting and ending at your home. Find the shortest route.

**Constraints:**
- Must visit each shop exactly once
- Return home at the end
- Minimize total travel distance

**Questions:**
1. How would you formulate this as a search problem?  
2. Why might standard search algorithms struggle with this?
3. What's the relationship to the famous "Traveling Salesman Problem"?

### Challenge 2: Search in Continuous Space  
**Problem:** A robot needs to navigate from point A to point B in a continuous 2D environment with circular obstacles.

**Challenges:**
1. State space is infinite (continuous positions and orientations)
2. How do you discretize the problem appropriately?
3. What are the trade-offs between discretization resolution and computational cost?

---

## 📝 Submission Guidelines

### Individual Exercises
- Complete exercises 1-6 individually
- Show your work clearly, including reasoning
- For coding exercises, include both code and test results

### Group Work  
- Discussion questions can be completed in groups of 3-4
- Submit group consensus answers with individual reflection

### Formatting
- Use clear headings and organize your work
- Include diagrams and tables where helpful
- For code, include comments and example outputs

### Evaluation Criteria
- **Correctness:** Are the solutions accurate?
- **Completeness:** Are all parts answered thoroughly?  
- **Understanding:** Do explanations show deep comprehension?
- **Application:** Can you apply concepts to new scenarios?

---

**💡 Study Tips:**
- Work through algorithms by hand before implementing
- Draw diagrams to visualize search trees and graphs
- Test your understanding by explaining algorithms to classmates
- Connect search concepts to real applications you use daily

**🔗 Next Week Preview:** We'll learn how to make search more efficient using heuristics and domain knowledge. Get ready for A* search - one of the most important algorithms in AI!