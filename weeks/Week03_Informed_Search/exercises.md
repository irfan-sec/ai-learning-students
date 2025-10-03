# Week 3 Exercises: Problem-Solving by Search II (Informed Search)

## 🎯 Learning Goals Review
After completing these exercises, you should be able to:
- Implement and apply informed search algorithms (greedy best-first, A*)
- Design and evaluate heuristic functions for different problems
- Prove admissibility and consistency of heuristics
- Compare performance of informed vs uninformed search
- Apply A* to real-world pathfinding problems

---

## 📝 Conceptual Questions

### Question 1: Understanding Heuristics
**Consider the 8-puzzle problem with the following heuristics:**
1. h₁(n) = number of misplaced tiles
2. h₂(n) = sum of Manhattan distances of tiles from goal positions
3. h₃(n) = sum of Euclidean distances of tiles from goal positions

**Tasks:**
a) Which heuristic(s) are admissible? Prove your answer.
b) Which heuristic is likely to be most effective in practice? Why?
c) Is h₂ consistent? Provide a proof or counterexample.

---

### Question 2: Algorithm Comparison
**Compare A* and Greedy Best-First Search:**

| Aspect | A* | Greedy Best-First |
|--------|-----|------------------|
| **Completeness** | | |
| **Optimality** | | |
| **Time Complexity** | | |
| **Space Complexity** | | |
| **Use Cases** | | |

**Follow-up:**
- When might you prefer greedy search over A*?
- Can greedy search ever be optimal? Under what conditions?

---

### Question 3: Heuristic Properties
**Given the following graph and heuristic values:**

```
      (5)
    A ─────── B
    │         │
(6) │         │ (2)
    │         │
    C ─────── D
      (3)     
```
Heuristics to goal D: h(A)=5, h(B)=2, h(C)=3, h(D)=0

**Determine:**
a) Is this heuristic admissible?
b) Is this heuristic consistent?
c) Trace A* execution from A to D
d) Would greedy best-first find the optimal path?

---

## 🔍 Algorithm Analysis

### Exercise 1: A* Trace
**Problem:** Find the shortest path from S to G using A*.

```
Grid with costs (each cell = cost 1):
S . . . .
# # . # .
. . . # .
. # # # .
. . . . G

h(n) = Manhattan distance to G
```

**Your Task:**
1. Show the open and closed lists at each step
2. Calculate f(n) = g(n) + h(n) for each node explored
3. Draw the final path found
4. Count total nodes expanded
5. Compare with BFS (how many nodes would BFS expand?)

**Template:**
```
Step 1:
Open: [(S, g=0, h=8, f=8)]
Closed: []
Current: S
Expand to: ...

Step 2:
Open: [...]
Closed: [S]
...
```

---

### Exercise 2: Heuristic Design
**Problem:** Route planning between cities.

**City Network:**
```
        50
    A ──────── B
    │          │
 80 │          │ 40
    │          │
    C ──────── D
        60
```
**Straight-line distances to goal D:**
- h(A) = 90
- h(B) = 50  
- h(C) = 60
- h(D) = 0

**Tasks:**
a) Is the straight-line distance heuristic admissible? Prove it.
b) Run A* from A to D. Show all steps.
c) Design an alternative heuristic for this problem
d) Would your heuristic be more or less effective? Why?

---

## 💻 Programming Exercises

### Exercise 3: Implement A* Search

Complete the A* implementation:

```python
from queue import PriorityQueue
from typing import List, Tuple, Callable, Optional

class Node:
    def __init__(self, state, parent=None, action=None, g=0, h=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.g = g  # Cost from start
        self.h = h  # Heuristic to goal
        self.f = g + h  # Total estimated cost
    
    def __lt__(self, other):
        return self.f < other.f

def a_star_search(problem, heuristic: Callable):
    """
    A* search algorithm.
    
    Args:
        problem: Search problem with initial state, goal test, and actions
        heuristic: Function that estimates cost from state to goal
    
    Returns:
        List of actions from start to goal, or None if no solution
    """
    # TODO: Implement A* search
    # Initialize open list with start node
    # Initialize closed set to track visited states
    # Loop:
    #   - Get node with lowest f-value from open list
    #   - If goal, reconstruct path
    #   - Generate successors
    #   - Add successors to open list with proper f-values
    pass

def reconstruct_path(node):
    """Reconstruct path from start to goal node."""
    # TODO: Implement path reconstruction
    pass

# Test with 8-puzzle
def manhattan_distance(state, goal):
    """Calculate Manhattan distance heuristic for 8-puzzle."""
    # TODO: Implement Manhattan distance
    pass

def misplaced_tiles(state, goal):
    """Count number of misplaced tiles."""
    # TODO: Implement misplaced tiles heuristic
    pass
```

**Test Cases:**
```python
# 8-puzzle initial state
initial = [[1, 2, 3],
           [4, 0, 5],
           [6, 7, 8]]

goal = [[1, 2, 3],
        [4, 5, 6],
        [7, 8, 0]]

# Your implementation should find solution and compare heuristics
solution_h1 = a_star_search(puzzle, misplaced_tiles)
solution_h2 = a_star_search(puzzle, manhattan_distance)

print(f"Nodes expanded with h1: {nodes_h1}")
print(f"Nodes expanded with h2: {nodes_h2}")
```

---

### Exercise 4: Grid Pathfinding

Implement A* for grid-based pathfinding:

```python
import numpy as np
import matplotlib.pyplot as plt

class GridWorld:
    def __init__(self, grid):
        """
        grid: 2D array where 0=free, 1=obstacle
        """
        self.grid = np.array(grid)
        self.rows, self.cols = self.grid.shape
    
    def is_valid(self, pos):
        """Check if position is valid and not an obstacle."""
        row, col = pos
        # TODO: Implement validity check
        pass
    
    def get_neighbors(self, pos):
        """Get valid neighboring positions (4-connected)."""
        # TODO: Return list of valid neighbors
        pass
    
    def heuristic(self, pos, goal):
        """Manhattan distance heuristic."""
        # TODO: Implement Manhattan distance
        pass
    
    def find_path(self, start, goal):
        """Find shortest path using A*."""
        # TODO: Implement A* for grid
        pass
    
    def visualize_path(self, start, goal, path):
        """Visualize the grid, obstacles, and found path."""
        # TODO: Create visualization
        pass

# Test case
grid = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0]
]

world = GridWorld(grid)
path = world.find_path((0, 0), (4, 4))
world.visualize_path((0, 0), (4, 4), path)
```

---

### Exercise 5: Heuristic Comparison

Write a program to compare different heuristics:

```python
def compare_heuristics(problem, heuristics, start, goal):
    """
    Compare performance of different heuristics.
    
    Args:
        problem: Search problem
        heuristics: List of (name, heuristic_function) tuples
        start: Start state
        goal: Goal state
    
    Returns:
        Dictionary with statistics for each heuristic
    """
    results = {}
    
    for name, h in heuristics:
        # TODO: Run A* with this heuristic
        # Track: nodes expanded, path length, runtime
        pass
    
    return results

def plot_comparison(results):
    """Visualize heuristic comparison."""
    # TODO: Create bar plots comparing metrics
    pass

# Example usage
heuristics = [
    ("Misplaced Tiles", misplaced_tiles),
    ("Manhattan Distance", manhattan_distance),
    ("Euclidean Distance", euclidean_distance)
]

results = compare_heuristics(puzzle_problem, heuristics, initial, goal)
plot_comparison(results)
```

---

## 🧩 Problem-Solving Challenges

### Challenge 1: Romania Road Trip
**Scenario:** Plan a route from Arad to Bucharest using the Romania map from AIMA.

**Cities and connections:**
- Arad → Sibiu (140), Timisoara (118), Zerind (75)
- Sibiu → Fagaras (99), Rimnicu Vilcea (80)
- Fagaras → Bucharest (211)
- Rimnicu Vilcea → Pitesti (97), Craiova (146)
- Pitesti → Bucharest (101)
- (... other connections)

**Straight-line distances to Bucharest:**
- Arad: 366, Sibiu: 253, Fagaras: 176, Pitesti: 100, ...

**Tasks:**
1. Implement the Romania problem
2. Find optimal path using A*
3. Compare with greedy best-first search
4. Visualize the search process
5. Analyze why A* outperforms greedy search

---

### Challenge 2: N-Puzzle Solver
**Build a general N-puzzle solver (works for 8-puzzle, 15-puzzle, etc.):**

**Requirements:**
- Support arbitrary puzzle sizes
- Implement multiple heuristics
- Add animation showing solution steps
- Track and display statistics
- Handle unsolvable configurations

**Advanced:**
- Implement IDA* for memory efficiency
- Add pattern database heuristic
- Compare with different variants (Weighted A*, etc.)

---

### Challenge 3: Game Pathfinding
**Create a simple maze game with intelligent pathfinding:**

**Features:**
- Player-controlled character
- AI-controlled enemies using A*
- Dynamic obstacles
- Real-time pathfinding
- Visualization of AI's search process

**Technical Requirements:**
```python
class MazeGame:
    def __init__(self, width, height):
        # TODO: Initialize game
        pass
    
    def update_enemy_positions(self):
        """Use A* to move enemies toward player."""
        # TODO: Implement enemy AI
        pass
    
    def handle_dynamic_obstacles(self):
        """Replan paths when obstacles change."""
        # TODO: Implement dynamic replanning
        pass
```

---

## 🔬 Advanced Exercises

### Exercise 6: Prove Heuristic Properties

**Theorem:** If h(n) is consistent, then A* is optimally efficient.

**Your task:**
1. Prove that consistency implies admissibility
2. Show that if h is consistent, no node is expanded twice
3. Prove that A* with consistent h is optimally efficient

---

### Exercise 7: Design Domain-Specific Heuristics

**For each domain, design an admissible heuristic:**

a) **Robot Vacuum Cleaner:**
   - State: (robot position, set of dirty cells)
   - Goal: All cells clean
   - Design heuristic and prove admissibility

b) **Blocks World:**
   - State: Configuration of blocks
   - Goal: Target configuration
   - Design heuristic considering block positions

c) **Traveling Salesman:**
   - State: (current city, unvisited cities)
   - Goal: Visit all cities, return to start
   - Design heuristic (hint: MST lower bound)

---

### Exercise 8: Memory-Bounded Search

**Implement IDA* (Iterative Deepening A*):**

```python
def ida_star(problem, heuristic):
    """
    Memory-efficient A* using iterative deepening.
    
    Args:
        problem: Search problem
        heuristic: Admissible heuristic function
    
    Returns:
        Solution path or None
    """
    # TODO: Implement IDA*
    # Use depth-first search with f-limit
    # Iteratively increase f-limit
    pass

# Compare memory usage
def memory_comparison():
    """Compare memory usage of A* vs IDA*."""
    # TODO: Measure and compare memory consumption
    pass
```

---

## 📊 Analysis Questions

### Question 4: Algorithm Trade-offs
**Analyze the following scenarios and choose the best algorithm:**

a) **Large state space, good heuristic available:**
   - Algorithm: ___________
   - Justification: ___________

b) **Memory constraints, need optimal solution:**
   - Algorithm: ___________
   - Justification: ___________

c) **Real-time requirements, optimality not critical:**
   - Algorithm: ___________
   - Justification: ___________

d) **Dynamic environment, obstacles change:**
   - Algorithm: ___________
   - Justification: ___________

---

### Question 5: Heuristic Effectiveness

**Effective Branching Factor (EBF):**
The EBF b* is defined such that: N = 1 + b* + (b*)² + ... + (b*)^d

Where N = total nodes generated, d = solution depth

**Calculate EBF for:**
- A* with h₁ (misplaced tiles): 50 nodes, depth 10
- A* with h₂ (Manhattan distance): 30 nodes, depth 10
- Greedy best-first with h₂: 40 nodes, depth 12

**Which heuristic is most effective? Why?**

---

## 📝 Submission Guidelines

### What to Submit:
1. **Conceptual answers** with clear explanations and proofs
2. **A* implementation** with complete working code
3. **Test results** showing algorithm comparisons
4. **Visualizations** of search processes and paths
5. **Analysis report** (500-750 words) discussing:
   - Heuristic design principles
   - Performance comparisons
   - Practical applications
   - Challenges encountered

### Evaluation Criteria:
- **Correctness:** Algorithms work properly and find optimal solutions
- **Efficiency:** Code is well-optimized and uses appropriate data structures
- **Analysis:** Deep understanding of heuristics and search properties
- **Clarity:** Code is clean, documented, and easy to understand
- **Creativity:** Novel heuristics or interesting applications

### Due Date: End of Week 3

---

## 💡 Study Tips

### Understanding A*
- **Visualize:** Draw out f-values and open/closed lists
- **Compare:** Run same problem with different heuristics
- **Prove:** Practice proving admissibility and consistency
- **Debug:** Check your priority queue implementation carefully

### Common Mistakes to Avoid
- **Wrong f-values:** Remember f(n) = g(n) + h(n)
- **Reopening nodes:** Properly check if node is in closed list
- **Inefficient priority queue:** Use heap/priority queue, not list
- **Inconsistent heuristics:** Can lead to non-optimal solutions

### Advanced Topics to Explore
- **Weighted A*:** Trade optimality for speed (f = g + w·h, w > 1)
- **Bidirectional Search:** Search from both start and goal
- **Jump Point Search:** Fast pathfinding on grids
- **Anytime algorithms:** Improve solution quality over time

---

**🎯 Key Takeaway:** A* is optimal and complete when using an admissible heuristic. The effectiveness of A* depends critically on the quality of the heuristic function. Practice designing and evaluating heuristics for different domains!
