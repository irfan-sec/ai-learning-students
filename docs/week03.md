---
layout: week
title: "Week 3: Problem-Solving by Search II (Informed Search)"
permalink: /week03.html
---

### Navigation
[🏠 Home](index.html) | [← Previous: Week 2](week02.html) | [Next: Week 4 →](week04.html)

---

# Week 3: Problem-Solving by Search II (Informed Search)

## 📚 Learning Objectives
By the end of this week, students will be able to:
- Understand the concept of heuristic functions and their role in search
- Implement and analyze informed search algorithms (Greedy, A*)
- Evaluate heuristic functions for admissibility and consistency
- Apply A* search to solve complex pathfinding problems
- Design domain-specific heuristics for different problem types

---

## 💡 From Uninformed to Informed Search

Last week we explored **uninformed search** strategies that systematically explore the search space without additional knowledge. This week, we'll learn how **domain-specific knowledge** can dramatically improve search efficiency.

![Informed vs Uninformed Search](images/informed-vs-uninformed-search.png)

### The Power of Heuristics

**Heuristic:** A rule of thumb or educated guess that helps guide search toward promising areas of the search space.

**Example:** In navigation, the straight-line distance to the destination is a good heuristic - it doesn't give the exact remaining cost, but provides useful guidance.

---

## 🎯 Heuristic Functions

### Definition
A **heuristic function** h(n) estimates the cost from node n to the nearest goal:
- h(n) = estimated cost from n to goal
- h(goal) = 0 for any goal state
- h(n) ≥ 0 for all nodes n

### Properties of Good Heuristics

#### 1. **Admissibility**
A heuristic is **admissible** if it never overestimates the true cost to reach a goal.
- h(n) ≤ h*(n) for all n
- Where h*(n) is the true optimal cost from n to goal
- Admissible heuristics guarantee optimal solutions in A* search

#### 2. **Consistency (Monotonicity)**  
A heuristic is **consistent** if for every node n and successor n':
- h(n) ≤ c(n,n') + h(n')
- Where c(n,n') is the cost of the action from n to n'
- Consistent heuristics are also admissible
- Ensures f(n) never decreases along any path

![Heuristic Properties](images/heuristic-properties.png)

### Example: Heuristics for Route Finding

```python
import math

def straight_line_distance(city1, city2, coordinates):
    """Euclidean distance between two cities"""
    x1, y1 = coordinates[city1]
    x2, y2 = coordinates[city2]
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def manhattan_distance_cities(city1, city2, coordinates):
    """Manhattan distance between two cities"""
    x1, y1 = coordinates[city1]
    x2, y2 = coordinates[city2]
    return abs(x1 - x2) + abs(y1 - y2)

# Example usage
coordinates = {
    'Arad': (366, 480),
    'Bucharest': (400, 327),
    'Craiova': (253, 288)
}

heuristic = straight_line_distance('Arad', 'Bucharest', coordinates)
print(f"Straight-line distance from Arad to Bucharest: {heuristic:.1f} km")
```

---

## 🔍 Informed Search Algorithms

### 1. Greedy Best-First Search

**Strategy:** Always expand the node that appears closest to the goal (lowest h(n))

```python
import heapq

def greedy_best_first_search(problem, heuristic_fn):
    """Greedy search using only heuristic values"""
    start_node = Node(problem.start_state)
    frontier = [(heuristic_fn(start_node.state), start_node)]
    explored = set()
    
    while frontier:
        _, current = heapq.heappop(frontier)
        
        if problem.is_goal(current.state):
            return reconstruct_path(current)
        
        if current.state in explored:
            continue
            
        explored.add(current.state)
        
        for successor_state, action, cost in problem.get_successors(current.state):
            if successor_state not in explored:
                successor = Node(successor_state, current, action)
                h_cost = heuristic_fn(successor_state)
                heapq.heappush(frontier, (h_cost, successor))
    
    return None
```

**Properties:**
- ❌ **Complete:** No (can get stuck in local minima)
- ❌ **Optimal:** No (ignores path cost)
- ✅ **Time/Space:** O(b^m) but often much better in practice
- **Use Case:** When you need fast solutions and optimality isn't critical

### 2. A* Search (A-star)

**Strategy:** Minimize f(n) = g(n) + h(n)
- g(n) = actual cost from start to n
- h(n) = heuristic estimate from n to goal  
- f(n) = estimated total cost of path through n

```python
import heapq

def a_star_search(problem, heuristic_fn):
    """A* search implementation"""
    start_node = Node(problem.start_state, g_cost=0)
    start_node.h_cost = heuristic_fn(start_node.state, problem.goal_state)
    start_node.f_cost = start_node.g_cost + start_node.h_cost
    
    frontier = [start_node]
    explored = set()
    frontier_dict = {start_node.state: start_node}  # For efficient lookup
    
    while frontier:
        current = heapq.heappop(frontier)
        del frontier_dict[current.state]
        
        if problem.is_goal(current.state):
            return reconstruct_path(current)
        
        explored.add(current.state)
        
        for successor_state, action, cost in problem.get_successors(current.state):
            if successor_state in explored:
                continue
                
            g_cost = current.g_cost + cost
            h_cost = heuristic_fn(successor_state, problem.goal_state)
            f_cost = g_cost + h_cost
            
            successor = Node(successor_state, current, action, g_cost, h_cost, f_cost)
            
            # If state already in frontier with worse cost, replace it
            if successor_state in frontier_dict:
                existing = frontier_dict[successor_state]
                if g_cost < existing.g_cost:
                    # Remove old node and add new one
                    frontier.remove(existing)
                    heapq.heapify(frontier)
                    heapq.heappush(frontier, successor)
                    frontier_dict[successor_state] = successor
            else:
                heapq.heappush(frontier, successor)
                frontier_dict[successor_state] = successor
    
    return None

class Node:
    def __init__(self, state, parent=None, action=None, g_cost=0, h_cost=0, f_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = f_cost
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost
```

**Properties:**
- ✅ **Complete:** Yes (if heuristic is admissible)
- ✅ **Optimal:** Yes (if heuristic is admissible)
- **Time:** O(b^d) but depends heavily on heuristic quality
- **Space:** Keeps all nodes in memory (like BFS)

**Why A* is Optimal:**
If A* selects goal G for expansion, then f(G) = g(G) (since h(G) = 0). For any other node n on the frontier, f(n) ≥ f(G), so g(n) + h(n) ≥ g(G). Since h is admissible, the optimal cost through any path must be ≥ g(G).

---

## 🧭 Common Heuristic Functions

### 1. Geometric Problems (Grid/Map Navigation)

#### Manhattan Distance (L1 Distance)
```python
def manhattan_distance(pos1, pos2):
    """Manhattan distance for grid-based movement"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
```
- **Use:** Grid worlds with 4-directional movement
- **Properties:** Admissible for unit-cost grids
- **Example:** |x1-x2| + |y1-y2|

#### Euclidean Distance (L2 Distance)
```python
import math

def euclidean_distance(pos1, pos2):
    """Straight-line distance"""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
```
- **Use:** Continuous spaces, straight-line movement
- **Properties:** Admissible for straight-line movement
- **Example:** √[(x1-x2)² + (y1-y2)²]

#### Chebyshev Distance (L∞ Distance)
```python
def chebyshev_distance(pos1, pos2):
    """Maximum coordinate difference"""
    return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))
```
- **Use:** Grid worlds with 8-directional movement
- **Properties:** Admissible for diagonal movement allowed

### 2. Puzzle Problems

#### 8-Puzzle Heuristics

**Misplaced Tiles Heuristic:**
```python
def misplaced_tiles(state, goal):
    """Count tiles in wrong positions"""
    return sum(1 for i, tile in enumerate(state) 
               if tile != 0 and tile != goal[i])
```
- Counts tiles in wrong positions
- Admissible (each misplaced tile needs at least 1 move)

**Manhattan Distance for Puzzles:**
```python
def puzzle_manhattan(state, goal):
    """Sum of Manhattan distances for all tiles"""
    distance = 0
    size = int(len(state) ** 0.5)
    
    for i, tile in enumerate(state):
        if tile != 0:
            current_row, current_col = divmod(i, size)
            goal_pos = goal.index(tile)
            goal_row, goal_col = divmod(goal_pos, size)
            distance += abs(current_row - goal_row) + abs(current_col - goal_col)
    
    return distance
```
- Sums Manhattan distances of all tiles to their goal positions
- More informed than misplaced tiles
- Still admissible

---

## 💻 Hands-On Exercise: A* for 8-Puzzle

```python
class EightPuzzleAStar:
    def __init__(self, initial, goal=(1,2,3,4,5,6,7,8,0)):
        self.initial = initial
        self.goal = goal
    
    def actions(self, state):
        moves = []
        blank_pos = state.index(0)
        row, col = divmod(blank_pos, 3)
        
        if row > 0: moves.append('UP')
        if row < 2: moves.append('DOWN') 
        if col > 0: moves.append('LEFT')
        if col < 2: moves.append('RIGHT')
        
        return moves
    
    def result(self, state, action):
        state = list(state)
        blank_pos = state.index(0)
        row, col = divmod(blank_pos, 3)
        
        if action == 'UP': new_pos = (row-1) * 3 + col
        elif action == 'DOWN': new_pos = (row+1) * 3 + col
        elif action == 'LEFT': new_pos = row * 3 + (col-1)
        elif action == 'RIGHT': new_pos = row * 3 + (col+1)
        
        state[blank_pos], state[new_pos] = state[new_pos], state[blank_pos]
        return tuple(state)
    
    def is_goal(self, state):
        return state == self.goal
    
    def get_successors(self, state):
        successors = []
        for action in self.actions(state):
            new_state = self.result(state, action)
            successors.append((new_state, action, 1))  # cost = 1
        return successors

# Test with different heuristics
puzzle = EightPuzzleAStar((1,2,3,4,0,6,7,5,8))

# Compare solutions
solution1 = a_star_search(puzzle, lambda s, g: misplaced_tiles(s, g))
solution2 = a_star_search(puzzle, lambda s, g: puzzle_manhattan(s, g))

print("Solutions found with both heuristics")
print(f"Misplaced tiles: {len(solution1) if solution1 else 'No solution'} steps")
print(f"Manhattan distance: {len(solution2) if solution2 else 'No solution'} steps")
```

---

## 📊 Heuristic Dominance

### Comparing Heuristics

If h₁(n) ≤ h₂(n) ≤ h*(n) for all n, then h₂ **dominates** h₁.
- A* with h₂ will never expand more nodes than A* with h₁
- Better heuristics lead to more efficient search

### Example: 8-Puzzle Heuristics
- h₁ = misplaced tiles
- h₂ = Manhattan distance  
- h₂ dominates h₁ because Manhattan distance is always ≥ misplaced tiles

![Heuristic Dominance](images/heuristic-dominance-comparison.png)

---

## ⚡ Optimizing A* Performance

### 1. Memory-Bounded Variants

**IDA* (Iterative Deepening A*)**
- Uses f-cost cutoffs instead of storing all nodes
- Space complexity: O(bd)
- Time overhead due to repeated work

```python
def ida_star(problem, heuristic_fn):
    """IDA* implementation"""
    threshold = heuristic_fn(problem.start_state, problem.goal_state)
    
    while True:
        result, new_threshold = search_with_threshold(
            problem, problem.start_state, 0, threshold, heuristic_fn, set()
        )
        
        if result != 'CUTOFF':
            return result
        
        if new_threshold == float('inf'):
            return None  # No solution
        
        threshold = new_threshold

def search_with_threshold(problem, state, g_cost, threshold, heuristic_fn, path):
    f_cost = g_cost + heuristic_fn(state, problem.goal_state)
    
    if f_cost > threshold:
        return 'CUTOFF', f_cost
    
    if problem.is_goal(state):
        return [state], f_cost
    
    if state in path:
        return 'CUTOFF', float('inf')
    
    path.add(state)
    min_threshold = float('inf')
    
    for successor_state, action, cost in problem.get_successors(state):
        result, new_threshold = search_with_threshold(
            problem, successor_state, g_cost + cost, threshold, heuristic_fn, path
        )
        
        if result != 'CUTOFF':
            path.remove(state)
            return [state] + result, new_threshold
        
        if new_threshold < min_threshold:
            min_threshold = new_threshold
    
    path.remove(state)
    return 'CUTOFF', min_threshold
```

### 2. Performance Analysis

**Time Complexity of A***
- **Best case:** O(bd) if heuristic is perfect
- **Worst case:** O(b^d) if heuristic provides no information  
- **Typical case:** Depends on heuristic accuracy

**Effective Branching Factor**
Measure of search efficiency:
- b* = effective branching factor
- Lower b* indicates better heuristic performance

---

## 🔗 Curated Resources

### Essential Reading
- [AIMA Chapter 3.5-3.6](https://aima.cs.berkeley.edu/) - Informed Search and Heuristics
- [A* Search Algorithm](https://theory.stanford.edu/~amitp/GameProgramming/AStarComparison.html) - Interactive tutorial

### Video Resources
- [A* Search Algorithm](https://www.youtube.com/watch?v=ySN5Wnu88nE) - Computerphile (9 min)
- [Heuristics and A* Search](https://www.youtube.com/watch?v=-L-WgKMFuhE) - CS50 AI (25 min)
- [A* Pathfinding Visualization](https://www.youtube.com/watch?v=aKYlikFAV4k) (15 min)

### Interactive Learning
- [A* Pathfinding Visualizer](https://qiao.github.io/PathFinding.js/visual/)
- [Red Blob Games - A* Tutorial](https://www.redblobgames.com/pathfinding/a-star/introduction.html)
- [Pathfinding Visualizer](https://clementmihailescu.github.io/Pathfinding-Visualizer/)

### Practice Problems
- [LeetCode: Word Ladder II](https://leetcode.com/problems/word-ladder-ii/) (A* application)
- [Sliding Puzzle](https://leetcode.com/problems/sliding-puzzle/) (8-puzzle variant)
- [Path with Maximum Gold](https://leetcode.com/problems/path-with-maximum-gold/)

---

## 🎯 Key Concepts Summary

### Heuristic Functions
- **Admissible:** Never overestimate true cost
- **Consistent:** Satisfy triangle inequality
- **Dominant:** More informed heuristics are better

### Algorithm Properties
- **Greedy Best-First:** Fast but not optimal
- **A*:** Optimal with admissible heuristics
- **Performance depends critically on heuristic quality**

### Design Principles
- Use domain knowledge to create good heuristics
- Balance computation time vs. search reduction
- Consider memory constraints in algorithm choice

---

## 🤔 Discussion Questions

1. Can you have a heuristic that's too good? What are the trade-offs?
2. How would you design a heuristic for the Traveling Salesman Problem?
3. Why is consistency a stronger property than admissibility?
4. In what situations might Greedy Best-First search outperform A*?
5. How do you balance heuristic computation time vs. search reduction?

---

## 🔍 Looking Ahead

Next week, we'll explore **adversarial search** and game playing, where we deal with opponents trying to work against us. We'll learn about Minimax algorithm and Alpha-Beta pruning - essential techniques for creating AI that can play strategic games!

**Preview of Week 4:**
- Game trees and the minimax algorithm
- Alpha-beta pruning for efficiency
- Evaluation functions for non-terminal positions
- Applications to chess, checkers, and other games

---

### Navigation
[🏠 Home](index.html) | [← Previous: Week 2](week02.html) | [Next: Week 4 →](week04.html)

---

*"A* is to search what quicksort is to sorting - a fundamental algorithm that every computer scientist should understand."*