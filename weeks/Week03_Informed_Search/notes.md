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

![Informed vs Uninformed Search](https://media.geeksforgeeks.org/wp-content/uploads/AI-algos-1-e1547043543151.png)

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

![Heuristic Properties](https://almablog-media.s3.ap-south-1.amazonaws.com/Heuristic_Search_Algorithms_in_AI_cf871e375c.png)

---

## 🔍 Informed Search Algorithms

### 1. Greedy Best-First Search

**Strategy:** Always expand the node that appears closest to the goal (lowest h(n))

**Algorithm:**
```
1. Initialize frontier as priority queue ordered by h(n)
2. While frontier is not empty:
   3. Remove node n with lowest h(n)
   4. If n is goal, return solution
   5. Expand n and add children to frontier
```

**Properties:**
- ❌ **Complete:** No (can get stuck in local minima)
- ❌ **Optimal:** No (ignores path cost)
- ✅ **Time/Space:** O(b^m) but often much better in practice
- **Use Case:** When you need fast solutions and optimality isn't critical

**Example:** In route finding, always head toward the destination regardless of accumulated distance.

### 2. A* Search (A-star)

**Strategy:** Minimize f(n) = g(n) + h(n)
- g(n) = actual cost from start to n
- h(n) = heuristic estimate from n to goal  
- f(n) = estimated total cost of path through n

**Algorithm:**
```
1. Initialize frontier as priority queue ordered by f(n)
2. Initialize explored set
3. While frontier is not empty:
   4. Remove node n with lowest f(n)
   5. If n is goal, return solution
   6. Add n to explored set
   7. For each successor n' of n:
      8. If n' in explored with lower g(n'), skip
      9. If n' in frontier with lower g(n'), skip
      10. Add n' to frontier
```

**Properties:**
- ✅ **Complete:** Yes (if heuristic is admissible)
- ✅ **Optimal:** Yes (if heuristic is admissible)
- **Time:** O(b^d) but depends heavily on heuristic quality
- **Space:** Keeps all nodes in memory (like BFS)

**Why A* is Optimal:**
If A* selects goal G for expansion, then f(G) = g(G) (since h(G) = 0). For any other node n on the frontier, f(n) ≥ f(G), so g(n) + h(n) ≥ g(G). Since h is admissible, g(n) + h(n) ≥ g(n) + h*(n) ≥ optimal cost through n ≥ g(G).

---

## 🧭 Common Heuristic Functions

### 1. Geometric Problems (Grid/Map Navigation)

#### Manhattan Distance (L1 Distance)
```python
def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
```
- **Use:** Grid worlds with 4-directional movement
- **Properties:** Admissible for unit-cost grids
- **Example:** |x1-x2| + |y1-y2|

#### Euclidean Distance (L2 Distance)
```python
import math

def euclidean_distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
```
- **Use:** Continuous spaces, straight-line movement
- **Properties:** Admissible for straight-line movement
- **Example:** √[(x1-x2)² + (y1-y2)²]

#### Chebyshev Distance (L∞ Distance)
```python
def chebyshev_distance(pos1, pos2):
    return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))
```
- **Use:** Grid worlds with 8-directional movement
- **Properties:** Admissible for diagonal movement allowed

### 2. Puzzle Problems

#### 8-Puzzle Heuristics

**Misplaced Tiles Heuristic:**
```python
def misplaced_tiles(state, goal):
    return sum(1 for i, tile in enumerate(state) 
               if tile != 0 and tile != goal[i])
```
- Counts tiles in wrong positions
- Admissible (each misplaced tile needs at least 1 move)

**Manhattan Distance for Puzzles:**
```python
def puzzle_manhattan(state, goal):
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

## 📊 Heuristic Dominance

### Comparing Heuristics

If h₁(n) ≤ h₂(n) ≤ h*(n) for all n, then h₂ **dominates** h₁.
- A* with h₂ will never expand more nodes than A* with h₁
- Better heuristics lead to more efficient search

### Example: 8-Puzzle Heuristics
- h₁ = misplaced tiles
- h₂ = Manhattan distance  
- h₂ dominates h₁ because Manhattan distance is always ≥ misplaced tiles

---

## 💻 A* Implementation Strategy

### Basic A* Template
```python
import heapq

def a_star_search(problem, heuristic_fn):
    start_node = Node(problem.start_state, g_cost=0)
    start_node.f_cost = heuristic_fn(start_node.state, problem.goal_state)
    
    frontier = [start_node]  # Priority queue
    explored = set()
    
    while frontier:
        current = heapq.heappop(frontier)
        
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
            
            # Check if we found a better path to this state
            existing = find_in_frontier(frontier, successor_state)
            if existing and existing.g_cost <= g_cost:
                continue
            
            heapq.heappush(frontier, successor)
    
    return None  # No solution found
```

---

## 🎮 Practical Applications

### 1. Video Game Pathfinding
- NPCs navigating game worlds
- Real-time constraints require fast heuristics
- Often use hierarchical approaches for large maps

### 2. Robotics Path Planning
- Navigate physical environments
- Account for robot size and movement constraints
- Safety considerations in heuristic design

### 3. Network Routing
- Find optimal paths through networks
- Heuristics based on geographic distance or network topology
- Quality of Service constraints

---

## ⚡ Optimizing A* Performance

### 1. Memory-Bounded Variants

**IDA\* (Iterative Deepening A\*)**
- Uses f-cost cutoffs instead of storing all nodes
- Space complexity: O(bd)
- Time overhead due to repeated work

**Simplified Memory-Bounded A\* (SMA\*)**
- Drops nodes when memory is full
- Keeps best nodes in memory

### 2. Heuristic Preprocessing
- Precompute heuristic values
- Pattern databases for puzzle problems
- Landmark-based heuristics

### 3. Bidirectional Search
- Search from both start and goal
- Meet in the middle
- Can significantly reduce search space

---

## 📈 Performance Analysis

### Time Complexity of A*
- **Best case:** O(bd) if heuristic is perfect
- **Worst case:** O(b^d) if heuristic provides no information  
- **Typical case:** Depends on heuristic accuracy

### Effective Branching Factor
Measure of search efficiency:
- b* = effective branching factor
- N = total nodes generated
- d = depth of solution
- N = 1 + b* + (b*)² + ... + (b*)^d

Lower b* indicates better heuristic performance.

---

## 🤔 Discussion Questions

1. Can you have a heuristic that's too good? What are the trade-offs?
2. How would you design a heuristic for the Traveling Salesman Problem?
3. Why is consistency a stronger property than admissibility?
4. In what situations might Greedy Best-First search outperform A*?

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

## 🔍 Looking Ahead

Next week, we'll explore **adversarial search** and game playing, where we deal with opponents trying to work against us. We'll learn about Minimax algorithm and Alpha-Beta pruning - essential techniques for creating AI that can play strategic games!

---

*"A* is to search what quicksort is to sorting - a fundamental algorithm that every computer scientist should understand."*
