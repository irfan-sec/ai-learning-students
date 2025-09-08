---
layout: week
title: "Week 2: Problem-Solving by Search I (Uninformed Search)"
permalink: /week02.html
---

### Navigation
[🏠 Home](index.html) | [← Previous: Week 1](week01.html) | [Next: Week 3 →](week03.html)

---

# Week 2: Problem-Solving by Search I (Uninformed Search)

## 📚 Learning Objectives
By the end of this week, students will be able to:
- Formulate real-world problems as search problems
- Understand the components of search problems: states, actions, goals
- Implement and analyze uninformed search algorithms (BFS, DFS, UCS)
- Compare search strategies based on completeness, optimality, and complexity
- Apply search algorithms to solve pathfinding and puzzle problems

---

## 🔍 What is Problem-Solving as Search?

Many AI problems can be solved by **searching** through a space of possible solutions. Search is fundamental to AI because intelligent agents often need to:
- Find a sequence of actions to reach a goal
- Choose the best path among many alternatives
- Explore possibilities systematically

![Search Tree Example](images/search-tree-example.png)

### Real-World Examples
- **GPS Navigation:** Find the shortest route from A to B
- **Game Playing:** Find the best move in chess or checkers  
- **Puzzle Solving:** Solve Rubik's cube or sliding puzzle
- **Resource Allocation:** Optimize schedules or assignments

---

## 🧩 Formulating Search Problems

### Components of a Search Problem

Every search problem has these key components:

1. **Initial State:** Where we start
2. **Actions:** What we can do in each state  
3. **Transition Model:** Results of actions (successor function)
4. **Goal Test:** How to recognize if we've reached the goal
5. **Path Cost:** Cost of each action sequence

### Example: 8-Puzzle Problem

![8-Puzzle](images/8-puzzle.png)

**Problem Formulation:**
- **Initial State:** Current arrangement of tiles
- **Actions:** Move blank space (up, down, left, right)
- **Transition Model:** New arrangement after moving blank
- **Goal Test:** Does current state match target arrangement?
- **Path Cost:** Number of moves (each move costs 1)

### Example: Route Finding

```python
class RouteFindingProblem:
    def __init__(self, initial, goal, map_graph):
        self.initial = initial
        self.goal = goal
        self.map = map_graph
    
    def actions(self, state):
        """Return list of cities reachable from current city"""
        return list(self.map[state].keys())
    
    def result(self, state, action):
        """Return the city reached by taking action from state"""
        return action
    
    def goal_test(self, state):
        """Check if we've reached the goal city"""
        return state == self.goal
    
    def path_cost(self, cost_so_far, state1, action, state2):
        """Return cost of path from state1 to state2 via action"""
        return cost_so_far + self.map[state1][action]
```

**Problem:** Find route from Arad to Bucharest in Romania

- **Initial State:** "At Arad"
- **Actions:** Drive to adjacent cities
- **Transition Model:** "At city X" after driving from current city
- **Goal Test:** "At Bucharest"  
- **Path Cost:** Sum of distances traveled

---

## 🌳 Search Trees vs. Search Graphs

### Search Trees
- **Nodes:** Represent states in the search space
- **Edges:** Represent actions/transitions
- **Root:** Initial state
- **Leaves:** Unexpanded states
- **Path:** Sequence from root to node

### Tree Search vs. Graph Search
- **Tree Search:** May revisit states (can get stuck in loops)
- **Graph Search:** Keeps track of explored states (avoids loops)

![Tree vs Graph Search](images/tree-vs-graph-search.png)

---

## 🔍 Uninformed Search Strategies

**Uninformed (Blind) Search:** Uses only information available in problem definition
- No domain-specific knowledge about which states are better
- Systematic exploration of the search space

### 1. Breadth-First Search (BFS)

**Strategy:** Expand shallowest unexpanded node first

```python
from collections import deque

def breadth_first_search(problem):
    """BFS implementation using a queue"""
    if problem.goal_test(problem.initial):
        return [problem.initial]
    
    frontier = deque([Node(problem.initial)])
    explored = set()
    
    while frontier:
        node = frontier.popleft()
        explored.add(node.state)
        
        for action in problem.actions(node.state):
            child_state = problem.result(node.state, action)
            child = Node(child_state, node, action)
            
            if child_state not in explored and child not in frontier:
                if problem.goal_test(child_state):
                    return solution_path(child)
                frontier.append(child)
    
    return None  # No solution found
```

**Properties:**
- ✅ **Complete:** Yes (if branching factor is finite)
- ✅ **Optimal:** Yes (if step costs are equal)
- ❌ **Time Complexity:** O(b^d)
- ❌ **Space Complexity:** O(b^d)

Where: b = branching factor, d = depth of solution

**When to Use:** When you need the shallowest solution

### 2. Depth-First Search (DFS)

**Strategy:** Expand deepest unexpanded node first

```python
def depth_first_search(problem):
    """DFS implementation using a stack"""
    frontier = [Node(problem.initial)]
    explored = set()
    
    while frontier:
        node = frontier.pop()  # Remove from end (stack behavior)
        
        if problem.goal_test(node.state):
            return solution_path(node)
        
        if node.state not in explored:
            explored.add(node.state)
            
            for action in problem.actions(node.state):
                child_state = problem.result(node.state, action)
                child = Node(child_state, node, action)
                frontier.append(child)
    
    return None
```

**Properties:**
- ❓ **Complete:** No (can get stuck in infinite paths)
- ❌ **Optimal:** No
- ✅ **Time Complexity:** O(b^m) - could be better or worse than BFS
- ✅ **Space Complexity:** O(bm) - much better than BFS

Where: m = maximum depth of search space

**When to Use:** When memory is limited and solutions are deep

### 3. Depth-Limited Search (DLS)

**Strategy:** DFS with a predetermined depth limit

- Solves the infinite-path problem of DFS
- Not complete if solution is deeper than limit
- Useful when you know maximum solution depth

```python
def depth_limited_search(problem, limit):
    """DFS with depth limit"""
    return recursive_dls(Node(problem.initial), problem, limit)

def recursive_dls(node, problem, limit):
    if problem.goal_test(node.state):
        return solution_path(node)
    elif limit == 0:
        return 'cutoff'
    else:
        cutoff_occurred = False
        for action in problem.actions(node.state):
            child_state = problem.result(node.state, action)
            child = Node(child_state, node, action)
            result = recursive_dls(child, problem, limit - 1)
            
            if result == 'cutoff':
                cutoff_occurred = True
            elif result is not None:
                return result
        
        return 'cutoff' if cutoff_occurred else None
```

### 4. Iterative Deepening Search (IDS)

**Strategy:** Gradually increase depth limit (DLS with limits 0, 1, 2, ...)

```python
def iterative_deepening_search(problem):
    """IDS: Gradually increase depth limit"""
    for depth in range(0, float('inf')):
        result = depth_limited_search(problem, depth)
        if result != 'cutoff':
            return result
```

**Properties:**
- ✅ **Complete:** Yes
- ✅ **Optimal:** Yes (if step costs equal)  
- ✅ **Time Complexity:** O(b^d)
- ✅ **Space Complexity:** O(bd)

**Why it Works:** Most nodes are at deepest level, so repetition isn't costly

### 5. Uniform-Cost Search (UCS)

**Strategy:** Expand node with lowest path cost first

```python
import heapq

def uniform_cost_search(problem):
    """UCS implementation using priority queue"""
    frontier = [(0, Node(problem.initial))]  # (cost, node)
    explored = set()
    
    while frontier:
        cost, node = heapq.heappop(frontier)
        
        if problem.goal_test(node.state):
            return solution_path(node)
        
        if node.state not in explored:
            explored.add(node.state)
            
            for action in problem.actions(node.state):
                child_state = problem.result(node.state, action)
                child_cost = problem.path_cost(node.path_cost, 
                                             node.state, action, child_state)
                child = Node(child_state, node, action, child_cost)
                
                if child_state not in explored:
                    heapq.heappush(frontier, (child_cost, child))
    
    return None
```

**Properties:**
- ✅ **Complete:** Yes (if step costs ≥ ε > 0)
- ✅ **Optimal:** Yes
- **Time/Space Complexity:** O(b^⌈C*/ε⌉) 

Where: C* = cost of optimal solution

**When to Use:** When actions have different costs

---

## 📊 Search Strategy Comparison

| Strategy | Complete? | Optimal? | Time | Space | Notes |
|----------|-----------|----------|------|-------|--------|
| **BFS** | Yes* | Yes** | O(b^d) | O(b^d) | High memory usage |
| **DFS** | No | No | O(b^m) | O(bm) | Memory efficient |
| **DLS** | No | No | O(b^l) | O(bl) | Needs good limit |
| **IDS** | Yes* | Yes** | O(b^d) | O(bd) | Best of both worlds |
| **UCS** | Yes*** | Yes | O(b^⌈C*/ε⌉) | O(b^⌈C*/ε⌉) | Handles varying costs |

\* If branching factor is finite  
** If step costs are equal  
*** If step costs ≥ ε > 0

---

## 💻 Hands-On Exercise: Implement BFS for 8-Puzzle

```python
class EightPuzzle:
    """8-puzzle problem implementation"""
    
    def __init__(self, initial, goal=(1,2,3,4,5,6,7,8,0)):
        self.initial = initial
        self.goal = goal
    
    def actions(self, state):
        """Return possible moves for blank tile (0)"""
        moves = []
        blank_pos = state.index(0)
        row, col = divmod(blank_pos, 3)
        
        if row > 0: moves.append('UP')
        if row < 2: moves.append('DOWN') 
        if col > 0: moves.append('LEFT')
        if col < 2: moves.append('RIGHT')
        
        return moves
    
    def result(self, state, action):
        """Return new state after moving blank tile"""
        state = list(state)
        blank_pos = state.index(0)
        row, col = divmod(blank_pos, 3)
        
        if action == 'UP':
            new_pos = (row-1) * 3 + col
        elif action == 'DOWN':
            new_pos = (row+1) * 3 + col
        elif action == 'LEFT':
            new_pos = row * 3 + (col-1)
        elif action == 'RIGHT':
            new_pos = row * 3 + (col+1)
        
        state[blank_pos], state[new_pos] = state[new_pos], state[blank_pos]
        return tuple(state)
    
    def goal_test(self, state):
        return state == self.goal

# Test your implementation
puzzle = EightPuzzle((1,2,3,4,5,6,0,7,8))
solution = breadth_first_search(puzzle)
print(f"Solution found: {solution is not None}")
```

---

## 🔗 Curated Resources

### Essential Reading
- [AIMA Chapter 3](https://aima.cs.berkeley.edu/) - Problem-Solving by Search
- [Pathfinding Visualizer](https://clementmihailescu.github.io/Pathfinding-Visualizer/) - Interactive search visualization

### Video Resources
- [Search Algorithms - CS50 AI](https://www.youtube.com/watch?v=TjFXEUCMqI8) (50 min)
- [BFS vs DFS Visualization](https://www.youtube.com/watch?v=pcKY4hjDrxk) (8 min)
- [Search Algorithms - Computerphile](https://www.youtube.com/watch?v=6VwqZWygk6E) (12 min)

### Interactive Learning
- [Graph Search Visualization - USF](https://www.cs.usfca.edu/~galles/visualization/BFS.html)
- [PathFinding.js Visual](https://qiao.github.io/PathFinding.js/visual/)
- [Search Algorithm Simulator](https://visualgo.net/en/dfs)

### Practice Problems
- [LeetCode: Number of Islands](https://leetcode.com/problems/number-of-islands/) (BFS/DFS)
- [LeetCode: Word Ladder](https://leetcode.com/problems/word-ladder/) (BFS)
- [HackerRank: Connected Cells](https://www.hackerrank.com/challenges/connected-cell-in-a-grid)

---

## 🎯 Key Concepts Summary

### Search Problem Components
1. **State Space:** All possible states
2. **Search Tree:** Tree of paths from initial state  
3. **Frontier:** Set of unexpanded nodes
4. **Explored Set:** Set of expanded nodes

### Performance Measures
- **Completeness:** Does it always find a solution if one exists?
- **Optimality:** Does it find the best solution?
- **Time Complexity:** How long does it take?
- **Space Complexity:** How much memory does it use?

### Strategy Selection
- **BFS:** When solution is shallow and memory isn't an issue
- **DFS:** When memory is limited and you need any solution
- **IDS:** When you want BFS benefits with DFS memory usage
- **UCS:** When actions have different costs

---

## 🤔 Discussion Questions

1. Why might you choose DFS over BFS for a very large search space?
2. In what scenarios would Uniform-Cost Search behave identically to BFS?
3. How would you modify BFS to find all solutions at the shallowest depth?
4. What are the trade-offs between tree search and graph search?
5. When would Iterative Deepening be preferred over BFS?

---

## 🔍 Looking Ahead

Next week, we'll learn about **informed search** strategies that use domain-specific knowledge (heuristics) to guide the search more efficiently. We'll explore A* search, one of the most important algorithms in AI!

**Preview of Week 3:**
- Heuristic functions and admissibility
- A* search algorithm and its properties  
- Greedy best-first search
- Heuristic design for different problems

---

### Navigation
[🏠 Home](index.html) | [← Previous: Week 1](week01.html) | [Next: Week 3 →](week03.html)

---

*"The key insight of search is that complex problems can be solved by systematically exploring the space of possibilities."*