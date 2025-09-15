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

![Search Tree Example](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5d/Graph_search_example.gif/400px-Graph_search_example.gif)

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

![8-Puzzle](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQtIm3pOpw_gcixz8SvQWVWsx6T3ilNiXAyTg&s)

**Problem Formulation:**
- **Initial State:** Current arrangement of tiles
- **Actions:** Move blank space (up, down, left, right)
- **Transition Model:** New arrangement after moving blank
- **Goal Test:** Does current state match target arrangement?
- **Path Cost:** Number of moves (each move costs 1)

### Example: Route Finding

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

![Tree vs Graph Search](https://media.geeksforgeeks.org/wp-content/uploads/20200630123458/tree_vs_graph.jpg)

---

## 🔍 Uninformed Search Strategies

**Uninformed (Blind) Search:** Uses only information available in problem definition
- No domain-specific knowledge about which states are better
- Systematic exploration of the search space

### 1. Breadth-First Search (BFS)

**Strategy:** Expand shallowest unexpanded node first

**Algorithm:**
```
1. Initialize frontier with initial state
2. While frontier is not empty:
   3. Remove shallowest node from frontier
   4. If node is goal, return solution
   5. Expand node and add children to frontier
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

**Algorithm:**
```
1. Initialize frontier as stack with initial state  
2. While frontier is not empty:
   3. Remove deepest node from frontier
   4. If node is goal, return solution
   5. Expand node and add children to frontier
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

### 4. Iterative Deepening Search (IDS)

**Strategy:** Gradually increase depth limit (DLS with limits 0, 1, 2, ...)

**Properties:**
- ✅ **Complete:** Yes
- ✅ **Optimal:** Yes (if step costs equal)  
- ✅ **Time Complexity:** O(b^d)
- ✅ **Space Complexity:** O(bd)

**Why it Works:** Most nodes are at deepest level, so repetition isn't costly

### 5. Uniform-Cost Search (UCS)

**Strategy:** Expand node with lowest path cost first

**Algorithm:**
```
1. Initialize frontier as priority queue with initial state (cost 0)
2. While frontier is not empty:
   3. Remove lowest-cost node from frontier  
   4. If node is goal, return solution
   5. If node not explored:
      6. Mark as explored
      7. Expand node and add children with cumulative costs
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

## 💻 Implementation Considerations

### Data Structures
- **Frontier:** Queue (BFS), Stack (DFS), Priority Queue (UCS)
- **Explored Set:** Hash table for efficient lookup
- **Nodes:** Store state, parent, action, path cost

### Avoiding Repeated States
```python
# Graph search template
def graph_search(problem):
    frontier = initialize_frontier(problem.initial)
    explored = set()
    
    while frontier:
        node = frontier.pop()
        
        if problem.goal_test(node.state):
            return solution(node)
            
        if node.state not in explored:
            explored.add(node.state)
            
            for action in problem.actions(node.state):
                child = child_node(problem, node, action)
                if child.state not in explored:
                    frontier.add(child)
    
    return failure
```

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

---

## 🔍 Looking Ahead

Next week, we'll learn about **informed search** strategies that use domain-specific knowledge (heuristics) to guide the search more efficiently. We'll explore A* search, one of the most important algorithms in AI!

---

*"The key insight of search is that complex problems can be solved by systematically exploring the space of possibilities."*
