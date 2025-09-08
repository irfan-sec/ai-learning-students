#!/usr/bin/env python3
"""
Week 2: Uninformed Search Algorithms Implementation
This module implements the core uninformed search algorithms: BFS, DFS, UCS, and IDS.
"""

from collections import deque
import heapq
from typing import List, Tuple, Dict, Set, Optional, Any
from abc import ABC, abstractmethod


class SearchProblem(ABC):
    """Abstract base class for search problems."""
    
    @abstractmethod
    def get_initial_state(self):
        """Return the initial state of the problem."""
        pass
    
    @abstractmethod
    def is_goal_state(self, state):
        """Return True if state is a goal state."""
        pass
    
    @abstractmethod
    def get_successors(self, state):
        """Return list of (successor_state, action, cost) tuples."""
        pass


class GridSearchProblem(SearchProblem):
    """
    Grid-based pathfinding problem.
    Grid cells: 0 = free, 1 = obstacle
    """
    
    def __init__(self, grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
    
    def get_initial_state(self) -> Tuple[int, int]:
        return self.start
    
    def is_goal_state(self, state: Tuple[int, int]) -> bool:
        return state == self.goal
    
    def get_successors(self, state: Tuple[int, int]) -> List[Tuple[Tuple[int, int], str, int]]:
        """Get valid neighboring cells."""
        successors = []
        row, col = state
        
        # Four directions: up, down, left, right
        directions = [(-1, 0, 'up'), (1, 0, 'down'), (0, -1, 'left'), (0, 1, 'right')]
        
        for dr, dc, action in directions:
            new_row, new_col = row + dr, col + dc
            
            # Check bounds
            if (0 <= new_row < self.rows and 
                0 <= new_col < self.cols and 
                self.grid[new_row][new_col] == 0):
                
                successors.append(((new_row, new_col), action, 1))
        
        return successors


class SearchNode:
    """Represents a node in the search tree."""
    
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
    
    def __lt__(self, other):
        """For priority queue comparison."""
        return self.path_cost < other.path_cost
    
    def get_path(self) -> List[Tuple]:
        """Reconstruct path from initial state to this node."""
        path = []
        node = self
        while node:
            if node.action:
                path.append((node.state, node.action))
            else:
                path.append((node.state, None))  # Initial state
            node = node.parent
        return list(reversed(path))


class SearchResult:
    """Container for search results."""
    
    def __init__(self, solution_path=None, nodes_expanded=0, max_frontier_size=0, 
                 solution_cost=0, success=False):
        self.solution_path = solution_path
        self.nodes_expanded = nodes_expanded
        self.max_frontier_size = max_frontier_size
        self.solution_cost = solution_cost
        self.success = success


def breadth_first_search(problem: SearchProblem) -> SearchResult:
    """
    Breadth-First Search implementation.
    Uses FIFO queue, explores shallowest nodes first.
    """
    # Initialize frontier with initial state
    initial_node = SearchNode(problem.get_initial_state())
    
    if problem.is_goal_state(initial_node.state):
        return SearchResult(
            solution_path=initial_node.get_path(),
            success=True,
            solution_cost=0
        )
    
    frontier = deque([initial_node])  # FIFO queue for BFS
    explored = set()  # Keep track of explored states
    
    nodes_expanded = 0
    max_frontier_size = 1
    
    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        
        # Remove first node from frontier
        node = frontier.popleft()
        
        # Add to explored set
        explored.add(node.state)
        nodes_expanded += 1
        
        # Expand node
        for successor_state, action, cost in problem.get_successors(node.state):
            child = SearchNode(
                state=successor_state,
                parent=node,
                action=action,
                path_cost=node.path_cost + cost
            )
            
            # Check if state hasn't been explored and isn't in frontier
            if (child.state not in explored and 
                not any(n.state == child.state for n in frontier)):
                
                if problem.is_goal_state(child.state):
                    return SearchResult(
                        solution_path=child.get_path(),
                        nodes_expanded=nodes_expanded,
                        max_frontier_size=max_frontier_size,
                        solution_cost=child.path_cost,
                        success=True
                    )
                
                frontier.append(child)
    
    # No solution found
    return SearchResult(
        nodes_expanded=nodes_expanded,
        max_frontier_size=max_frontier_size,
        success=False
    )


def depth_first_search(problem: SearchProblem, max_depth: int = 100) -> SearchResult:
    """
    Depth-First Search implementation.
    Uses LIFO stack, explores deepest nodes first.
    """
    initial_node = SearchNode(problem.get_initial_state())
    
    if problem.is_goal_state(initial_node.state):
        return SearchResult(
            solution_path=initial_node.get_path(),
            success=True,
            solution_cost=0
        )
    
    frontier = [initial_node]  # Stack for DFS
    explored = set()
    
    nodes_expanded = 0
    max_frontier_size = 1
    
    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        
        # Remove last node from frontier (LIFO)
        node = frontier.pop()
        
        # Skip if already explored or too deep
        if node.state in explored or node.path_cost >= max_depth:
            continue
            
        explored.add(node.state)
        nodes_expanded += 1
        
        # Check if goal
        if problem.is_goal_state(node.state):
            return SearchResult(
                solution_path=node.get_path(),
                nodes_expanded=nodes_expanded,
                max_frontier_size=max_frontier_size,
                solution_cost=node.path_cost,
                success=True
            )
        
        # Expand node (add in reverse order for consistent behavior)
        successors = problem.get_successors(node.state)
        for successor_state, action, cost in reversed(successors):
            if successor_state not in explored:
                child = SearchNode(
                    state=successor_state,
                    parent=node,
                    action=action,
                    path_cost=node.path_cost + cost
                )
                frontier.append(child)
    
    return SearchResult(
        nodes_expanded=nodes_expanded,
        max_frontier_size=max_frontier_size,
        success=False
    )


def uniform_cost_search(problem: SearchProblem) -> SearchResult:
    """
    Uniform-Cost Search implementation.
    Uses priority queue ordered by path cost.
    """
    initial_node = SearchNode(problem.get_initial_state())
    frontier = [initial_node]  # Priority queue
    explored = set()
    
    nodes_expanded = 0
    max_frontier_size = 1
    
    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        
        # Remove lowest-cost node
        node = heapq.heappop(frontier)
        
        if problem.is_goal_state(node.state):
            return SearchResult(
                solution_path=node.get_path(),
                nodes_expanded=nodes_expanded,
                max_frontier_size=max_frontier_size,
                solution_cost=node.path_cost,
                success=True
            )
        
        if node.state not in explored:
            explored.add(node.state)
            nodes_expanded += 1
            
            for successor_state, action, cost in problem.get_successors(node.state):
                child = SearchNode(
                    state=successor_state,
                    parent=node,
                    action=action,
                    path_cost=node.path_cost + cost
                )
                heapq.heappush(frontier, child)
    
    return SearchResult(
        nodes_expanded=nodes_expanded,
        max_frontier_size=max_frontier_size,
        success=False
    )


def iterative_deepening_search(problem: SearchProblem, max_depth: int = 50) -> SearchResult:
    """
    Iterative Deepening Search implementation.
    Combines benefits of DFS and BFS.
    """
    total_nodes_expanded = 0
    max_frontier_size = 0
    
    for depth_limit in range(max_depth + 1):
        result = depth_limited_search(problem, depth_limit)
        total_nodes_expanded += result.nodes_expanded
        max_frontier_size = max(max_frontier_size, result.max_frontier_size)
        
        if result.success:
            return SearchResult(
                solution_path=result.solution_path,
                nodes_expanded=total_nodes_expanded,
                max_frontier_size=max_frontier_size,
                solution_cost=result.solution_cost,
                success=True
            )
    
    return SearchResult(
        nodes_expanded=total_nodes_expanded,
        max_frontier_size=max_frontier_size,
        success=False
    )


def depth_limited_search(problem: SearchProblem, limit: int) -> SearchResult:
    """Helper function for iterative deepening."""
    initial_node = SearchNode(problem.get_initial_state())
    frontier = [initial_node]
    
    nodes_expanded = 0
    max_frontier_size = 1
    
    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        node = frontier.pop()
        
        if problem.is_goal_state(node.state):
            return SearchResult(
                solution_path=node.get_path(),
                nodes_expanded=nodes_expanded,
                max_frontier_size=max_frontier_size,
                solution_cost=node.path_cost,
                success=True
            )
        
        if node.path_cost < limit:
            nodes_expanded += 1
            successors = problem.get_successors(node.state)
            for successor_state, action, cost in reversed(successors):
                child = SearchNode(
                    state=successor_state,
                    parent=node,
                    action=action,
                    path_cost=node.path_cost + cost
                )
                frontier.append(child)
    
    return SearchResult(
        nodes_expanded=nodes_expanded,
        max_frontier_size=max_frontier_size,
        success=False
    )


def visualize_grid_solution(problem: GridSearchProblem, result: SearchResult):
    """Visualize the solution path on the grid."""
    if not result.success:
        print("No solution found!")
        return
    
    # Create a copy of the grid for visualization
    visual_grid = [row[:] for row in problem.grid]
    
    # Mark the path
    path_positions = [state for state, action in result.solution_path]
    for i, (row, col) in enumerate(path_positions):
        if (row, col) == problem.start:
            visual_grid[row][col] = 'S'  # Start
        elif (row, col) == problem.goal:
            visual_grid[row][col] = 'G'  # Goal
        else:
            visual_grid[row][col] = '.'  # Path
    
    # Print the grid
    print("\nSolution Path Visualization:")
    print("S=Start, G=Goal, .=Path, 1=Obstacle, 0=Free")
    for row in visual_grid:
        print(' '.join(str(cell) for cell in row))
    
    print(f"\nPath length: {len(result.solution_path)}")
    print(f"Solution cost: {result.solution_cost}")


def compare_algorithms():
    """Compare all search algorithms on the same problem."""
    print("=" * 60)
    print("UNINFORMED SEARCH ALGORITHMS COMPARISON")
    print("=" * 60)
    
    # Create test problem: 5x5 grid with obstacles
    grid = [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1],
        [1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0]
    ]
    
    start = (0, 0)
    goal = (4, 4)
    problem = GridSearchProblem(grid, start, goal)
    
    print(f"Problem: Find path from {start} to {goal}")
    print("Grid (0=free, 1=obstacle):")
    for row in grid:
        print(' '.join(str(cell) for cell in row))
    print()
    
    # Test all algorithms
    algorithms = [
        ("Breadth-First Search", breadth_first_search),
        ("Depth-First Search", depth_first_search),
        ("Uniform-Cost Search", uniform_cost_search),
        ("Iterative Deepening", iterative_deepening_search)
    ]
    
    results = {}
    
    for name, algorithm in algorithms:
        print(f"Running {name}...")
        result = algorithm(problem)
        results[name] = result
        
        print(f"  Success: {result.success}")
        if result.success:
            print(f"  Path length: {len(result.solution_path)}")
            print(f"  Solution cost: {result.solution_cost}")
        print(f"  Nodes expanded: {result.nodes_expanded}")
        print(f"  Max frontier size: {result.max_frontier_size}")
        print()
    
    # Show solution path for BFS (optimal)
    if results["Breadth-First Search"].success:
        visualize_grid_solution(problem, results["Breadth-First Search"])
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("ALGORITHM COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Algorithm':<20} {'Success':<8} {'Cost':<6} {'Nodes':<8} {'Frontier':<10}")
    print("-" * 60)
    
    for name in results:
        result = results[name]
        cost = result.solution_cost if result.success else "N/A"
        print(f"{name:<20} {result.success:<8} {cost:<6} {result.nodes_expanded:<8} {result.max_frontier_size:<10}")


def demonstrate_algorithm_properties():
    """Demonstrate key properties of different algorithms."""
    print("\n" + "=" * 60)
    print("ALGORITHM PROPERTIES DEMONSTRATION")
    print("=" * 60)
    
    # Problem with multiple solution paths
    grid = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]
    
    start = (0, 0)
    goal = (2, 2)
    problem = GridSearchProblem(grid, start, goal)
    
    print("Simple 3x3 grid with one obstacle:")
    for row in grid:
        print(' '.join(str(cell) for cell in row))
    print(f"\nStart: {start}, Goal: {goal}")
    
    # Show different paths found
    bfs_result = breadth_first_search(problem)
    dfs_result = depth_first_search(problem)
    
    print(f"\nBFS path length: {len(bfs_result.solution_path) if bfs_result.success else 'No solution'}")
    print(f"DFS path length: {len(dfs_result.solution_path) if dfs_result.success else 'No solution'}")
    
    if bfs_result.success and dfs_result.success:
        bfs_path = [state for state, _ in bfs_result.solution_path]
        dfs_path = [state for state, _ in dfs_result.solution_path]
        
        print(f"\nBFS path: {bfs_path}")
        print(f"DFS path: {dfs_path}")
        print(f"\nBFS finds {'optimal' if len(bfs_path) <= len(dfs_path) else 'suboptimal'} solution")
        print(f"DFS finds {'optimal' if len(dfs_path) <= len(bfs_path) else 'suboptimal'} solution")


if __name__ == "__main__":
    print("Week 2: Uninformed Search Algorithms")
    print("This demonstration shows BFS, DFS, UCS, and IDS in action.")
    
    compare_algorithms()
    demonstrate_algorithm_properties()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS:")
    print("1. BFS guarantees optimal solutions for unweighted graphs")
    print("2. DFS uses less memory but may find suboptimal solutions") 
    print("3. UCS handles weighted graphs optimally")
    print("4. IDS combines BFS optimality with DFS memory efficiency")
    print("5. Algorithm choice depends on problem constraints")
    print("=" * 60)