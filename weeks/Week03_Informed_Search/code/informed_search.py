#!/usr/bin/env python3
"""
Week 3: Informed Search Algorithms Implementation
This module implements A* search and demonstrates different heuristic functions.
"""

import heapq
import math
from typing import List, Tuple, Dict, Set, Optional, Callable
from dataclasses import dataclass, field
import time


@dataclass
class SearchNode:
    """Node in the search tree with f, g, h costs for A* search."""
    state: Tuple
    parent: Optional['SearchNode'] = None
    action: Optional[str] = None
    g_cost: float = 0  # Cost from start to this node
    h_cost: float = 0  # Heuristic cost from this node to goal
    f_cost: float = field(init=False)  # Total estimated cost: f = g + h
    
    def __post_init__(self):
        self.f_cost = self.g_cost + self.h_cost
    
    def __lt__(self, other):
        """For priority queue comparison - prioritize by f_cost, then h_cost."""
        if self.f_cost == other.f_cost:
            return self.h_cost < other.h_cost
        return self.f_cost < other.f_cost
    
    def get_path(self) -> List[Tuple]:
        """Reconstruct path from start to this node."""
        path = []
        node = self
        while node:
            path.append((node.state, node.action))
            node = node.parent
        return list(reversed(path))


class GridWorldProblem:
    """Grid-based pathfinding problem for demonstrating informed search."""
    
    def __init__(self, grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
    
    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within bounds and not an obstacle."""
        row, col = pos
        return (0 <= row < self.rows and 
                0 <= col < self.cols and 
                self.grid[row][col] == 0)
    
    def get_successors(self, state: Tuple[int, int]) -> List[Tuple[Tuple[int, int], str, float]]:
        """Get valid neighboring cells with their costs."""
        successors = []
        row, col = state
        
        # 8-directional movement (including diagonals)
        directions = [
            (-1, 0, 'up', 1.0),
            (1, 0, 'down', 1.0), 
            (0, -1, 'left', 1.0),
            (0, 1, 'right', 1.0),
            (-1, -1, 'up-left', math.sqrt(2)),
            (-1, 1, 'up-right', math.sqrt(2)),
            (1, -1, 'down-left', math.sqrt(2)),
            (1, 1, 'down-right', math.sqrt(2))
        ]
        
        for dr, dc, action, cost in directions:
            new_pos = (row + dr, col + dc)
            if self.is_valid_position(new_pos):
                successors.append((new_pos, action, cost))
        
        return successors


# Heuristic Functions for Grid World
def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """Manhattan (L1) distance heuristic."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def euclidean_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """Euclidean (L2) distance heuristic."""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def chebyshev_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """Chebyshev (L∞) distance heuristic."""
    return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))


def zero_heuristic(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """Zero heuristic - turns A* into Dijkstra's algorithm."""
    return 0


# 8-Puzzle Problem and Heuristics
class EightPuzzleProblem:
    """8-puzzle sliding tile problem."""
    
    def __init__(self, initial_state: Tuple[int, ...], goal_state: Tuple[int, ...]):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.size = 3  # 3x3 puzzle
    
    def find_blank(self, state: Tuple[int, ...]) -> int:
        """Find position of blank tile (0)."""
        return state.index(0)
    
    def get_successors(self, state: Tuple[int, ...]) -> List[Tuple[Tuple[int, ...], str, float]]:
        """Get all valid moves from current state."""
        successors = []
        blank_pos = self.find_blank(state)
        blank_row, blank_col = divmod(blank_pos, self.size)
        
        # Possible moves: up, down, left, right
        moves = [
            (-1, 0, 'up'),
            (1, 0, 'down'),
            (0, -1, 'left'),
            (0, 1, 'right')
        ]
        
        for dr, dc, move_name in moves:
            new_row = blank_row + dr
            new_col = blank_col + dc
            
            if 0 <= new_row < self.size and 0 <= new_col < self.size:
                new_blank_pos = new_row * self.size + new_col
                
                # Swap blank with tile
                new_state = list(state)
                new_state[blank_pos], new_state[new_blank_pos] = new_state[new_blank_pos], new_state[blank_pos]
                
                successors.append((tuple(new_state), move_name, 1))
        
        return successors


def misplaced_tiles_heuristic(state: Tuple[int, ...], goal: Tuple[int, ...]) -> float:
    """Count number of misplaced tiles (excluding blank)."""
    return sum(1 for i, (s, g) in enumerate(zip(state, goal)) 
               if s != 0 and s != g)


def manhattan_puzzle_heuristic(state: Tuple[int, ...], goal: Tuple[int, ...]) -> float:
    """Manhattan distance sum for all tiles to their goal positions."""
    size = 3  # 3x3 puzzle
    distance = 0
    
    for i, tile in enumerate(state):
        if tile != 0:  # Skip blank tile
            current_row, current_col = divmod(i, size)
            goal_pos = goal.index(tile)
            goal_row, goal_col = divmod(goal_pos, size)
            distance += abs(current_row - goal_row) + abs(current_col - goal_col)
    
    return distance


def a_star_search(problem, heuristic_fn: Callable, start_state, goal_state) -> Dict:
    """
    A* search algorithm implementation.
    
    Returns dictionary with solution path, costs, and statistics.
    """
    start_time = time.time()
    
    # Initialize
    start_node = SearchNode(
        state=start_state,
        g_cost=0,
        h_cost=heuristic_fn(start_state, goal_state)
    )
    
    frontier = [start_node]
    explored = set()
    nodes_expanded = 0
    max_frontier_size = 1
    
    # For tracking nodes in frontier with their g_costs
    frontier_states = {start_state: 0}
    
    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        
        # Get node with lowest f_cost
        current = heapq.heappop(frontier)
        
        # Remove from frontier tracking
        if current.state in frontier_states:
            del frontier_states[current.state]
        
        # Goal test
        if current.state == goal_state:
            end_time = time.time()
            return {
                'success': True,
                'path': current.get_path(),
                'cost': current.g_cost,
                'nodes_expanded': nodes_expanded,
                'max_frontier_size': max_frontier_size,
                'time': end_time - start_time
            }
        
        # Add to explored
        explored.add(current.state)
        nodes_expanded += 1
        
        # Expand current node
        for successor_state, action, step_cost in problem.get_successors(current.state):
            # Skip if already explored
            if successor_state in explored:
                continue
            
            g_cost = current.g_cost + step_cost
            h_cost = heuristic_fn(successor_state, goal_state)
            
            # If successor is in frontier with higher g_cost, skip
            if successor_state in frontier_states:
                if frontier_states[successor_state] <= g_cost:
                    continue
            
            # Create successor node
            successor = SearchNode(
                state=successor_state,
                parent=current,
                action=action,
                g_cost=g_cost,
                h_cost=h_cost
            )
            
            # Add to frontier
            heapq.heappush(frontier, successor)
            frontier_states[successor_state] = g_cost
    
    # No solution found
    end_time = time.time()
    return {
        'success': False,
        'path': None,
        'cost': float('inf'),
        'nodes_expanded': nodes_expanded,
        'max_frontier_size': max_frontier_size,
        'time': end_time - start_time
    }


def greedy_best_first_search(problem, heuristic_fn: Callable, start_state, goal_state) -> Dict:
    """
    Greedy Best-First Search - only considers heuristic cost.
    """
    start_time = time.time()
    
    start_node = SearchNode(
        state=start_state,
        g_cost=0,
        h_cost=heuristic_fn(start_state, goal_state)
    )
    # Override f_cost to be just h_cost for greedy search
    start_node.f_cost = start_node.h_cost
    
    frontier = [start_node]
    explored = set()
    nodes_expanded = 0
    max_frontier_size = 1
    
    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        
        current = heapq.heappop(frontier)
        
        if current.state == goal_state:
            end_time = time.time()
            return {
                'success': True,
                'path': current.get_path(),
                'cost': current.g_cost,
                'nodes_expanded': nodes_expanded,
                'max_frontier_size': max_frontier_size,
                'time': end_time - start_time
            }
        
        if current.state in explored:
            continue
            
        explored.add(current.state)
        nodes_expanded += 1
        
        for successor_state, action, step_cost in problem.get_successors(current.state):
            if successor_state not in explored:
                successor = SearchNode(
                    state=successor_state,
                    parent=current,
                    action=action,
                    g_cost=current.g_cost + step_cost,
                    h_cost=heuristic_fn(successor_state, goal_state)
                )
                # Use only heuristic for greedy search
                successor.f_cost = successor.h_cost
                
                heapq.heappush(frontier, successor)
    
    end_time = time.time()
    return {
        'success': False,
        'path': None,
        'cost': float('inf'),
        'nodes_expanded': nodes_expanded,
        'max_frontier_size': max_frontier_size,
        'time': end_time - start_time
    }


def visualize_grid_path(problem: GridWorldProblem, path: List[Tuple]):
    """Visualize the solution path on the grid."""
    if not path:
        print("No solution path to visualize!")
        return
    
    # Create visualization grid
    vis_grid = [['1' if cell == 1 else '.' for cell in row] for row in problem.grid]
    
    # Mark path
    for i, (state, action) in enumerate(path):
        row, col = state
        if (row, col) == problem.start:
            vis_grid[row][col] = 'S'
        elif (row, col) == problem.goal:
            vis_grid[row][col] = 'G'
        else:
            vis_grid[row][col] = str(i % 10)  # Path step number
    
    print("\nPath Visualization:")
    print("S=Start, G=Goal, 1=Obstacle, .=Free, Numbers=Path")
    for row in vis_grid:
        print(' '.join(row))


def compare_heuristics_grid():
    """Compare different heuristics on a grid pathfinding problem."""
    print("=" * 70)
    print("INFORMED SEARCH: HEURISTIC COMPARISON ON GRID WORLD")
    print("=" * 70)
    
    # Create test grid with obstacles
    grid = [
        [0, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ]
    
    start = (0, 0)
    goal = (4, 6)
    problem = GridWorldProblem(grid, start, goal)
    
    print(f"Grid size: {len(grid)}x{len(grid[0])}")
    print(f"Start: {start}, Goal: {goal}")
    print("\nGrid (0=free, 1=obstacle):")
    for row in grid:
        print(' '.join(str(cell) for cell in row))
    
    # Test different heuristics with A*
    heuristics = [
        ("Zero (Dijkstra)", zero_heuristic),
        ("Manhattan", manhattan_distance),
        ("Euclidean", euclidean_distance),
        ("Chebyshev", chebyshev_distance)
    ]
    
    results = {}
    
    print(f"\n{'Heuristic':<15} {'Success':<8} {'Cost':<8} {'Nodes':<8} {'Frontier':<10} {'Time':<8}")
    print("-" * 70)
    
    for name, heuristic_fn in heuristics:
        result = a_star_search(problem, heuristic_fn, start, goal)
        results[name] = result
        
        cost = f"{result['cost']:.1f}" if result['success'] else "N/A"
        time_str = f"{result['time']*1000:.1f}ms"
        
        print(f"{name:<15} {result['success']:<8} {cost:<8} "
              f"{result['nodes_expanded']:<8} {result['max_frontier_size']:<10} {time_str:<8}")
    
    # Show optimal path
    if results["Euclidean"]['success']:
        visualize_grid_path(problem, results["Euclidean"]['path'])
    
    return results


def compare_puzzle_heuristics():
    """Compare heuristics on 8-puzzle problem."""
    print("\n" + "=" * 70)
    print("INFORMED SEARCH: 8-PUZZLE HEURISTIC COMPARISON")
    print("=" * 70)
    
    # Easy puzzle instance
    initial = (1, 2, 3, 4, 0, 6, 7, 5, 8)  # One move from solution
    goal = (1, 2, 3, 4, 5, 6, 7, 8, 0)
    
    print("Initial state:    Goal state:")
    print("1 2 3             1 2 3")
    print("4 _ 6             4 5 6") 
    print("7 5 8             7 8 _")
    
    problem = EightPuzzleProblem(initial, goal)
    
    heuristics = [
        ("Zero", lambda s, g: 0),
        ("Misplaced Tiles", misplaced_tiles_heuristic),
        ("Manhattan", manhattan_puzzle_heuristic)
    ]
    
    print(f"\n{'Heuristic':<20} {'Success':<8} {'Cost':<6} {'Nodes':<8} {'Time':<10}")
    print("-" * 60)
    
    for name, heuristic_fn in heuristics:
        result = a_star_search(problem, heuristic_fn, initial, goal)
        time_str = f"{result['time']*1000:.1f}ms"
        cost = result['cost'] if result['success'] else "N/A"
        
        print(f"{name:<20} {result['success']:<8} {cost:<6} "
              f"{result['nodes_expanded']:<8} {time_str:<10}")


def demonstrate_greedy_vs_astar():
    """Compare Greedy Best-First vs A* search."""
    print("\n" + "=" * 70)
    print("GREEDY BEST-FIRST vs A* COMPARISON")
    print("=" * 70)
    
    # Grid where greedy might make suboptimal choice
    grid = [
        [0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ]
    
    start = (0, 0)
    goal = (0, 4)
    problem = GridWorldProblem(grid, start, goal)
    
    print("Test Grid:")
    for i, row in enumerate(grid):
        row_str = ' '.join(str(cell) for cell in row)
        if i == 0:
            row_str += "  (Start at (0,0), Goal at (0,4))"
        print(row_str)
    
    algorithms = [
        ("Greedy Best-First", greedy_best_first_search),
        ("A* Search", a_star_search)
    ]
    
    print(f"\n{'Algorithm':<20} {'Success':<8} {'Cost':<8} {'Nodes':<8} {'Optimal?':<10}")
    print("-" * 65)
    
    results = {}
    for name, algorithm in algorithms:
        result = algorithm(problem, euclidean_distance, start, goal)
        results[name] = result
        
        cost = f"{result['cost']:.1f}" if result['success'] else "N/A"
        optimal = "Yes" if result['success'] and result['cost'] <= 4.0 else "No"
        
        print(f"{name:<20} {result['success']:<8} {cost:<8} "
              f"{result['nodes_expanded']:<8} {optimal:<10}")
    
    # Show paths
    print("\nPath Comparison:")
    for name in results:
        result = results[name]
        if result['success']:
            path_states = [state for state, _ in result['path']]
            print(f"{name}: {path_states}")


if __name__ == "__main__":
    print("Week 3: Informed Search Algorithms")
    print("Demonstrating A*, Greedy Best-First, and heuristic functions")
    
    # Run demonstrations
    compare_heuristics_grid()
    compare_puzzle_heuristics() 
    demonstrate_greedy_vs_astar()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS:")
    print("1. Better heuristics lead to fewer node expansions")
    print("2. A* guarantees optimal solutions with admissible heuristics")
    print("3. Greedy search is faster but may find suboptimal solutions")
    print("4. Manhattan distance dominates misplaced tiles for puzzles")
    print("5. Euclidean distance works well for grid pathfinding")
    print("=" * 70)