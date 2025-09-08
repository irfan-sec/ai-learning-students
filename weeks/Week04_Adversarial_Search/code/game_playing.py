#!/usr/bin/env python3
"""
Week 4: Adversarial Search - Game Playing Algorithms
This module implements Minimax and Alpha-Beta pruning for game playing.
"""

from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
import random
import time
import copy


class GameState(ABC):
    """Abstract base class for game states."""
    
    @abstractmethod
    def get_legal_moves(self) -> List[Any]:
        """Return list of legal moves from this state."""
        pass
    
    @abstractmethod
    def make_move(self, move: Any) -> 'GameState':
        """Return new state after making the given move."""
        pass
    
    @abstractmethod
    def is_terminal(self) -> bool:
        """Return True if this is a terminal state."""
        pass
    
    @abstractmethod
    def evaluate(self) -> float:
        """Return evaluation score from current player's perspective."""
        pass
    
    @abstractmethod
    def get_current_player(self) -> str:
        """Return current player ('X' or 'O').""" 
        pass


class TicTacToeState(GameState):
    """Tic-tac-toe game state implementation."""
    
    def __init__(self, board: List[List[str]] = None, current_player: str = 'X'):
        self.board = board if board else [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = current_player
        self.size = 3
    
    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """Return list of (row, col) positions that are empty."""
        moves = []
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] == ' ':
                    moves.append((row, col))
        return moves
    
    def make_move(self, move: Tuple[int, int]) -> 'TicTacToeState':
        """Return new state after placing current player's mark."""
        row, col = move
        new_board = copy.deepcopy(self.board)
        new_board[row][col] = self.current_player
        
        next_player = 'O' if self.current_player == 'X' else 'X'
        return TicTacToeState(new_board, next_player)
    
    def is_terminal(self) -> bool:
        """Check if game is over (win or draw)."""
        return self.get_winner() is not None or len(self.get_legal_moves()) == 0
    
    def get_winner(self) -> Optional[str]:
        """Return winning player or None if no winner yet."""
        # Check rows
        for row in self.board:
            if row[0] == row[1] == row[2] != ' ':
                return row[0]
        
        # Check columns
        for col in range(self.size):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] != ' ':
                return self.board[0][col]
        
        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ':
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
            return self.board[0][2]
        
        return None
    
    def evaluate(self) -> float:
        """Evaluate position from current player's perspective."""
        winner = self.get_winner()
        
        if winner == self.current_player:
            return 1.0  # Current player wins
        elif winner is not None:
            return -1.0  # Current player loses
        else:
            return 0.0  # Draw or game continues
    
    def get_current_player(self) -> str:
        return self.current_player
    
    def display(self):
        """Print the current board state."""
        print("Current board:")
        for i, row in enumerate(self.board):
            print(" | ".join(row))
            if i < len(self.board) - 1:
                print("-" * 9)
        print(f"Current player: {self.current_player}")
        print()


class Connect4State(GameState):
    """Connect Four game state implementation."""
    
    def __init__(self, board: List[List[str]] = None, current_player: str = 'X'):
        self.rows = 6
        self.cols = 7
        self.board = board if board else [[' ' for _ in range(self.cols)] for _ in range(self.rows)]
        self.current_player = current_player
    
    def get_legal_moves(self) -> List[int]:
        """Return list of columns where a piece can be dropped."""
        moves = []
        for col in range(self.cols):
            if self.board[0][col] == ' ':  # Top row is empty
                moves.append(col)
        return moves
    
    def make_move(self, move: int) -> 'Connect4State':
        """Drop a piece in the specified column."""
        col = move
        new_board = copy.deepcopy(self.board)
        
        # Find lowest empty row in this column
        for row in range(self.rows - 1, -1, -1):
            if new_board[row][col] == ' ':
                new_board[row][col] = self.current_player
                break
        
        next_player = 'O' if self.current_player == 'X' else 'X'
        return Connect4State(new_board, next_player)
    
    def is_terminal(self) -> bool:
        """Check if game is over."""
        return self.get_winner() is not None or len(self.get_legal_moves()) == 0
    
    def get_winner(self) -> Optional[str]:
        """Check for four in a row horizontally, vertically, or diagonally."""
        # Check horizontal
        for row in range(self.rows):
            for col in range(self.cols - 3):
                if (self.board[row][col] != ' ' and
                    self.board[row][col] == self.board[row][col+1] == 
                    self.board[row][col+2] == self.board[row][col+3]):
                    return self.board[row][col]
        
        # Check vertical
        for row in range(self.rows - 3):
            for col in range(self.cols):
                if (self.board[row][col] != ' ' and
                    self.board[row][col] == self.board[row+1][col] == 
                    self.board[row+2][col] == self.board[row+3][col]):
                    return self.board[row][col]
        
        # Check diagonal (positive slope)
        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                if (self.board[row][col] != ' ' and
                    self.board[row][col] == self.board[row+1][col+1] == 
                    self.board[row+2][col+2] == self.board[row+3][col+3]):
                    return self.board[row][col]
        
        # Check diagonal (negative slope)
        for row in range(3, self.rows):
            for col in range(self.cols - 3):
                if (self.board[row][col] != ' ' and
                    self.board[row][col] == self.board[row-1][col+1] == 
                    self.board[row-2][col+2] == self.board[row-3][col+3]):
                    return self.board[row][col]
        
        return None
    
    def evaluate(self) -> float:
        """Simple evaluation function for Connect Four."""
        winner = self.get_winner()
        if winner == self.current_player:
            return 1.0
        elif winner is not None:
            return -1.0
        
        # Heuristic: prefer center columns
        score = 0.0
        center_col = self.cols // 2
        
        for row in range(self.rows):
            if self.board[row][center_col] == self.current_player:
                score += 0.1
            elif self.board[row][center_col] != ' ':
                score -= 0.1
        
        return score
    
    def get_current_player(self) -> str:
        return self.current_player
    
    def display(self):
        """Print the current board."""
        print("Connect Four board:")
        for row in self.board:
            print("|" + "|".join(f" {cell} " for cell in row) + "|")
        print("+" + "+".join("---" for _ in range(self.cols)) + "+")
        print(" " + "   ".join(str(i) for i in range(self.cols)))
        print(f"Current player: {self.current_player}")
        print()


class GameResult:
    """Container for game playing results and statistics."""
    
    def __init__(self):
        self.move = None
        self.value = None
        self.nodes_evaluated = 0
        self.max_depth_reached = 0
        self.time_taken = 0.0
        self.prunings = 0


def minimax(state: GameState, depth: int, maximizing_player: bool, 
            result: GameResult = None) -> float:
    """
    Minimax algorithm for perfect game play.
    
    Args:
        state: Current game state
        depth: Maximum depth to search
        maximizing_player: True if current player is maximizing
        result: Object to store search statistics
        
    Returns:
        Best value achievable from this position
    """
    if result:
        result.nodes_evaluated += 1
        result.max_depth_reached = max(result.max_depth_reached, 
                                      result.max_depth_reached - depth)
    
    # Terminal test
    if depth == 0 or state.is_terminal():
        return state.evaluate()
    
    if maximizing_player:
        max_eval = float('-inf')
        for move in state.get_legal_moves():
            child_state = state.make_move(move)
            eval_score = minimax(child_state, depth - 1, False, result)
            max_eval = max(max_eval, eval_score)
        return max_eval
    else:
        min_eval = float('inf')
        for move in state.get_legal_moves():
            child_state = state.make_move(move)
            eval_score = minimax(child_state, depth - 1, True, result)
            min_eval = min(min_eval, eval_score)
        return min_eval


def alpha_beta(state: GameState, depth: int, alpha: float, beta: float, 
               maximizing_player: bool, result: GameResult = None) -> float:
    """
    Alpha-beta pruning algorithm for efficient game tree search.
    
    Args:
        state: Current game state
        depth: Maximum depth to search
        alpha: Best value that maximizer can guarantee
        beta: Best value that minimizer can guarantee
        maximizing_player: True if current player is maximizing
        result: Object to store search statistics
        
    Returns:
        Best value achievable from this position
    """
    if result:
        result.nodes_evaluated += 1
    
    # Terminal test
    if depth == 0 or state.is_terminal():
        return state.evaluate()
    
    if maximizing_player:
        max_eval = float('-inf')
        for move in state.get_legal_moves():
            child_state = state.make_move(move)
            eval_score = alpha_beta(child_state, depth - 1, alpha, beta, False, result)
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            
            if beta <= alpha:
                if result:
                    result.prunings += 1
                break  # Beta cutoff
                
        return max_eval
    else:
        min_eval = float('inf')
        for move in state.get_legal_moves():
            child_state = state.make_move(move)
            eval_score = alpha_beta(child_state, depth - 1, alpha, beta, True, result)
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            
            if beta <= alpha:
                if result:
                    result.prunings += 1
                break  # Alpha cutoff
                
        return min_eval


def get_best_move(state: GameState, depth: int, use_alpha_beta: bool = True) -> Tuple[Any, GameResult]:
    """
    Find the best move using minimax or alpha-beta pruning.
    
    Returns:
        Tuple of (best_move, search_statistics)
    """
    result = GameResult()
    start_time = time.time()
    
    best_move = None
    best_value = float('-inf')
    
    for move in state.get_legal_moves():
        child_state = state.make_move(move)
        
        if use_alpha_beta:
            value = alpha_beta(child_state, depth - 1, float('-inf'), float('inf'), False, result)
        else:
            value = minimax(child_state, depth - 1, False, result)
        
        if value > best_value:
            best_value = value
            best_move = move
    
    result.move = best_move
    result.value = best_value
    result.time_taken = time.time() - start_time
    
    return best_move, result


def play_game(game_class, player1_depth: int = 4, player2_depth: int = 4, 
              display: bool = True) -> Dict[str, Any]:
    """
    Play a complete game between two AI players.
    
    Returns:
        Dictionary with game results and statistics
    """
    state = game_class()
    move_count = 0
    total_time = {'X': 0, 'O': 0}
    total_nodes = {'X': 0, 'O': 0}
    
    if display:
        print(f"Starting new {game_class.__name__} game")
        print("=" * 40)
        state.display()
    
    while not state.is_terminal():
        current_player = state.get_current_player()
        depth = player1_depth if current_player == 'X' else player2_depth
        
        if display:
            print(f"Player {current_player}'s turn (depth {depth})")
        
        move, result = get_best_move(state, depth)
        
        total_time[current_player] += result.time_taken
        total_nodes[current_player] += result.nodes_evaluated
        
        if display:
            print(f"Chosen move: {move}")
            print(f"Evaluation: {result.value:.3f}")
            print(f"Nodes evaluated: {result.nodes_evaluated}")
            print(f"Time: {result.time_taken:.3f}s")
            print()
        
        state = state.make_move(move)
        move_count += 1
        
        if display:
            state.display()
    
    winner = state.get_winner()
    
    if display:
        if winner:
            print(f"Game over! Winner: {winner}")
        else:
            print("Game over! It's a draw!")
        
        print("\nGame Statistics:")
        print(f"Total moves: {move_count}")
        print(f"Player X - Time: {total_time['X']:.3f}s, Nodes: {total_nodes['X']}")
        print(f"Player O - Time: {total_time['O']:.3f}s, Nodes: {total_nodes['O']}")
    
    return {
        'winner': winner,
        'moves': move_count,
        'time': total_time,
        'nodes': total_nodes
    }


def compare_algorithms():
    """Compare Minimax vs Alpha-Beta pruning performance."""
    print("=" * 60)
    print("MINIMAX vs ALPHA-BETA COMPARISON")
    print("=" * 60)
    
    # Create test position
    state = TicTacToeState()
    state = state.make_move((1, 1))  # X plays center
    state = state.make_move((0, 0))  # O plays corner
    
    print("Test position:")
    state.display()
    
    depths = [3, 4, 5]
    
    print(f"{'Depth':<6} {'Algorithm':<12} {'Nodes':<8} {'Prunings':<10} {'Time':<8} {'Best Move':<10}")
    print("-" * 60)
    
    for depth in depths:
        # Test Minimax
        result_minimax = GameResult()
        start_time = time.time()
        
        best_move = None
        best_value = float('-inf')
        
        for move in state.get_legal_moves():
            child_state = state.make_move(move)
            value = minimax(child_state, depth - 1, False, result_minimax)
            if value > best_value:
                best_value = value
                best_move = move
        
        result_minimax.time_taken = time.time() - start_time
        
        print(f"{depth:<6} {'Minimax':<12} {result_minimax.nodes_evaluated:<8} {'-':<10} "
              f"{result_minimax.time_taken:.3f}s{'':<3} {best_move}")
        
        # Test Alpha-Beta
        result_ab = GameResult()
        start_time = time.time()
        
        best_move = None
        best_value = float('-inf')
        
        for move in state.get_legal_moves():
            child_state = state.make_move(move)
            value = alpha_beta(child_state, depth - 1, float('-inf'), float('inf'), False, result_ab)
            if value > best_value:
                best_value = value
                best_move = move
        
        result_ab.time_taken = time.time() - start_time
        
        print(f"{depth:<6} {'Alpha-Beta':<12} {result_ab.nodes_evaluated:<8} {result_ab.prunings:<10} "
              f"{result_ab.time_taken:.3f}s{'':<3} {best_move}")
        
        # Calculate improvement
        if result_minimax.nodes_evaluated > 0:
            reduction = 100 * (1 - result_ab.nodes_evaluated / result_minimax.nodes_evaluated)
            print(f"{'':>18} Node reduction: {reduction:.1f}%")
        
        print()


def demonstrate_games():
    """Demonstrate AI playing different games."""
    print("=" * 60)
    print("AI GAME PLAYING DEMONSTRATION")
    print("=" * 60)
    
    # Play Tic-Tac-Toe
    print("\n1. TIC-TAC-TOE GAME")
    print("-" * 30)
    result = play_game(TicTacToeState, player1_depth=6, player2_depth=6, display=True)
    
    print("\n2. CONNECT FOUR GAME (first few moves)")
    print("-" * 40)
    # Play just a few moves of Connect Four to show the concept
    state = Connect4State()
    for i in range(6):  # Play 6 moves
        current_player = state.get_current_player()
        print(f"Move {i+1}: Player {current_player}")
        
        move, result = get_best_move(state, depth=3)
        print(f"Chosen column: {move} (evaluation: {result.value:.3f})")
        
        state = state.make_move(move)
        state.display()
        
        if state.is_terminal():
            winner = state.get_winner()
            print(f"Game over! Winner: {winner}" if winner else "Draw!")
            break


if __name__ == "__main__":
    print("Week 4: Adversarial Search - Game Playing")
    print("Demonstrating Minimax and Alpha-Beta pruning algorithms")
    
    # Run demonstrations
    compare_algorithms()
    demonstrate_games()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS:")
    print("1. Minimax finds optimal moves assuming perfect opponent")
    print("2. Alpha-beta pruning dramatically reduces search space")
    print("3. Move ordering affects pruning efficiency")
    print("4. Evaluation functions are crucial for deep search")
    print("5. Game complexity determines feasible search depth")
    print("=" * 60)