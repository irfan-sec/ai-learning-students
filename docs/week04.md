---
layout: week
title: "Week 4: Adversarial Search (Game Playing)"
permalink: /week04.html
---

### Navigation
[🏠 Home](index.html) | [← Previous: Week 3](week03.html) | [Next: Week 5 →](week05.html)

---

# Week 4: Adversarial Search (Game Playing)

## 📚 Learning Objectives
By the end of this week, students will be able to:
- Understand game theory fundamentals and zero-sum games
- Implement the Minimax algorithm for perfect decision making
- Apply Alpha-Beta pruning to optimize search efficiency
- Design evaluation functions for imperfect information games
- Analyze the computational complexity of game-playing algorithms

---

## 🎮 Games as Search Problems

Games provide a rich domain for AI research because they involve:
- **Strategic thinking** - planning several moves ahead
- **Adversarial environments** - opponents work against you
- **Real-time constraints** - decisions must be made quickly
- **Imperfect information** - can't see opponent's cards/plans

![Game Tree](images/minimax-game-tree.png)

### Why Study Game Playing?
1. **Historical Significance:** Chess, checkers, Go breakthroughs
2. **Strategic Reasoning:** Planning under uncertainty
3. **Real-world Applications:** Economics, military strategy, business
4. **Computational Challenges:** Exponential search spaces

---

## 🏗️ Game Formalization

### Components of a Game
- **Players:** Who makes decisions (MAX and MIN)
- **States:** Configurations of the game
- **Actions:** Legal moves from each state  
- **Terminal Test:** When the game ends
- **Utility Function:** Payoff for terminal states

### Types of Games

#### 1. **Perfect vs. Imperfect Information**
- **Perfect:** All information visible (Chess, Checkers, Go)
- **Imperfect:** Hidden information (Poker, Bridge, Kriegspiel)

#### 2. **Deterministic vs. Stochastic**
- **Deterministic:** No random elements (Chess, Tic-tac-toe)
- **Stochastic:** Random elements (Backgammon, Monopoly)

#### 3. **Zero-sum vs. Non-zero-sum**
- **Zero-sum:** One player's gain = other's loss (Chess, Poker)
- **Non-zero-sum:** Both players can win/lose (Prisoner's Dilemma)

### Game Tree Representation

```python
class GameState:
    def __init__(self, board, current_player, move_history):
        self.board = board
        self.current_player = current_player  # 'MAX' or 'MIN'
        self.move_history = move_history
    
    def get_legal_moves(self):
        """Return list of legal moves from current state"""
        pass
    
    def make_move(self, move):
        """Return new game state after making move"""
        pass
    
    def is_terminal(self):
        """Check if game is over"""
        pass
    
    def get_utility(self):
        """Return utility value for terminal state"""
        pass
```

---

## 🎯 The Minimax Algorithm

### Core Concept
**Minimax** assumes both players play optimally:
- **MAX player** tries to maximize the game value
- **MIN player** tries to minimize the game value
- Each player assumes opponent plays perfectly

### Algorithm Implementation

```python
def minimax(state, depth, maximizing_player):
    """Basic minimax algorithm implementation"""
    
    # Base case: terminal state or depth limit reached
    if depth == 0 or state.is_terminal():
        return state.evaluate(), None
    
    if maximizing_player:
        max_eval = float('-inf')
        best_move = None
        
        for move in state.get_legal_moves():
            new_state = state.make_move(move)
            eval_score, _ = minimax(new_state, depth-1, False)
            
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
        
        return max_eval, best_move
    
    else:  # Minimizing player
        min_eval = float('inf')
        best_move = None
        
        for move in state.get_legal_moves():
            new_state = state.make_move(move)
            eval_score, _ = minimax(new_state, depth-1, True)
            
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
        
        return min_eval, best_move

# Usage example
def choose_move(game_state, search_depth=4):
    """Choose the best move using minimax"""
    score, move = minimax(game_state, search_depth, True)
    return move
```

### Minimax Properties
- ✅ **Complete:** Yes (if game tree is finite)
- ✅ **Optimal:** Yes (against perfect opponent)
- ❌ **Time Complexity:** O(b^m) where b=branching factor, m=max depth
- ❌ **Space Complexity:** O(bm) for DFS implementation

### Example: Tic-Tac-Toe Implementation

```python
class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9  # 3x3 board
        self.current_player = 'X'  # X is MAX, O is MIN
    
    def get_legal_moves(self):
        return [i for i, cell in enumerate(self.board) if cell == ' ']
    
    def make_move(self, position):
        new_game = TicTacToe()
        new_game.board = self.board.copy()
        new_game.board[position] = self.current_player
        new_game.current_player = 'O' if self.current_player == 'X' else 'X'
        return new_game
    
    def is_terminal(self):
        return self.check_winner() is not None or ' ' not in self.board
    
    def check_winner(self):
        # Check rows, columns, and diagonals
        lines = [
            [0,1,2], [3,4,5], [6,7,8],  # rows
            [0,3,6], [1,4,7], [2,5,8],  # columns  
            [0,4,8], [2,4,6]            # diagonals
        ]
        
        for line in lines:
            if (self.board[line[0]] == self.board[line[1]] == self.board[line[2]] 
                and self.board[line[0]] != ' '):
                return self.board[line[0]]
        return None
    
    def evaluate(self):
        winner = self.check_winner()
        if winner == 'X': return 1    # MAX wins
        elif winner == 'O': return -1 # MIN wins
        else: return 0                # Tie

# Perfect tic-tac-toe player
def perfect_tictactoe_move(game):
    _, move = minimax(game, 9, game.current_player == 'X')
    return move
```

Game tree for tic-tac-toe:
- **Depth:** Up to 9 moves
- **Branching factor:** Decreases from 9 to 1
- **States:** ~300,000 total positions
- **Result:** Perfect play always leads to tie

---

## ⚡ Alpha-Beta Pruning

### The Problem with Pure Minimax
Chess game tree:
- **Branching factor:** ~35
- **Game length:** ~100 plies (half-moves)
- **Total positions:** ~35^100 ≈ 10^154 (more than atoms in universe!)

### Alpha-Beta Solution
**Key Insight:** We can eliminate branches that won't affect the final decision

### Alpha-Beta Variables
- **α (alpha):** Best value that MAX can guarantee so far
- **β (beta):** Best value that MIN can guarantee so far
- **Pruning condition:** If α ≥ β, prune remaining branches

![Alpha-Beta Pruning](images/alpha-beta-pruning.png)

### Alpha-Beta Implementation

```python
def alpha_beta(state, depth, alpha, beta, maximizing_player):
    """Alpha-beta pruning implementation"""
    
    if depth == 0 or state.is_terminal():
        return state.evaluate(), None
    
    best_move = None
    
    if maximizing_player:
        max_eval = float('-inf')
        
        for move in state.get_legal_moves():
            new_state = state.make_move(move)
            eval_score, _ = alpha_beta(new_state, depth-1, alpha, beta, False)
            
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            
            alpha = max(alpha, eval_score)
            
            # Alpha-beta pruning
            if beta <= alpha:
                break  # Beta cutoff
        
        return max_eval, best_move
    
    else:  # Minimizing player
        min_eval = float('inf')
        
        for move in state.get_legal_moves():
            new_state = state.make_move(move)
            eval_score, _ = alpha_beta(new_state, depth-1, alpha, beta, True)
            
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            
            beta = min(beta, eval_score)
            
            # Alpha-beta pruning
            if beta <= alpha:
                break  # Alpha cutoff
        
        return min_eval, best_move

# Wrapper function for easier use
def alpha_beta_search(state, depth):
    """Find best move using alpha-beta pruning"""
    _, move = alpha_beta(state, depth, float('-inf'), float('inf'), True)
    return move
```

### Alpha-Beta Effectiveness
- **Best case:** O(b^(m/2)) - can search twice as deep!
- **Average case:** O(b^(3m/4))
- **Move ordering crucial:** Best moves first = more pruning

### Move Ordering for Better Pruning

```python
def order_moves(state, moves):
    """Order moves to improve alpha-beta pruning"""
    def move_priority(move):
        # Higher priority = examined first
        new_state = state.make_move(move)
        
        # Prioritize:
        # 1. Winning moves
        if new_state.is_terminal() and new_state.evaluate() > 0:
            return 1000
        
        # 2. Center moves (for positional games)
        if move in get_center_positions():
            return 100
        
        # 3. Captures (if applicable)
        if is_capture_move(move):
            return 50
        
        return 0
    
    return sorted(moves, key=move_priority, reverse=True)
```

---

## 📊 Evaluation Functions

### The Need for Evaluation
Real games are too large to search to terminal states:
- **Chess:** ~10^120 positions
- **Go:** ~10^360 positions

**Solution:** Use heuristic evaluation function for non-terminal states

### Designing Evaluation Functions

#### 1. **Material Value** (Chess example)
```python
piece_values = {
    'pawn': 100, 'knight': 320, 'bishop': 330,
    'rook': 500, 'queen': 900, 'king': 20000
}

def material_evaluation(board):
    """Simple material-based evaluation"""
    white_material = sum(piece_values[piece.type] 
                        for piece in board.white_pieces)
    black_material = sum(piece_values[piece.type] 
                        for piece in board.black_pieces)
    return white_material - black_material
```

#### 2. **Positional Factors**
```python
def positional_evaluation(board):
    """More sophisticated evaluation considering position"""
    score = 0
    
    # Material value
    score += material_evaluation(board)
    
    # Piece mobility (number of legal moves)
    white_mobility = len(board.get_legal_moves('white'))
    black_mobility = len(board.get_legal_moves('black'))
    score += (white_mobility - black_mobility) * 10
    
    # King safety
    score += evaluate_king_safety(board, 'white') * 50
    score -= evaluate_king_safety(board, 'black') * 50
    
    # Pawn structure
    score += evaluate_pawn_structure(board, 'white') * 20
    score -= evaluate_pawn_structure(board, 'black') * 20
    
    # Control of center squares
    score += evaluate_center_control(board, 'white') * 30
    score -= evaluate_center_control(board, 'black') * 30
    
    return score

def evaluate_king_safety(board, color):
    """Evaluate how safe the king is"""
    king_pos = board.get_king_position(color)
    threats = count_threats_to_square(board, king_pos)
    return -threats  # Fewer threats = better safety

def evaluate_center_control(board, color):
    """Evaluate control of central squares"""
    center_squares = [(3,3), (3,4), (4,3), (4,4)]
    control_score = 0
    
    for square in center_squares:
        if board.is_attacked_by(square, color):
            control_score += 1
        if board.has_piece_on(square, color):
            control_score += 2
    
    return control_score
```

#### 3. **Weighted Linear Evaluation**
```python
class ChessEvaluator:
    def __init__(self):
        self.weights = {
            'material': 1.0,
            'mobility': 0.1,
            'king_safety': 0.5,
            'pawn_structure': 0.2,
            'center_control': 0.3
        }
    
    def evaluate(self, board):
        """Comprehensive position evaluation"""
        features = {
            'material': material_evaluation(board),
            'mobility': mobility_evaluation(board),
            'king_safety': king_safety_evaluation(board),
            'pawn_structure': pawn_structure_evaluation(board),
            'center_control': center_control_evaluation(board)
        }
        
        return sum(self.weights[feature] * value 
                  for feature, value in features.items())
```

### Evaluation Function Properties
- **Computational efficiency:** Must be fast to evaluate
- **Strong correlation:** With actual winning probability
- **Consistent ordering:** Better positions get higher values

---

## 🔧 Advanced Game-Playing Techniques

### 1. **Iterative Deepening**
```python
def iterative_deepening_search(state, time_limit):
    """Search with increasing depth until time runs out"""
    import time
    
    start_time = time.time()
    best_move = None
    depth = 1
    
    while time.time() - start_time < time_limit:
        try:
            _, move = alpha_beta(state, depth, float('-inf'), float('inf'), True)
            best_move = move
            depth += 1
        except TimeoutError:
            break
    
    return best_move
```

- **Benefits:** Anytime algorithm, better move ordering
- **Used in:** Most competitive chess programs

### 2. **Transposition Tables**
```python
class TranspositionTable:
    def __init__(self):
        self.table = {}
    
    def get_hash(self, state):
        """Generate unique hash for game state"""
        return hash(str(state.board))
    
    def lookup(self, state, depth):
        """Look up previously computed evaluation"""
        state_hash = self.get_hash(state)
        
        if state_hash in self.table:
            stored_depth, stored_value = self.table[state_hash]
            if stored_depth >= depth:
                return stored_value
        
        return None
    
    def store(self, state, depth, value):
        """Store evaluation result"""
        state_hash = self.get_hash(state)
        self.table[state_hash] = (depth, value)

# Usage in alpha-beta search
def alpha_beta_with_tt(state, depth, alpha, beta, maximizing_player, tt):
    """Alpha-beta with transposition table"""
    
    # Check transposition table first
    tt_result = tt.lookup(state, depth)
    if tt_result is not None:
        return tt_result, None
    
    if depth == 0 or state.is_terminal():
        value = state.evaluate()
        tt.store(state, depth, value)
        return value, None
    
    # ... rest of alpha-beta algorithm
    
    # Store result in transposition table
    tt.store(state, depth, best_value)
    return best_value, best_move
```

### 3. **Opening Books and Endgame Databases**
- **Opening books:** Pre-computed good opening moves
- **Endgame tables:** Perfect play for ≤7 pieces (chess)
- **Reduces search** in well-understood positions

---

## 💻 Hands-On Exercise: Complete Tic-Tac-Toe AI

```python
class TicTacToeAI:
    """Complete tic-tac-toe AI with minimax and alpha-beta"""
    
    def __init__(self, algorithm='alpha_beta'):
        self.algorithm = algorithm
    
    def get_best_move(self, game_state):
        """Get the best move for current position"""
        if self.algorithm == 'minimax':
            _, move = self.minimax(game_state, 9, True)
        else:  # alpha-beta
            _, move = self.alpha_beta(game_state, 9, float('-inf'), float('inf'), True)
        return move
    
    def minimax(self, state, depth, maximizing):
        if state.is_terminal() or depth == 0:
            return state.evaluate(), None
        
        moves = state.get_legal_moves()
        best_move = moves[0] if moves else None
        
        if maximizing:
            best_value = float('-inf')
            for move in moves:
                new_state = state.make_move(move)
                value, _ = self.minimax(new_state, depth-1, False)
                if value > best_value:
                    best_value = value
                    best_move = move
            return best_value, best_move
        else:
            best_value = float('inf')
            for move in moves:
                new_state = state.make_move(move)
                value, _ = self.minimax(new_state, depth-1, True)
                if value < best_value:
                    best_value = value
                    best_move = move
            return best_value, best_move
    
    def alpha_beta(self, state, depth, alpha, beta, maximizing):
        if state.is_terminal() or depth == 0:
            return state.evaluate(), None
        
        moves = state.get_legal_moves()
        best_move = moves[0] if moves else None
        
        if maximizing:
            best_value = float('-inf')
            for move in moves:
                new_state = state.make_move(move)
                value, _ = self.alpha_beta(new_state, depth-1, alpha, beta, False)
                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, value)
                if beta <= alpha:
                    break  # Prune
            return best_value, best_move
        else:
            best_value = float('inf')
            for move in moves:
                new_state = state.make_move(move)
                value, _ = self.alpha_beta(new_state, depth-1, alpha, beta, True)
                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, value)
                if beta <= alpha:
                    break  # Prune
            return best_value, best_move

# Test the AI
def play_game():
    game = TicTacToe()
    ai = TicTacToeAI()
    
    while not game.is_terminal():
        if game.current_player == 'X':
            # AI move
            move = ai.get_best_move(game)
            print(f"AI plays position {move}")
        else:
            # Human move
            move = int(input("Enter your move (0-8): "))
        
        game = game.make_move(move)
        print(game.display_board())
    
    winner = game.check_winner()
    if winner:
        print(f"{winner} wins!")
    else:
        print("It's a tie!")

# Uncomment to play: play_game()
```

---

## 🏆 Famous AI Game Milestones

### Chess Achievements
- **1997:** Deep Blue defeats Garry Kasparov
  - 200 million positions per second
  - Specialized chess hardware
  - Extensive opening book and endgame database

### Go Breakthroughs  
- **2016:** AlphaGo defeats Lee Sedol
  - Monte Carlo Tree Search + Deep Learning
  - Value networks + policy networks
  - Self-play reinforcement learning

### Modern Developments
- **2017:** AlphaZero masters chess, Go, shogi from scratch
- **2019:** Pluribus beats professional poker players
- **Recent:** AI systems excel at real-time strategy games

---

## 🔗 Curated Resources

### Essential Reading
- [AIMA Chapter 5](https://aima.cs.berkeley.edu/) - Adversarial Search and Games
- [Chess Programming Wiki](https://www.chessprogramming.org/) - Comprehensive game programming resource

### Video Resources
- [Minimax Algorithm](https://www.youtube.com/watch?v=l-hh51ncgDI) - CS50 AI (20 min)
- [Alpha-Beta Pruning](https://www.youtube.com/watch?v=xBXHtz4Gbdo) - Computerphile (9 min)
- [Deep Blue vs Kasparov](https://www.youtube.com/watch?v=NJarxpYyoFI) - Documentary clip

### Interactive Learning
- [Minimax Visualization](https://madebyevan.com/algosim/) - Interactive algorithm simulator
- [Chess.com Analysis Board](https://www.chess.com/analysis) - Analyze positions with engines
- [Tic-Tac-Toe AI Trainer](https://playtictactoe.org/) - Play against perfect AI

### Practice Problems
- [LeetCode: Predict the Winner](https://leetcode.com/problems/predict-the-winner/) (Minimax)
- [Stone Game Problems](https://leetcode.com/problems/stone-game/) (Game theory)
- [Nim Game](https://leetcode.com/problems/nim-game/) (Game theory basics)

---

## 🎯 Key Concepts Summary

### Minimax Algorithm
- **Optimal strategy** assuming perfect opponent
- **Exponential complexity** limits search depth
- **Foundation** for all competitive game AI

### Alpha-Beta Pruning
- **Eliminates irrelevant branches** from search
- **Can double effective search depth**
- **Move ordering crucial** for effectiveness

### Evaluation Functions
- **Heuristic assessment** of non-terminal positions
- **Balance accuracy with speed**
- **Domain knowledge essential** for good evaluation

### Real-World Considerations
- **Time constraints** require depth limits
- **Memory management** for large search trees
- **Opening preparation** and **endgame knowledge**

---

## 🤔 Discussion Questions

1. Why can't we just increase computer speed to solve games like chess perfectly?
2. How would you design an evaluation function for a new game?
3. What makes some games more suitable for AI than others?
4. How do modern AI techniques (deep learning) change game playing?
5. Is perfect play always the most fun to play against?

---

## 🔍 Looking Ahead

Next week, we'll move from adversarial search to **knowledge representation and reasoning**. We'll learn how to encode what an AI system knows about the world and how to make logical inferences from that knowledge.

**Preview of Week 5:**
- Propositional and first-order logic
- Knowledge bases and inference engines
- Resolution theorem proving
- Applications in expert systems

---

### Navigation
[🏠 Home](index.html) | [← Previous: Week 3](week03.html) | [Next: Week 5 →](week05.html)

---

*"The question isn't whether a computer can think like a human, but whether it can play games well enough to beat humans - and the answer is increasingly yes."*