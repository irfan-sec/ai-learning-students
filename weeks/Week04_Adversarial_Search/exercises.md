# Week 4 Exercises: Adversarial Search (Game Playing)

## 🎯 Learning Goals Review
After completing these exercises, you should be able to:
- Implement minimax algorithm for two-player games
- Apply alpha-beta pruning to optimize search
- Design evaluation functions for game states
- Analyze game trees and predict optimal play
- Build complete game-playing AI systems

---

## 📝 Conceptual Questions

### Question 1: Game Tree Analysis
**Consider this game tree (MAX moves first):**

```
           A
        /  |  \
       B   C   D
      /\  /\  /\
     E F G H I J
     3 2 5 1 4 6
```

**Tasks:**
a) Apply minimax algorithm. What is the minimax value of A?
b) Show the values backed up at each level
c) Which move should MAX choose at the root?
d) Apply alpha-beta pruning. Which nodes can be pruned?
e) What is the order of node evaluation? (left to right)

---

### Question 2: Alpha-Beta Pruning
**For the following game tree, show alpha-beta pruning:**

```
                    MAX
           /         |         \
         MIN        MIN        MIN
       /  |  \    /  |  \    /  |  \
      3   5   2  8   2   7  4   1   9
```

**Your task:**
1. Show alpha and beta values at each node
2. Mark pruned branches with an X
3. Count nodes evaluated: With/without pruning
4. Explain why each pruning occurs

**Template:**
```
Node A (MAX): α = -∞, β = +∞
  Node B (MIN): α = -∞, β = +∞
    Leaf: 3 → β = 3
    Leaf: 5 → β = 3 (no change)
    ...
```

---

### Question 3: Evaluation Functions
**Consider a Tic-Tac-Toe position:**

```
X | O | X
---------
O |   | O
---------
X |   |
```

**Design three different evaluation functions:**
a) **Simple:** Based only on potential wins
b) **Medium:** Consider multiple features
c) **Complex:** Weight different patterns

**Compare:**
- Which would make better decisions?
- What are the computational trade-offs?
- How would you test which is best?

---

## 🔍 Algorithm Analysis

### Exercise 1: Minimax Trace
**Nim Game:** Players alternate taking 1-3 objects from a pile. The player taking the last object loses.

**Game tree (4 objects, MAX goes first):**

```
                    4 (MAX)
          /         |         \
        3(MIN)     2(MIN)    1(MIN)
       /  |  \     /   \        |
     2   1   0   1     0        0
    MAX MAX WIN MIN  WIN       WIN
```

**Your tasks:**
1. Complete the game tree to depth 4
2. Assign terminal values (WIN for MIN = -1, WIN for MAX = +1)
3. Apply minimax from bottom up
4. Determine optimal strategy for both players
5. Who wins with optimal play from state 4?

---

### Exercise 2: Pruning Analysis
**Show that alpha-beta pruning doesn't affect the minimax decision:**

**Proof outline:**
1. State what alpha-beta pruning guarantees
2. Show that pruned subtrees can't affect the result
3. Prove that the root value is identical to minimax

**Mathematical formulation:**
- Let M(n) = minimax value of node n
- Let AB(n) = alpha-beta value of node n
- Prove: M(root) = AB(root) for any game tree

---

## 💻 Programming Exercises

### Exercise 3: Implement Minimax

Complete the minimax implementation:

```python
from typing import Tuple, Optional

class GameState:
    """Base class for game states."""
    
    def is_terminal(self) -> bool:
        """Check if game is over."""
        raise NotImplementedError
    
    def get_utility(self, player: int) -> float:
        """Return utility for the given player in terminal state."""
        raise NotImplementedError
    
    def get_legal_moves(self):
        """Return list of legal moves."""
        raise NotImplementedError
    
    def make_move(self, move):
        """Return new state after making move."""
        raise NotImplementedError
    
    def current_player(self) -> int:
        """Return current player (1 or -1)."""
        raise NotImplementedError

def minimax(state: GameState, depth: int) -> Tuple[float, Optional[object]]:
    """
    Minimax algorithm with depth limit.
    
    Args:
        state: Current game state
        depth: Maximum search depth
    
    Returns:
        (best_value, best_move) tuple
    """
    # TODO: Implement minimax
    # Base cases:
    #   1. Terminal state
    #   2. Depth limit reached
    # Recursive case:
    #   1. Get legal moves
    #   2. Try each move
    #   3. Recursively evaluate
    #   4. Choose max/min based on player
    pass

def minimax_decision(state: GameState, depth: int):
    """Return the best move using minimax."""
    # TODO: Call minimax and return best move
    pass
```

**Test with Tic-Tac-Toe:**

```python
class TicTacToe(GameState):
    def __init__(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.player = 1  # 1 for X, -1 for O
    
    def is_terminal(self):
        """Check for win or draw."""
        # TODO: Implement
        pass
    
    def get_utility(self, player):
        """Return 1 if player wins, -1 if loses, 0 for draw."""
        # TODO: Implement
        pass
    
    def get_legal_moves(self):
        """Return list of (row, col) for empty cells."""
        # TODO: Implement
        pass
    
    def make_move(self, move):
        """Return new state after move."""
        # TODO: Implement (return new TicTacToe object)
        pass
    
    def display(self):
        """Print the board."""
        for row in self.board:
            print(' | '.join(row))
            print('-' * 9)

# Test the implementation
game = TicTacToe()
while not game.is_terminal():
    if game.current_player() == 1:
        # Human move
        move = get_human_move(game)
    else:
        # AI move
        move = minimax_decision(game, depth=9)
    game = game.make_move(move)
    game.display()
```

---

### Exercise 4: Implement Alpha-Beta Pruning

Add alpha-beta pruning to minimax:

```python
def alpha_beta(state: GameState, depth: int, 
               alpha: float = float('-inf'), 
               beta: float = float('inf')) -> Tuple[float, Optional[object]]:
    """
    Minimax with alpha-beta pruning.
    
    Args:
        state: Current game state
        depth: Maximum search depth
        alpha: Best value for MAX along path to root
        beta: Best value for MIN along path to root
    
    Returns:
        (best_value, best_move) tuple
    """
    # TODO: Implement alpha-beta pruning
    # Key differences from minimax:
    #   1. Pass alpha and beta through recursion
    #   2. Update alpha (MAX) or beta (MIN) after each child
    #   3. Prune when alpha >= beta
    #   4. Return best value and move
    pass

# Track nodes evaluated
nodes_minimax = 0
nodes_alphabeta = 0

def compare_pruning(state: GameState, depth: int):
    """Compare minimax and alpha-beta performance."""
    global nodes_minimax, nodes_alphabeta
    
    # Reset counters
    nodes_minimax = 0
    nodes_alphabeta = 0
    
    # Run both algorithms
    val_mm, move_mm = minimax(state, depth)
    val_ab, move_ab = alpha_beta(state, depth)
    
    print(f"Minimax: {nodes_minimax} nodes evaluated")
    print(f"Alpha-Beta: {nodes_alphabeta} nodes evaluated")
    print(f"Pruning efficiency: {(1 - nodes_alphabeta/nodes_minimax)*100:.1f}%")
    
    # Verify they give same result
    assert val_mm == val_ab, "Values should match!"
    print(f"Both found best move: {move_ab}")
```

---

### Exercise 5: Evaluation Function for Connect Four

Implement Connect Four with a heuristic evaluation function:

```python
class ConnectFour(GameState):
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.board = [[0 for _ in range(cols)] for _ in range(rows)]
        self.player = 1
    
    def drop_piece(self, col: int):
        """Drop a piece in the specified column."""
        for row in range(self.rows-1, -1, -1):
            if self.board[row][col] == 0:
                new_state = self.copy()
                new_state.board[row][col] = self.player
                new_state.player *= -1
                return new_state
        return None  # Column full
    
    def evaluate(self) -> float:
        """
        Heuristic evaluation function for non-terminal states.
        Consider:
        - Number of 2-in-a-row, 3-in-a-row patterns
        - Center column control
        - Potential winning opportunities
        - Blocking opponent threats
        """
        # TODO: Implement evaluation function
        score = 0
        
        # Count patterns for each player
        # Give higher scores to better patterns
        # Consider horizontal, vertical, and diagonal lines
        
        return score
    
    def count_patterns(self, player: int, length: int) -> int:
        """Count number of patterns of given length for player."""
        # TODO: Count horizontal, vertical, diagonal patterns
        pass
    
    def check_winner(self) -> Optional[int]:
        """Return winner (1, -1) or None if no winner yet."""
        # TODO: Check all possible 4-in-a-row combinations
        pass
```

**Design a good evaluation function that considers:**
1. Immediate wins and losses
2. 3-in-a-row with empty spot (threat)
3. 2-in-a-row with two empty spots
4. Center column control
5. Height advantage

---

### Exercise 6: Move Ordering Optimization

Implement move ordering to improve alpha-beta efficiency:

```python
def order_moves(state: GameState, moves: list) -> list:
    """
    Order moves to search most promising first.
    Better move ordering = more pruning.
    
    Strategies:
    1. Captures/winning moves first
    2. Center positions in many games
    3. Moves that worked well at previous depth
    4. Quick evaluation heuristic
    """
    # TODO: Implement move ordering
    pass

def alpha_beta_with_ordering(state: GameState, depth: int,
                             alpha: float = float('-inf'),
                             beta: float = float('inf')):
    """Alpha-beta with move ordering."""
    if state.is_terminal() or depth == 0:
        return state.evaluate(), None
    
    moves = state.get_legal_moves()
    moves = order_moves(state, moves)  # Order moves first!
    
    # Rest of alpha-beta implementation...
    # TODO: Complete implementation
    pass

# Test move ordering effectiveness
def test_move_ordering():
    """Compare pruning with and without move ordering."""
    # TODO: Run experiments and report results
    pass
```

---

## 🧩 Problem-Solving Challenges

### Challenge 1: Othello/Reversi AI
**Build a complete Othello game with AI:**

**Requirements:**
- Implement all game rules (flipping pieces)
- Create GUI using pygame or similar
- Implement minimax with alpha-beta pruning
- Design evaluation function considering:
  - Piece count
  - Corner control (very valuable!)
  - Mobility (number of legal moves)
  - Edge control
- Allow human vs AI play
- Display AI's thinking (best move, evaluation)

**Advanced features:**
- Iterative deepening for time management
- Opening book for early game
- Endgame perfect play (when few squares remain)

---

### Challenge 2: Chess Tactics Solver
**Build a chess position analyzer:**

**Objectives:**
- Load chess positions (FEN notation)
- Find tactical combinations (checkmates, winning material)
- Use minimax to search for forced wins
- Handle special moves (castling, en passant)

**Evaluation function components:**
```python
def evaluate_chess_position(board):
    """
    Evaluate chess position.
    Consider:
    - Material balance (piece values)
    - Piece positioning (piece-square tables)
    - King safety
    - Pawn structure
    - Mobility
    """
    score = 0
    
    # Material (Q=9, R=5, B=3, N=3, P=1)
    score += count_material(board)
    
    # Position bonuses
    score += evaluate_piece_positions(board)
    
    # TODO: Add more factors
    
    return score
```

**Use python-chess library for move generation.**

---

### Challenge 3: Custom Game AI
**Design and implement your own board game with AI:**

**Requirements:**
1. Define unique game rules
2. Implement game logic
3. Create visual representation
4. Build AI opponent using minimax/alpha-beta
5. Design domain-specific evaluation function

**Example games:**
- Modified chess variant
- Custom connection game
- Strategic tile-placement game
- Unique combat card game

**Documentation:**
- Explain game rules clearly
- Justify evaluation function design
- Analyze AI playing strength
- Discuss interesting strategic patterns discovered

---

## 🔬 Advanced Exercises

### Exercise 7: Prove Alpha-Beta Correctness

**Theorem:** Alpha-beta pruning returns the same value as minimax.

**Proof structure:**
1. Define alpha and beta invariants
2. Show pruning only occurs when subtree can't affect result
3. Prove by induction on tree depth

**Your task:** Write formal proof with all steps.

---

### Exercise 8: Complexity Analysis

**Analyze time complexity:**

a) **Minimax without pruning:**
   - Branching factor: b
   - Depth: d
   - Time complexity: ___________

b) **Alpha-beta with random move ordering:**
   - Best case: ___________
   - Worst case: ___________
   - Average case: ___________

c) **Alpha-beta with perfect move ordering:**
   - Time complexity: ___________
   - Proof: ___________

**Experimental validation:**
- Implement experiments to verify theoretical analysis
- Plot nodes evaluated vs depth
- Compare with predictions

---

### Exercise 9: Iterative Deepening

**Implement iterative deepening for time-constrained games:**

```python
import time

def iterative_deepening(state: GameState, time_limit: float):
    """
    Search with increasing depth until time runs out.
    
    Args:
        state: Current game state
        time_limit: Maximum time in seconds
    
    Returns:
        Best move found within time limit
    """
    start_time = time.time()
    best_move = None
    depth = 1
    
    while time.time() - start_time < time_limit:
        try:
            # Search to current depth with time check
            value, move = alpha_beta(state, depth)
            best_move = move
            depth += 1
            
            print(f"Depth {depth-1}: value={value}, move={move}")
            
        except TimeoutError:
            break
    
    return best_move

def alpha_beta_with_timeout(state, depth, timeout):
    """Alpha-beta that raises TimeoutError if time exceeded."""
    # TODO: Implement with time checking
    pass
```

---

### Exercise 10: Transposition Table

**Implement transposition table for caching:**

```python
class TranspositionTable:
    """Cache for previously evaluated positions."""
    
    def __init__(self):
        self.table = {}
    
    def store(self, state_hash: int, depth: int, value: float, flag: str):
        """
        Store position evaluation.
        
        Args:
            state_hash: Hash of game state
            depth: Search depth for this value
            value: Evaluation value
            flag: 'EXACT', 'LOWER_BOUND', or 'UPPER_BOUND'
        """
        # TODO: Implement
        pass
    
    def lookup(self, state_hash: int, depth: int):
        """Look up position in table."""
        # TODO: Implement
        pass

def alpha_beta_with_transposition(state, depth, alpha, beta, tt):
    """Alpha-beta with transposition table."""
    state_hash = hash(state)
    
    # Try to retrieve from cache
    cached = tt.lookup(state_hash, depth)
    if cached:
        # TODO: Use cached value if applicable
        pass
    
    # Normal alpha-beta search...
    # Store result in transposition table before returning
    
    # TODO: Complete implementation
    pass
```

**Measure cache effectiveness:**
- Hit rate (% of lookups that succeed)
- Nodes saved
- Memory usage

---

## 📊 Analysis Questions

### Question 4: Game Complexity
**Classify these games by complexity:**

| Game | State Space | Branching Factor | Game Tree Complexity |
|------|-------------|------------------|---------------------|
| Tic-Tac-Toe | ~3^9 | ~5 | _____ |
| Connect Four | _____ | ~7 | _____ |
| Checkers | _____ | ~10 | _____ |
| Chess | _____ | ~35 | _____ |
| Go | _____ | ~250 | _____ |

**Questions:**
- Which can be solved completely?
- Which require heuristic evaluation?
- How does branching factor affect AI difficulty?

---

### Question 5: Evaluation Function Design
**For a given game, good evaluation functions should be:**
1. **Correlated with winning:** Higher values = better positions
2. **Efficient to compute:** Fast enough for deep search
3. **Consistent:** Similar positions get similar values

**Design evaluation functions for:**
a) Checkers
b) Nine Men's Morris
c) Battleship

**Justify your design choices.**

---

## 📝 Submission Guidelines

### What to Submit:
1. **Minimax and alpha-beta implementations** (working code)
2. **Complete game with AI** (Tic-Tac-Toe minimum, more complex preferred)
3. **Evaluation function analysis** (design and testing)
4. **Performance comparison** (minimax vs alpha-beta, with data)
5. **Analysis report** (750-1000 words):
   - Algorithm explanations
   - Design decisions
   - Performance results
   - Playing strength analysis
   - Challenges and solutions

### Evaluation Criteria:
- **Correctness:** Algorithms work properly and play legally
- **Efficiency:** Good use of alpha-beta pruning and optimization
- **Evaluation:** Well-designed, tested evaluation function
- **Analysis:** Deep understanding of game tree search
- **Code Quality:** Clean, documented, efficient code

### Bonus Points:
- Advanced games (chess, Go, etc.)
- Novel optimization techniques
- Sophisticated evaluation functions
- Tournament between different AIs
- GUI with visualization of AI thinking

### Due Date: End of Week 4

---

## 💡 Study Tips

### Mastering Game Trees
- **Draw by hand:** Small trees help intuition
- **Verify values:** Ensure backed-up values are correct
- **Practice pruning:** Trace alpha-beta many times
- **Play games:** Understand what makes positions good/bad

### Common Pitfalls
- **Forgetting to negate values:** Min nodes need negation
- **Wrong alpha-beta updates:** Track which player is moving
- **Poor move ordering:** Dramatic impact on pruning
- **Expensive evaluation:** Balance accuracy vs speed

### Advanced Topics
- **Quiescence search:** Extend search in tactical positions
- **Killer move heuristic:** Moves that caused cutoffs before
- **Principal variation search:** More aggressive pruning
- **Monte Carlo Tree Search:** Modern alternative for complex games

---

**🎯 Key Takeaway:** Game-playing AI demonstrates how search algorithms can create intelligent behavior in adversarial settings. The combination of deep search and smart evaluation creates superhuman play in many domains. Understanding these techniques is crucial for competitive AI development!
