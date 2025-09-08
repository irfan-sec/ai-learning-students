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

![Game Tree](https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Minimax.svg/400px-Minimax.svg.png)

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

---

## 🎯 The Minimax Algorithm

### Core Concept
**Minimax** assumes both players play optimally:
- **MAX player** tries to maximize the game value
- **MIN player** tries to minimize the game value
- Each player assumes opponent plays perfectly

### Algorithm Structure
```python
def minimax(state, depth, maximizing_player):
    if depth == 0 or game_over(state):
        return evaluate(state)
    
    if maximizing_player:
        max_eval = -infinity
        for child in get_children(state):
            eval = minimax(child, depth-1, False)
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = +infinity
        for child in get_children(state):
            eval = minimax(child, depth-1, True)
            min_eval = min(min_eval, eval)
        return min_eval
```

### Minimax Properties
- ✅ **Complete:** Yes (if game tree is finite)
- ✅ **Optimal:** Yes (against perfect opponent)
- ❌ **Time Complexity:** O(b^m) where b=branching factor, m=max depth
- ❌ **Space Complexity:** O(bm) for DFS implementation

### Example: Tic-Tac-Toe
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

### Alpha-Beta Algorithm
```python
def alpha_beta(state, depth, alpha, beta, maximizing_player):
    if depth == 0 or game_over(state):
        return evaluate(state)
    
    if maximizing_player:
        max_eval = -infinity
        for child in get_children(state):
            eval = alpha_beta(child, depth-1, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cutoff
        return max_eval
    else:
        min_eval = +infinity
        for child in get_children(state):
            eval = alpha_beta(child, depth-1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cutoff
        return min_eval
```

### Alpha-Beta Effectiveness
- **Best case:** O(b^(m/2)) - can search twice as deep!
- **Average case:** O(b^(3m/4))
- **Move ordering crucial:** Best moves first = more pruning

![Alpha-Beta Pruning](https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/AB_pruning.svg/400px-AB_pruning.svg.png)

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
    'pawn': 1, 'knight': 3, 'bishop': 3,
    'rook': 5, 'queen': 9, 'king': 1000
}

def material_evaluation(board):
    white_material = sum(piece_values[piece] for piece in white_pieces)
    black_material = sum(piece_values[piece] for piece in black_pieces) 
    return white_material - black_material
```

#### 2. **Positional Factors**
- **Piece mobility:** Number of legal moves
- **King safety:** Proximity to threats
- **Pawn structure:** Doubled, isolated, passed pawns
- **Control of center:** Central square occupation

#### 3. **Weighted Linear Evaluation**
```python
def evaluate_position(state):
    return (w1 * material(state) + 
            w2 * mobility(state) +
            w3 * king_safety(state) +
            w4 * pawn_structure(state))
```

### Evaluation Function Properties
- **Computational efficiency:** Must be fast to evaluate
- **Strong correlation:** With actual winning probability
- **Consistent ordering:** Better positions get higher values

---

## 🔧 Advanced Game-Playing Techniques

### 1. **Iterative Deepening**
- Search to depth 1, then 2, then 3, etc.
- **Benefits:** Anytime algorithm, better move ordering
- **Used in:** Most competitive chess programs

### 2. **Transposition Tables**
- Cache previously computed positions
- **Same position** can be reached via different move orders
- **Hash tables** for O(1) lookup time

### 3. **Forward Pruning**
- Eliminate "obviously bad" moves early
- **Null move pruning:** Skip a turn to detect zugzwang
- **Late move reductions:** Search likely poor moves less deeply

### 4. **Opening Books and Endgame Databases**
- **Opening books:** Pre-computed good opening moves
- **Endgame tables:** Perfect play for ≤7 pieces (chess)
- **Reduces search** in well-understood positions

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

## 🎲 Stochastic Games

### Adding Chance Nodes
In games with randomness (dice, card draws):
- **Chance nodes** represent random events
- **Expected value** over all possible outcomes

### Expectiminimax Algorithm
```python
def expectiminimax(state, depth, player):
    if depth == 0 or game_over(state):
        return evaluate(state)
    
    if player == 'MAX':
        return max(expectiminimax(child, depth-1, 'MIN') 
                  for child in get_children(state))
    elif player == 'MIN':
        return min(expectiminimax(child, depth-1, 'MAX')
                  for child in get_children(state))
    else:  # Chance node
        return sum(probability * expectiminimax(child, depth-1, next_player)
                  for child, probability in get_chance_outcomes(state))
```

---

## 💻 Implementation Considerations

### Move Ordering
**Critical for alpha-beta efficiency:**
1. **Previous best move** (from iterative deepening)
2. **Captures** (especially high-value targets)
3. **Checks and threats**
4. **Center moves** (positional games)

### Time Management
Real games have **time constraints:**
- **Fixed time per move**
- **Total time per game**
- **Increment per move**

**Strategies:**
- Allocate more time for critical positions
- Use remaining time information
- Implement graceful degradation

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

---

## 🔍 Looking Ahead

Next week, we'll move from adversarial search to **knowledge representation and reasoning**. We'll learn how to encode what an AI system knows about the world and how to make logical inferences from that knowledge.

---

*"The question isn't whether a computer can think like a human, but whether it can play games well enough to beat humans - and the answer is increasingly yes."*