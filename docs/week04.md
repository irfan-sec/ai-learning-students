---
layout: week
title: "Week 4: Adversarial Search and Game Playing"
subtitle: "AI in Competitive Environments"
description: "Learn how AI systems compete and make decisions in adversarial environments. Explore minimax, alpha-beta pruning, and evaluation functions for game playing."
week_number: 4
total_weeks: 14
github_folder: "Week04_Adversarial_Search"
notes_link: "https://github.com/irfan-sec/ai-learning-students/blob/main/weeks/Week04_Adversarial_Search/notes.md"
resources_link: "https://github.com/irfan-sec/ai-learning-students/blob/main/weeks/Week04_Adversarial_Search/resources.md"
exercises_link: "https://github.com/irfan-sec/ai-learning-students/blob/main/weeks/Week04_Adversarial_Search/exercises.md"
code_link: "https://github.com/irfan-sec/ai-learning-students/tree/main/weeks/Week04_Adversarial_Search/code"
prev_week:
  title: "Informed Search"
  url: "/week03.html"
next_week:
  title: "ML Fundamentals"
  url: "/week07.html"
objectives:
  - "Understand adversarial search problems and game theory basics"
  - "Learn and implement the minimax algorithm"
  - "Master alpha-beta pruning for efficiency"
  - "Design effective evaluation functions for games"
  - "Explore Monte Carlo Tree Search (MCTS) principles"
  - "Understand the challenges of real-time game playing"
key_concepts:
  - "Game Trees"
  - "Minimax Algorithm"
  - "Alpha-Beta Pruning"
  - "Evaluation Functions"
  - "Zero-Sum Games"
  - "Monte Carlo Tree Search"
  - "Iterative Deepening"
  - "Transposition Tables"
---

## 🎮 AI vs. Human: The Ultimate Challenge

Welcome to the exciting world of game-playing AI! This week, we'll explore how intelligent agents make decisions when facing adversaries. From chess grandmasters to Go champions, AI has achieved superhuman performance in games by mastering the principles you'll learn this week.

## 📚 What You'll Learn

### Core Topics

1. **Game Theory Fundamentals**
   - Zero-sum vs. non-zero-sum games
   - Perfect vs. imperfect information
   - Deterministic vs. stochastic games

2. **The Minimax Algorithm**
   - Game tree representation
   - Maximizing and minimizing players
   - Backup values and optimal play

3. **Alpha-Beta Pruning**
   - Eliminating unnecessary branches
   - Move ordering for better pruning
   - Time complexity improvements

4. **Evaluation Functions**
   - Designing good heuristics for games
   - Balancing different game aspects
   - Learning evaluation functions

## 🎯 The Minimax Algorithm

The foundation of game-playing AI:

```python
def minimax(state, depth, maximizing_player):
    if depth == 0 or game_over(state):
        return evaluate(state)
    
    if maximizing_player:
        max_eval = float('-inf')
        for move in get_legal_moves(state):
            new_state = make_move(state, move)
            eval_score = minimax(new_state, depth-1, False)
            max_eval = max(max_eval, eval_score)
        return max_eval
    else:
        min_eval = float('inf')
        for move in get_legal_moves(state):
            new_state = make_move(state, move)
            eval_score = minimax(new_state, depth-1, True)
            min_eval = min(min_eval, eval_score)
        return min_eval
```

## ⚡ Alpha-Beta Pruning

Dramatically reduce the search space:

```python
def minimax_ab(state, depth, alpha, beta, maximizing_player):
    if depth == 0 or game_over(state):
        return evaluate(state)
    
    if maximizing_player:
        max_eval = float('-inf')
        for move in get_legal_moves(state):
            new_state = make_move(state, move)
            eval_score = minimax_ab(new_state, depth-1, alpha, beta, False)
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # Beta cutoff
        return max_eval
    # ... similar for minimizing player
```

**Performance Gain:** With perfect move ordering, alpha-beta examines only O(b^(d/2)) nodes instead of O(b^d)!

## 🏆 Famous AI Game Victories

### Deep Blue vs. Kasparov (1997)
- **Game:** Chess
- **Key Techniques:** Minimax with sophisticated evaluation, specialized hardware
- **Significance:** First computer to defeat a reigning world chess champion

### AlphaGo vs. Lee Sedol (2016)
- **Game:** Go
- **Key Techniques:** Monte Carlo Tree Search + Deep Neural Networks
- **Significance:** Conquered a game thought too complex for computers

### OpenAI Five vs. Dota 2 Pros (2018)
- **Game:** Dota 2
- **Key Techniques:** Reinforcement Learning, massive parallel training
- **Significance:** Mastered complex real-time strategy

## 🎲 Game Types and Challenges

### Perfect Information Games
**Examples:** Chess, Checkers, Go, Tic-Tac-Toe
- All information visible to both players
- Minimax works perfectly (in theory)
- Challenge: Huge search spaces

### Imperfect Information Games
**Examples:** Poker, Bridge, Stratego
- Hidden information creates uncertainty
- Need probabilistic reasoning
- Information gathering becomes crucial

### Real-Time Games
**Examples:** StarCraft, Dota, FPS games
- Time pressure limits computation
- Anytime algorithms essential
- Must balance planning and reaction

## 🎯 Designing Evaluation Functions

### Tic-Tac-Toe Example
```python
def evaluate_ttt(board):
    # Simple but effective
    if winner(board) == 'X': return 100
    if winner(board) == 'O': return -100
    if is_draw(board): return 0
    
    # Heuristic for non-terminal positions
    score = 0
    score += count_potential_wins('X', board) * 10
    score -= count_potential_wins('O', board) * 10
    return score
```

### Chess Evaluation Components
- **Material:** Piece values (Queen=9, Rook=5, etc.)
- **Position:** Central squares more valuable
- **King Safety:** Castling, pawn shield
- **Pawn Structure:** Doubled, isolated, passed pawns
- **Piece Activity:** Mobility, outposts, coordination

## 🔬 Advanced Techniques

### Monte Carlo Tree Search (MCTS)
Used in AlphaGo and many modern game AIs:
1. **Selection:** Navigate tree using UCB1
2. **Expansion:** Add new nodes
3. **Simulation:** Random playout
4. **Backpropagation:** Update statistics

### Iterative Deepening
Search deeper when time permits:
```python
def iterative_deepening_search(state, time_limit):
    depth = 1
    best_move = None
    
    while time_remaining() > 0:
        try:
            move = minimax_with_cutoff(state, depth)
            best_move = move
            depth += 1
        except TimeoutException:
            break
            
    return best_move
```

## 🧠 Strategic Insights

### Opening Books
- Memorize known good opening sequences
- Transition to search when leaving "book"
- Balances theory and computation

### Endgame Databases
- Perfect play for positions with few pieces
- Tablebase: Pre-computed optimal moves
- Example: 6-piece chess endgames solved completely

### Time Management
- Allocate search time wisely
- Spend more time on critical positions
- Reserve time for endgame complications

## 🚀 Modern Applications

### Beyond Traditional Games
- **Financial Trading:** Market as adversarial environment
- **Cybersecurity:** Attack vs. defense scenarios  
- **Military Strategy:** Resource allocation under opposition
- **Sports Analytics:** Opponent modeling and strategy

### AI in Esports
- **Coaching Tools:** Analyze player strategies
- **Training Partners:** Consistent practice opponents
- **Balance Testing:** Identify overpowered strategies

## 💡 Key Takeaways

- **Minimax provides optimal play** in perfect-information games
- **Alpha-beta pruning** can provide massive speedups
- **Good evaluation functions** are often game-specific and domain-dependent
- **Time constraints** require anytime algorithms and smart time allocation
- **Modern game AI** combines traditional search with machine learning

## 🔗 Connections

- **Search Foundation:** Builds directly on previous search algorithms
- **Future ML:** Evaluation functions can be learned from data
- **Real-World Applications:** Decision-making under competition appears everywhere

---

*Ready to build your own game-playing AI? Start with tic-tac-toe and work your way up to more complex games!*