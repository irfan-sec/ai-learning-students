# Week 4 Resources: Adversarial Search (Game Playing)

## 📖 Required Readings

### Primary Textbook: AIMA (Russell & Norvig)
- **Chapter 5:** Adversarial Search and Games
  - Section 5.1: Game Theory
  - Section 5.2: Optimal Decisions in Games (Minimax)
  - Section 5.3: Alpha-Beta Pruning
  - Section 5.4: Imperfect Real-Time Decisions
  - Section 5.5: Stochastic Games (optional)

---

## 🎥 Video Resources

### Foundational Videos
1. **[Minimax Algorithm Explained](https://www.youtube.com/watch?v=l-hh51ncgDI)** - Sebastian Lague (10 min)
   - Clear visualization of minimax in Tic-Tac-Toe
   - Shows how the algorithm evaluates game trees

2. **[Alpha-Beta Pruning](https://www.youtube.com/watch?v=xBXHtz4Gbdo)** - MIT OpenCourseWare (15 min)
   - Explains how alpha-beta speeds up minimax
   - Demonstrates with concrete examples

3. **[Game Playing in AI](https://www.youtube.com/watch?v=J1GoI5WHBto)** - Computerphile (12 min)
   - Historical perspective on game-playing AI
   - From chess to modern systems

### Advanced Videos (Optional)
4. **[Monte Carlo Tree Search](https://www.youtube.com/watch?v=UXW2yZndl7U)** - DeepMind (15 min)
   - Modern alternative to minimax
   - Used in AlphaGo and AlphaZero

5. **[Evaluation Functions in Games](https://www.youtube.com/watch?v=STjW3eH0Cik)** - Gate Lectures (20 min)
   - How to design good evaluation functions
   - Trade-offs and heuristics

---

## 🌐 Interactive Resources

### Online Demos and Simulations
1. **[Minimax Visualizer](https://www.neverstopbuilding.com/blog/minimax)**
   - Interactive minimax algorithm visualization
   - Step through Tic-Tac-Toe game trees
   - See alpha-beta pruning in action

2. **[Chess Programming Wiki](https://www.chessprogramming.org/Main_Page)**
   - Comprehensive resource on game AI
   - Algorithms, evaluation functions, and optimizations
   - Historical and modern techniques

3. **[Nim Game AI](https://www.archimedes-lab.org/game_nim/play_nim_game.html)**
   - Play against minimax AI
   - Simple game to understand concepts

### Algorithm Comparisons
4. **[Game Tree Search Visualizer](https://raphsilva.github.io/utilities/minimax_simulator/)**
   - Compare minimax with and without alpha-beta
   - See number of nodes evaluated

---

## 💻 Code Resources

### Official AIMA Code
1. **[AIMA Python Repository](https://github.com/aimacode/aima-python)**
   - Chapter 5: [`games.py`](https://github.com/aimacode/aima-python/blob/master/games.py)
   - Complete implementations of minimax, alpha-beta
   - Examples: Tic-Tac-Toe, Connect Four

2. **[Games Notebook](https://github.com/aimacode/aima-python/blob/master/games.ipynb)**
   - Jupyter notebook with interactive game examples
   - Play against AI and visualize search trees

### Game Implementations
3. **[Python Chess Library](https://python-chess.readthedocs.io/)**
   - `pip install python-chess`
   - Full chess implementation with legal move generation
   - Perfect for building chess AI

4. **[Pygame Game Examples](https://github.com/techwithtim/Python-Checkers-AI)**
   - Checkers with minimax AI
   - Complete game with GUI

---

## 📚 Additional Reading Materials

### Academic Papers

1. **"Computing Machinery and Intelligence"** - Alan Turing (1950)
   - [Original paper](https://www.csee.umbc.edu/courses/471/papers/turing.pdf)
   - Turing's thoughts on game playing

2. **"Deep Blue"** - Campbell, Hoane, Hsu (2002)
   - [Paper on chess computer](https://www.sciencedirect.com/science/article/pii/S0004370201001291)
   - Beat world champion Garry Kasparov

3. **"Mastering the Game of Go with Deep Neural Networks"** - Silver et al. (2016)
   - [AlphaGo paper](https://www.nature.com/articles/nature16961)
   - Revolutionary combination of tree search and deep learning

4. **"Mastering Chess and Shogi by Self-Play"** - Silver et al. (2017)
   - [AlphaZero paper](https://arxiv.org/abs/1712.01815)
   - General game-playing system

### Tutorials and Guides

5. **[Building a Tic-Tac-Toe AI](https://www.geeksforgeeks.org/minimax-algorithm-in-game-theory-set-1-introduction/)**
   - Step-by-step minimax tutorial
   - With code examples

6. **[Alpha-Beta Pruning Guide](https://www.baeldung.com/cs/minimax-alpha-beta-pruning)**
   - Detailed explanation with examples
   - Optimization techniques

### Blog Posts and Articles

7. **[Chess Programming 101](https://www.freecodecamp.org/news/simple-chess-ai-step-by-step-1d55a9266977/)**
   - Building a chess AI from scratch
   - Evaluation functions and move ordering

8. **[MCTS Explained](https://int8.io/monte-carlo-tree-search-beginners-guide/)**
   - Modern alternative to minimax
   - Used in AlphaGo and many games

---

## 📊 Datasets and Problem Sets

### Classic Games
1. **[Chess Game Database](https://database.lichess.org/)**
   - Millions of real chess games
   - Learn from grandmaster strategies

2. **[Go Game Records](https://www.u-go.net/gamerecords/)**
   - Professional Go games
   - Analyze expert strategies

### Test Positions
3. **[Chess Puzzles](https://www.chessprogramming.org/Test-Positions)**
   - Test positions for chess engines
   - Evaluate AI strength

4. **[Connect Four Positions](http://blog.gamesolver.org/)**
   - Optimal play database
   - Benchmark your AI

---

## 🔧 Tools and Libraries

### Python Game Libraries
1. **Pygame** - Game development library
   - `pip install pygame`
   - Create game GUIs and visualizations
   - [Documentation](https://www.pygame.org/docs/)

2. **python-chess** - Chess library
   - `pip install python-chess`
   - Legal move generation, position evaluation
   - [Documentation](https://python-chess.readthedocs.io/)

### AI Libraries
3. **Minimax Implementations**
   ```python
   # Simple minimax template
   def minimax(state, depth, maximizing_player):
       if depth == 0 or is_terminal(state):
           return evaluate(state)
       
       if maximizing_player:
           max_eval = float('-inf')
           for child in get_children(state):
               eval = minimax(child, depth-1, False)
               max_eval = max(max_eval, eval)
           return max_eval
       else:
           min_eval = float('inf')
           for child in get_children(state):
               eval = minimax(child, depth-1, True)
               min_eval = min(min_eval, eval)
           return min_eval
   ```

4. **Tree Visualization**
   - `pip install graphviz`
   - Visualize game trees
   - Debug minimax execution

---

## 🎮 Practical Applications

### Game Development
1. **[Unity AI Tutorial](https://learn.unity.com/tutorial/programming-a-simple-ai)**
   - AI in Unity game engine
   - Minimax and behavior trees

2. **[Unreal Engine AI](https://docs.unrealengine.com/en-US/InteractiveExperiences/ArtificialIntelligence/)**
   - Game AI in Unreal Engine
   - State machines and planning

### Competition Platforms
3. **[Kaggle Connect X Competition](https://www.kaggle.com/c/connectx)**
   - Build AI for Connect Four variant
   - Compete against other AIs

4. **[CodinGame](https://www.codingame.com/)**
   - Programming games with AI
   - Multiplayer bot competitions

---

## 📖 Books (Supplementary)

1. **"Programming Game AI by Example"** - Mat Buckland
   - Comprehensive game AI techniques
   - Includes state machines, pathfinding, and more

2. **"Artificial Intelligence for Games"** - Ian Millington & John Funge
   - Industry-standard reference
   - Movement, decision making, learning

3. **"Game AI Pro"** series - Various authors
   - Collection of advanced techniques
   - Industry best practices

---

## 🏆 Notable Game AI Systems

### Historical Milestones
1. **Deep Blue (1997)** - Beat world chess champion
   - Brute force search with specialized hardware
   - 200 million positions per second

2. **Chinook (1994)** - Solved checkers completely
   - First game solved by computer
   - Perfect play demonstrated

3. **TD-Gammon (1992)** - Backgammon master
   - Used reinforcement learning
   - Near-human performance

### Modern Systems
4. **AlphaGo (2016)** - Beat world Go champion
   - Deep learning + Monte Carlo Tree Search
   - Revolutionized game AI

5. **AlphaZero (2017)** - General game playing
   - Mastered chess, shogi, and Go
   - Self-play learning from scratch

6. **OpenAI Five (2018)** - Dota 2 team
   - Complex real-time strategy game
   - Multi-agent coordination

---

## 🔬 Research Topics

### Advanced Algorithms
1. **Monte Carlo Tree Search (MCTS)**
   - Probabilistic tree search
   - Exploration vs exploitation balance

2. **Proof Number Search**
   - For game tree analysis
   - Finds winning strategies

3. **UCT (Upper Confidence Bound for Trees)**
   - MCTS variant
   - Used in AlphaGo

### Evaluation Functions
4. **Machine Learning for Evaluation**
   - Neural networks for position evaluation
   - Feature learning from data

5. **Endgame Databases**
   - Pre-computed perfect play
   - Used in chess engines

---

## 💡 Study Tips and Best Practices

### Understanding Minimax
- **Draw Trees:** Sketch small game trees by hand
- **Trace Execution:** Follow algorithm step-by-step
- **Verify Values:** Check that values propagate correctly
- **Compare Strategies:** Play games yourself vs having AI play

### Debugging Game AI
- **Start Simple:** Begin with Tic-Tac-Toe before complex games
- **Verify Terminal States:** Ensure win/loss/draw detection works
- **Check Move Generation:** All legal moves must be considered
- **Test Evaluation:** Does your evaluation function make sense?

### Optimization Techniques
- **Move Ordering:** Evaluate promising moves first
- **Transposition Tables:** Cache previously seen positions
- **Iterative Deepening:** Gradually increase search depth
- **Quiescence Search:** Extend search in volatile positions

### Performance Analysis
- **Count Nodes:** Track how many positions evaluated
- **Measure Pruning:** Compare with/without alpha-beta
- **Profile Code:** Find bottlenecks
- **Compare Depths:** How does depth affect play strength?

---

## 🔍 Related Topics to Explore

- **Reinforcement Learning for Games:** Learn through self-play
- **Neural Network Evaluation:** Deep learning for position assessment
- **Opening Books:** Pre-computed opening strategies
- **Endgame Tablebases:** Perfect play in simple positions
- **Parallel Search:** Use multiple processors for faster search
- **Progressive Deepening:** Refine search over time

---

## 🌟 Notable Research Groups

1. **DeepMind** - [deepmind.com](https://www.deepmind.com/)
   - AlphaGo, AlphaZero, and more
   - Cutting-edge game AI research

2. **OpenAI** - [openai.com](https://openai.com/)
   - OpenAI Five (Dota 2)
   - Multi-agent systems

3. **University of Alberta Game Research** - [webdocs.cs.ualberta.ca](https://webdocs.cs.ualberta.ca/~games/)
   - Chinook, poker AI
   - Game theory and search

---

## 🎯 Practical Project Ideas

1. **Chess Engine:** Build a chess AI with minimax and alpha-beta
2. **Connect Four AI:** Implement game with GUI and smart opponent
3. **Othello/Reversi:** Complex enough for interesting AI
4. **Custom Board Game:** Create and implement AI for your own game
5. **AI Tournament:** Have different AIs compete against each other

---

**📌 Remember:** Game-playing AI combines search algorithms with domain knowledge through evaluation functions. The key is balancing search depth with computational efficiency while making intelligent move choices. Start simple with Tic-Tac-Toe and gradually tackle more complex games!
