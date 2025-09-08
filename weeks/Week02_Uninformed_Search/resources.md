# Week 2 Resources: Problem-Solving by Search I (Uninformed Search)

## 📖 Required Readings

### Primary Textbook: AIMA (Russell & Norvig)
- **Chapter 3:** Solving Problems by Searching (sections 3.1-3.4)
  - 3.1: Problem-solving agents
  - 3.2: Example problems
  - 3.3: Search algorithms  
  - 3.4: Uninformed search strategies

### Key Concepts to Focus On
- Search problem formulation
- Tree search vs. graph search
- BFS, DFS, UCS algorithms and properties
- Complexity analysis (time and space)

---

## 🎥 Video Resources

### Foundational Videos
1. **[Introduction to Search Algorithms](https://www.youtube.com/watch?v=dRMvK76xQJI)** - MIT OpenCourseWare (25 min)
   - Comprehensive overview of search problem formulation

2. **[Breadth-First Search (BFS) Explained](https://www.youtube.com/watch?v=oDqjPvD54Ss)** - CS Dojo (11 min)
   - Clear visualization of BFS algorithm

3. **[Depth-First Search (DFS) Explained](https://www.youtube.com/watch?v=7fujbpJ0LB4)** - CS Dojo (12 min)
   - Step-by-step DFS implementation

4. **[Uniform Cost Search Algorithm](https://www.youtube.com/watch?v=dv1m3L6QXWs)** - Zach Star (8 min)
   - UCS with practical examples

### Advanced Videos
5. **[Search Algorithms - Computerphile](https://www.youtube.com/watch?v=6VwqZWygk6E)** (12 min)
   - Deeper dive into search algorithm design principles

6. **[Graph vs Tree Search](https://www.youtube.com/watch?v=TIbUeeksXcI)** - AI Lectures (15 min)
   - Important distinctions between search approaches

---

## 🌐 Interactive Resources

### Online Visualizations
1. **[Pathfinding Visualizer](https://clementmihailescu.github.io/Pathfinding-Visualizer/)**
   - Interactive tool to visualize BFS, DFS, and other algorithms
   - Create obstacles and see how algorithms navigate around them

2. **[Graph Search Visualization](https://www.cs.usfca.edu/~galles/visualization/BFS.html)**
   - Step-through visualizations of BFS and DFS
   - University of San Francisco interactive algorithms

3. **[Search Algorithm Simulator](https://qiao.github.io/PathFinding.js/visual/)**
   - Compare different search strategies on customizable grids
   - Shows step-by-step execution

### Interactive Puzzles
4. **[8-Puzzle Online Solver](https://www.cs.princeton.edu/courses/archive/spr10/cos226/demo/15puzzle/)**
   - Try solving sliding puzzles manually
   - Understand the state space of classic AI problems

5. **[15-Puzzle Game](https://15puzzle.netlify.app/)**
   - Play the classic sliding puzzle game
   - Think about how you would formulate this as a search problem

---

## 💻 Code Resources

### Official AIMA Code
1. **[AIMA Python Repository](https://github.com/aimacode/aima-python)**
   - Chapter 3: [`search.py`](https://github.com/aimacode/aima-python/blob/master/search.py)
   - Reference implementations of all search algorithms
   - Well-documented and tested code

2. **[Search Problem Examples](https://github.com/aimacode/aima-python/blob/master/search.ipynb)**
   - Jupyter notebook with interactive examples
   - Romania map problem, 8-puzzle, and more

### Additional Implementations
3. **[Python Algorithm Implementations](https://github.com/TheAlgorithms/Python/tree/master/graphs)**
   - Community-contributed search algorithm implementations
   - Multiple approaches and optimizations

4. **[GeeksforGeeks Search Algorithms](https://www.geeksforgeeks.org/graph-data-structure-and-algorithms/)**
   - Detailed explanations with code examples
   - Both theory and implementation

---

## 📚 Additional Reading Materials

### Academic Papers (Optional)
1. **"A Formal Basis for the Heuristic Determination of Minimum Cost Paths"** - Hart, Nilsson, Raphael (1968)
   - Classic paper that introduced A* algorithm (preview for next week)
   - [PDF Link](https://ieeexplore.ieee.org/document/4082128)

### Online Tutorials
2. **[Search Algorithms Tutorial - HackerRank](https://www.hackerrank.com/domains/ai/ai-introduction)**
   - Practice problems and explanations
   - Progressive difficulty levels

3. **[MIT 6.034 Search Notes](https://ocw.mit.edu/courses/6-034-artificial-intelligence-fall-2010/)**
   - Comprehensive lecture notes and problem sets
   - Advanced treatment of search algorithms

---

## 🏛️ Classic AI Problems

### Problem Formulation Practice
1. **8-Queens Problem**
   - Place 8 queens on chess board without attacks
   - [Interactive solver](https://eight-queens-puzzle.online/)

2. **Missionary and Cannibals**
   - Classic river-crossing puzzle
   - [Problem description and analysis](https://en.wikipedia.org/wiki/Missionaries_and_cannibals_problem)

3. **Water Jug Problem**
   - Measure exact amounts using jugs of different sizes
   - [Online simulator](https://www.mathsisfun.com/puzzles/water-jugs.html)

4. **Tower of Hanoi**
   - Move disks between pegs following rules
   - [Interactive version](https://www.mathsisfun.com/games/towerofhanoi.html)

---

## 🛠️ Development Tools

### Python Libraries
1. **NetworkX** - Graph manipulation and algorithms
   ```bash
   pip install networkx
   ```
   - [Documentation](https://networkx.org/)
   - Built-in graph search algorithms

2. **Pygame** - For creating search visualizations
   ```bash
   pip install pygame
   ```
   - Create interactive search demonstrations

3. **Matplotlib** - Plotting search trees and graphs
   ```bash
   pip install matplotlib
   ```

### Online Development
4. **[Replit](https://replit.com/)** - Online Python environment
   - No setup required, run code in browser
   - Great for experimenting with algorithms

5. **[Google Colab](https://colab.research.google.com/)** - Free Jupyter notebooks
   - GPU access for larger problems
   - Easy sharing and collaboration

---

## 📊 Datasets and Test Cases

### Graph Datasets
1. **Romania Map** - Classic AIMA example
   - [Graph data](https://github.com/aimacode/aima-python/blob/master/search.py#L50)
   - Good for testing pathfinding algorithms

2. **Small World Networks**
   - [Stanford SNAP datasets](https://snap.stanford.edu/data/)
   - Test scalability of search algorithms

### Puzzle Collections
3. **8-Puzzle Test Cases**
   - Various difficulty levels
   - [15-puzzle datasets](http://korf.cs.ucla.edu/instance.html)

---

## 🎯 Practice Problems

### Beginner Level
1. **[HackerRank: Connected Cells](https://www.hackerrank.com/challenges/connected-cell-in-a-grid)**
2. **[LeetCode: Number of Islands](https://leetcode.com/problems/number-of-islands/)**
3. **[LeetCode: Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)**

### Intermediate Level
4. **[LeetCode: Word Ladder](https://leetcode.com/problems/word-ladder/)**
5. **[LeetCode: Shortest Path in Binary Matrix](https://leetcode.com/problems/shortest-path-in-binary-matrix/)**

### Advanced Level
6. **[LeetCode: Sliding Puzzle](https://leetcode.com/problems/sliding-puzzle/)**

---

## 🔬 Research and Applications

### Real-World Applications
1. **GPS Navigation Systems**
   - How routing algorithms work
   - [OpenStreetMap](https://www.openstreetmap.org/) - Open geographic data

2. **Network Routing**
   - Internet packet routing protocols
   - [BGP and routing algorithms](https://en.wikipedia.org/wiki/Border_Gateway_Protocol)

3. **Game AI**
   - Pathfinding in video games
   - [Pathfinding in games article](https://www.gamasutra.com/view/feature/131505/toward_more_realistic_pathfinding.php)

---

## 📱 Mobile Apps for Learning

1. **Algorithms Explained** (iOS/Android)
   - Interactive algorithm visualizations
   - Step-by-step execution

2. **Graph Theory Visualizer** (Web app)
   - Create and analyze graphs
   - Apply different algorithms

---

## 🏆 Competition Platforms

### Coding Competitions
1. **[Codeforces](https://codeforces.com/)**
   - Graph and search problems
   - Various difficulty levels

2. **[AtCoder](https://atcoder.jp/)**
   - Japanese competitive programming platform
   - Excellent graph problems

3. **[TopCoder](https://www.topcoder.com/)**
   - Algorithm competitions
   - Historical problem archive

---

## ✅ Self-Assessment Resources

### Online Quizzes
1. **[Search Algorithms Quiz - Coursera](https://www.coursera.org/)**
   - Various AI courses with search algorithm modules

2. **[Khan Academy - Algorithms](https://www.khanacademy.org/computing/computer-science/algorithms)**
   - Interactive exercises and quizzes

### Practice Worksheets
3. **[MIT Problem Sets](https://ocw.mit.edu/courses/6-034-artificial-intelligence-fall-2010/pages/assignments/)**
   - Academic-level problems with solutions
   - Test your understanding rigorously

---

## 🔧 Debugging and Optimization

### Common Issues
1. **Infinite Loops in DFS**
   - Always use explored set in graph search
   - Consider depth limiting

2. **Memory Issues with BFS**
   - Monitor frontier size
   - Consider iterative deepening

3. **Incorrect Path Costs**
   - Verify transition model implementation
   - Check priority queue ordering

### Performance Tips
4. **Efficient Data Structures**
   - Use appropriate frontier implementation
   - Optimize state representation

---

## 📝 Study Schedule Suggestion

### Day 1-2: Foundation
- Read AIMA Chapter 3.1-3.2
- Watch introduction videos
- Try online visualizations

### Day 3-4: Algorithms
- Study BFS and DFS in detail
- Implement basic versions
- Practice on simple problems

### Day 5-6: Advanced Topics
- Learn UCS and IDS
- Compare algorithm properties
- Work on harder problems

### Day 7: Review and Practice
- Review all algorithms
- Complete practice problems
- Prepare for next week

---

**💡 Study Tip:** Don't just memorize algorithms - understand when and why to use each one. Practice implementing them from scratch to build intuition about their behavior and complexity.