# Week 3 Resources: Problem-Solving by Search II (Informed Search)

## 📖 Required Readings

### Primary Textbook: AIMA (Russell & Norvig)
- **Chapter 3:** Solving Problems by Searching
  - Section 3.5: Informed (Heuristic) Search Strategies
  - Section 3.6: Heuristic Functions
- **Chapter 4:** Beyond Classical Search (optional for advanced topics)

---

## 🎥 Video Resources

### Foundational Videos
1. **[A* Search Algorithm](https://www.youtube.com/watch?v=ySN5Wnu88nE)** - Computerphile (9 min)
   - Clear explanation of A* with visual examples
   - Shows how heuristics guide search

2. **[Heuristic Search](https://www.youtube.com/watch?v=dv1m3L6QXWs)** - MIT OpenCourseWare (48 min)
   - Comprehensive lecture on informed search strategies
   - Covers greedy best-first, A*, and heuristic design

3. **[A* Pathfinding for Beginners](https://www.youtube.com/watch?v=-L-WgKMFuhE)** - Sebastian Lague (10 min)
   - Practical walkthrough with game development examples
   - Visualizes how A* explores the search space

### Advanced Videos (Optional)
4. **[Admissible and Consistent Heuristics](https://www.youtube.com/watch?v=zNZBPHyhO3s)** - Gate Lectures (15 min)
   - Deep dive into heuristic properties
   - Mathematical proofs and examples

5. **[Iterative Deepening A* (IDA*)](https://www.youtube.com/watch?v=t75I0dQU_Rg)** - AI Programming (12 min)
   - Memory-efficient variant of A*
   - Combines benefits of IDS and A*

---

## 🌐 Interactive Resources

### Online Demos and Simulations
1. **[PathFinding.js Visualizer](https://qiao.github.io/PathFinding.js/visual/)**
   - Interactive visualization of A*, Dijkstra, BFS, DFS
   - Compare different algorithms side-by-side
   - Create custom mazes and obstacles

2. **[A* Algorithm Visualizer](https://www.redblobgames.com/pathfinding/a-star/introduction.html)**
   - Excellent interactive tutorial by Red Blob Games
   - Step-by-step visualization of A* execution
   - Experiment with different heuristics

3. **[8-Puzzle Solver](https://tristanpenman.com/demos/n-puzzle/)**
   - Solve sliding tile puzzles with A*
   - See the algorithm in action on classic AI problem

### Algorithm Comparisons
4. **[Search Algorithm Comparison Tool](https://clementmihailescu.github.io/Pathfinding-Visualizer/)**
   - Visual comparison of search algorithms
   - Analyze time and space complexity differences

---

## 💻 Code Resources

### Official AIMA Code
1. **[AIMA Python Repository](https://github.com/aimacode/aima-python)**
   - Chapter 3: [`search.py`](https://github.com/aimacode/aima-python/blob/master/search.py)
   - Complete implementations of A*, greedy search, and heuristics
   - Well-documented and ready to use

2. **[Search Algorithms Notebook](https://github.com/aimacode/aima-python/blob/master/search.ipynb)**
   - Jupyter notebook with interactive examples
   - Includes 8-puzzle, route planning, and more

### Additional Implementations
3. **[Python A* Implementation](https://github.com/jrialland/python-astar)**
   - Clean, simple A* library
   - Easy to understand and modify

4. **[Pathfinding Algorithms in Python](https://github.com/brean/python-pathfinding)**
   - Multiple algorithms including A*, Dijkstra, IDA*
   - Grid-based pathfinding implementations

---

## 📚 Additional Reading Materials

### Academic Papers
1. **"A Formal Basis for the Heuristic Determination of Minimum Cost Paths"** - Hart, Nilsson, Raphael (1968)
   - [Original A* paper](https://ieeexplore.ieee.org/document/4082128)
   - Historical significance and theoretical foundations

2. **"Depth-First Iterative-Deepening: An Optimal Admissible Tree Search"** - Korf (1985)
   - [Paper on IDDFS](https://www.aaai.org/ojs/index.php/AITOPICS/article/view/5745)
   - Combines best of depth-first and breadth-first

### Tutorials and Guides
3. **[Introduction to A* by Red Blob Games](https://www.redblobgames.com/pathfinding/a-star/introduction.html)**
   - Best interactive tutorial on A* algorithm
   - Clear explanations with visual examples

4. **[Heuristic Design Patterns](http://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html)**
   - Amit's guide to creating effective heuristics
   - Practical advice for game development

### Blog Posts and Articles
5. **[Understanding A* Algorithm](https://www.baeldung.com/cs/a-star-algorithm)** - Baeldung
   - Step-by-step explanation with examples
   - Includes pseudocode and complexity analysis

6. **[Optimizing A* for Games](https://gamedevelopment.tutsplus.com/tutorials/how-to-speed-up-a-pathfinding--gamedev-1483)**
   - Performance optimization techniques
   - Trade-offs between optimality and speed

---

## 📊 Datasets and Problem Sets

### Classic AI Problems
1. **[8-Puzzle Problem Instances](http://www.cs.Princeton.edu/courses/archive/fall06/cos402/assignments/8puzzle/)**
   - Collection of puzzle configurations
   - Various difficulty levels

2. **[Grid World Problems](https://github.com/openai/gym/tree/master/gym/envs/toy_text)**
   - OpenAI Gym grid world environments
   - Perfect for testing pathfinding algorithms

### Real-World Applications
3. **[OpenStreetMap Data](https://www.openstreetmap.org/)**
   - Real map data for route planning
   - Use with A* for GPS-like navigation

4. **[Maze Datasets](https://github.com/razimantv/mazegenerator)**
   - Generated mazes for pathfinding testing
   - Various sizes and complexities

---

## 🔧 Tools and Libraries

### Python Libraries
1. **NetworkX** - Graph library for Python
   - `pip install networkx`
   - Includes A* implementation: `nx.astar_path()`
   - [Documentation](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.astar.astar_path.html)

2. **pathfinding** - Pathfinding algorithms library
   - `pip install pathfinding`
   - Multiple algorithms ready to use
   - [GitHub](https://github.com/brean/python-pathfinding)

### Visualization Tools
3. **Matplotlib** - For visualizing search spaces
   - `pip install matplotlib`
   - Create heatmaps and path visualizations

4. **Pygame** - For interactive visualizations
   - `pip install pygame`
   - Build custom search visualizers

---

## 🎮 Practical Applications

### Game Development
1. **[Unity Pathfinding Tutorial](https://learn.unity.com/tutorial/pathfinding)**
   - A* in game development context
   - Real-world application examples

2. **[Godot Navigation Tutorial](https://docs.godotengine.org/en/stable/tutorials/navigation/navigation_introduction_2d.html)**
   - Pathfinding in Godot game engine

### Robotics
3. **[ROS Navigation Stack](http://wiki.ros.org/navigation)**
   - A* and variants in robotics
   - Real robot path planning

4. **[Robot Path Planning Overview](https://github.com/zhm-real/PathPlanning)**
   - Collection of pathfinding algorithms
   - Includes visualizations and comparisons

---

## 📖 Books (Supplementary)

1. **"The Algorithm Design Manual"** - Steven Skiena
   - Chapter on graph algorithms and heuristics
   - Practical problem-solving approach

2. **"Introduction to Algorithms"** - Cormen, Leiserson, Rivest, Stein
   - Chapter 24: Single-Source Shortest Paths
   - Theoretical foundations

3. **"Artificial Intelligence for Games"** - Millington & Funge
   - Chapter on pathfinding and movement
   - Game-specific optimizations

---

## 🌟 Notable Research Groups and Labs

1. **Stanford AI Lab** - [ai.stanford.edu](https://ai.stanford.edu/)
   - Research on search and planning

2. **Berkeley AI Research (BAIR)** - [bair.berkeley.edu](https://bair.berkeley.edu/)
   - Home of AIMA textbook authors

3. **MIT CSAIL** - [csail.mit.edu](https://www.csail.mit.edu/)
   - Planning and decision-making research

---

## 💡 Study Tips and Best Practices

### Understanding Heuristics
- **Start Simple:** Begin with Manhattan distance and Euclidean distance
- **Visualize:** Draw out how heuristics guide the search
- **Experiment:** Try different heuristics on the same problem
- **Prove Properties:** Verify admissibility and consistency mathematically

### Debugging A*
- **Track the Open List:** Monitor which nodes are being explored
- **Verify f-values:** Check f(n) = g(n) + h(n) calculations
- **Test on Simple Cases:** Start with small grids before complex problems
- **Compare with Dijkstra:** A* with h=0 should match Dijkstra's algorithm

### Performance Optimization
- **Use Priority Queues:** Essential for efficient implementation
- **Consider Memory:** A* can use lots of memory; try IDA* for large spaces
- **Profile Your Code:** Identify bottlenecks in your implementation
- **Learn from Games:** Game developers have optimized A* extensively

---

## 🔍 Related Topics to Explore

- **Bidirectional Search:** Search from both start and goal
- **Jump Point Search:** Fast pathfinding on uniform grids
- **Theta*:** Any-angle pathfinding variant of A*
- **Lifelong Planning A* (LPA*):** Efficient replanning for dynamic environments
- **D* and D* Lite:** Dynamic A* for changing environments (used in Mars rovers!)

---

**📌 Remember:** The key to mastering informed search is understanding how heuristics guide the search and practicing with various problems. Don't just implement algorithms—understand why they work and when to use each one!
