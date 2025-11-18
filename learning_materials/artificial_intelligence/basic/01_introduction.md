# Artificial Intelligence - Basic Level

## What is Artificial Intelligence?

Artificial Intelligence (AI) is the science of making machines capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding.

### The AI Hierarchy

```
Artificial Intelligence (Broad concept)
    ├── Machine Learning (Learning from data)
    │   └── Deep Learning (Neural networks with many layers)
    ├── Natural Language Processing (Understanding language)
    ├── Computer Vision (Understanding images/video)
    ├── Robotics (Physical interaction with world)
    └── Expert Systems (Rule-based reasoning)
```

## History of AI

### Key Milestones

**1950s**: Alan Turing proposes the Turing Test
- Can a machine exhibit intelligent behavior indistinguishable from humans?

**1956**: Dartmouth Conference - Birth of AI as a field

**1960s-1970s**: Early AI programs
- ELIZA (chatbot)
- SHRDLU (natural language understanding)
- Expert systems

**1980s-1990s**: AI Winter (reduced funding and interest)

**1997**: IBM's Deep Blue defeats world chess champion Garry Kasparov

**2011**: IBM Watson wins Jeopardy!

**2012**: Deep Learning revolution begins (AlexNet wins ImageNet)

**2016**: AlphaGo defeats world Go champion Lee Sedol

**2020s**: Large Language Models (GPT, Claude, ChatGPT)
- Advanced AI assistants
- Multimodal AI (text, images, audio)

## Types of AI

### 1. Based on Capabilities

#### Narrow AI (Weak AI)
- Designed for specific tasks
- Current state of AI
- **Examples**:
  - Spam filters
  - Virtual assistants (Siri, Alexa)
  - Recommendation systems (Netflix, YouTube)
  - Self-driving cars

#### General AI (Strong AI)
- Human-level intelligence across all domains
- Can perform any intellectual task a human can
- **Status**: Theoretical, not yet achieved

#### Super AI
- Surpasses human intelligence
- **Status**: Hypothetical future possibility

### 2. Based on Functionality

#### Reactive Machines
- No memory of past
- Responds to current situations
- **Example**: Chess programs that evaluate current board position

#### Limited Memory
- Uses past experiences for short-term decisions
- **Example**: Self-driving cars tracking other vehicles

#### Theory of Mind
- Understands emotions, beliefs, intentions
- **Status**: Research stage

#### Self-Aware AI
- Conscious, self-aware
- **Status**: Hypothetical

---

## Core AI Concepts

### 1. Intelligent Agents

An agent is anything that:
- Perceives its environment through sensors
- Acts upon the environment through actuators

**Components**:
- **Sensors**: Receive input from environment
- **Actuators**: Perform actions in environment
- **Agent Function**: Maps perceptions to actions

**Example - Vacuum Cleaner Robot**:
- Sensors: Dirt detector, bump sensor
- Actuators: Wheels, vacuum
- Goal: Clean all rooms

### 2. Rationality

A rational agent chooses actions that maximize expected performance.

**Factors**:
- Performance measure (goals)
- Prior knowledge
- Available actions
- Perception history

### 3. Environment Types

#### Observable vs Partially Observable
- **Fully Observable**: Agent can see complete state (chess)
- **Partially Observable**: Limited information (poker)

#### Deterministic vs Stochastic
- **Deterministic**: Next state completely determined (chess)
- **Stochastic**: Randomness involved (dice games)

#### Episodic vs Sequential
- **Episodic**: Independent episodes (spam classification)
- **Sequential**: Current decision affects future (chess)

#### Static vs Dynamic
- **Static**: Environment doesn't change while agent thinks (crossword)
- **Dynamic**: Environment changes (self-driving car)

#### Discrete vs Continuous
- **Discrete**: Finite number of states (chess)
- **Continuous**: Infinite states (driving)

---

## Search Algorithms

Many AI problems can be formulated as search problems: finding a path from initial state to goal state.

### 1. Uninformed Search (Blind Search)

#### Breadth-First Search (BFS)
Explores all nodes at current depth before moving to next depth.

```python
from collections import deque

def bfs(graph, start, goal):
    queue = deque([start])
    visited = {start}
    parent = {start: None}

    while queue:
        node = queue.popleft()

        if node == goal:
            # Reconstruct path
            path = []
            while node is not None:
                path.append(node)
                node = parent[node]
            return path[::-1]

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                queue.append(neighbor)

    return None  # No path found

# Example
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

path = bfs(graph, 'A', 'F')
print(path)  # ['A', 'C', 'F']
```

**Properties**:
- Complete: Always finds solution if exists
- Optimal: Finds shortest path
- Time/Space: O(b^d) where b=branching factor, d=depth

#### Depth-First Search (DFS)
Explores as deep as possible before backtracking.

```python
def dfs(graph, start, goal, visited=None):
    if visited is None:
        visited = set()

    visited.add(start)

    if start == goal:
        return [start]

    for neighbor in graph[start]:
        if neighbor not in visited:
            path = dfs(graph, neighbor, goal, visited)
            if path:
                return [start] + path

    return None

path = dfs(graph, 'A', 'F')
print(path)  # ['A', 'C', 'F'] or another valid path
```

**Properties**:
- Complete: Not guaranteed in infinite spaces
- Not optimal
- Space: O(bm) where m=maximum depth

### 2. Informed Search (Heuristic Search)

Uses domain knowledge to guide search.

#### A* Search
Uses cost so far (g) + estimated cost to goal (h).

```python
import heapq

def a_star(graph, start, goal, heuristic):
    """
    graph: adjacency dict with costs {node: [(neighbor, cost), ...]}
    heuristic: dict {node: estimated_cost_to_goal}
    """
    open_set = [(0, start)]  # (f_score, node)
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic[start]}

    while open_set:
        current_f, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for neighbor, cost in graph[current]:
            tentative_g = g_score[current] + cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic[neighbor]
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None

# Example: Grid pathfinding
graph = {
    'A': [('B', 1), ('C', 3)],
    'B': [('D', 3), ('E', 1)],
    'C': [('F', 2)],
    'D': [],
    'E': [('F', 2)],
    'F': []
}

# Manhattan distance heuristic
heuristic = {'A': 4, 'B': 3, 'C': 2, 'D': 2, 'E': 1, 'F': 0}

path = a_star(graph, 'A', 'F', heuristic)
print(path)
```

**Properties**:
- Complete: Yes (with admissible heuristic)
- Optimal: Yes (with admissible heuristic)
- More efficient than BFS/DFS

**Heuristic Requirements**:
- **Admissible**: Never overestimates (h ≤ actual cost)
- **Consistent**: h(n) ≤ cost(n, n') + h(n')

---

## Game Playing

AI for games like chess, checkers, Go.

### Minimax Algorithm

For two-player zero-sum games.

```python
def minimax(state, depth, is_maximizing):
    """
    state: current game state
    depth: how many moves to look ahead
    is_maximizing: True if maximizing player's turn
    """
    if depth == 0 or is_terminal(state):
        return evaluate(state)

    if is_maximizing:
        max_eval = float('-inf')
        for move in get_possible_moves(state):
            new_state = make_move(state, move)
            eval = minimax(new_state, depth - 1, False)
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for move in get_possible_moves(state):
            new_state = make_move(state, move)
            eval = minimax(new_state, depth - 1, True)
            min_eval = min(min_eval, eval)
        return min_eval

# Tic-Tac-Toe example
def evaluate(board):
    # Return score based on board state
    # +10 for win, -10 for loss, 0 for draw
    pass

def is_terminal(board):
    # Check if game is over
    pass

def get_possible_moves(board):
    # Return list of available moves
    pass

def make_move(board, move):
    # Return new board after move
    pass
```

### Alpha-Beta Pruning

Optimizes minimax by eliminating branches that won't affect final decision.

```python
def minimax_alpha_beta(state, depth, alpha, beta, is_maximizing):
    if depth == 0 or is_terminal(state):
        return evaluate(state)

    if is_maximizing:
        max_eval = float('-inf')
        for move in get_possible_moves(state):
            new_state = make_move(state, move)
            eval = minimax_alpha_beta(new_state, depth - 1, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cutoff
        return max_eval
    else:
        min_eval = float('inf')
        for move in get_possible_moves(state):
            new_state = make_move(state, move)
            eval = minimax_alpha_beta(new_state, depth - 1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cutoff
        return min_eval

# Use with initial alpha=-inf, beta=+inf
best_score = minimax_alpha_beta(initial_state, 5, float('-inf'), float('inf'), True)
```

---

## Knowledge Representation

How AI systems represent and reason about knowledge.

### 1. Logic-Based Representation

#### Propositional Logic
- Simple statements that are true or false
- **Example**: "It is raining" ∧ "I have an umbrella" → "I stay dry"

#### Predicate Logic (First-Order Logic)
- More expressive with variables and quantifiers
- **Example**: ∀x (Human(x) → Mortal(x))

### 2. Semantic Networks
Graph representation of knowledge.

```
Example:
Animal
  ├── Mammal
  │   ├── Dog
  │   └── Cat
  └── Bird
      └── Penguin

Properties:
- Dog has_property "four legs"
- Dog is_a Mammal
- Mammal is_a Animal
```

### 3. Frames
Structured representation of objects.

```python
dog_frame = {
    'is_a': 'mammal',
    'has_legs': 4,
    'can_bark': True,
    'typical_size': 'medium'
}
```

---

## Real-World AI Applications

### 1. Computer Vision
- Facial recognition
- Medical image analysis
- Autonomous vehicles
- Quality control in manufacturing

### 2. Natural Language Processing
- Virtual assistants (Siri, Alexa, Google Assistant)
- Machine translation
- Sentiment analysis
- Chatbots

### 3. Robotics
- Manufacturing robots
- Surgical robots
- Drones
- Warehouse automation

### 4. Expert Systems
- Medical diagnosis
- Financial advising
- Legal research

### 5. Recommendation Systems
- Netflix movie recommendations
- Amazon product suggestions
- Spotify music recommendations

### 6. Autonomous Systems
- Self-driving cars
- Autonomous drones
- Smart homes

---

## AI Ethics and Challenges

### Ethical Concerns
1. **Bias**: AI systems can inherit human biases
2. **Privacy**: Data collection and surveillance
3. **Job Displacement**: Automation replacing human workers
4. **Accountability**: Who is responsible for AI decisions?
5. **Transparency**: "Black box" models are hard to interpret
6. **Safety**: Ensuring AI systems behave as intended

### Current Limitations
1. **Data Requirements**: Most AI needs lots of labeled data
2. **Lack of Common Sense**: AI doesn't understand context like humans
3. **Brittleness**: Small changes can break AI systems
4. **Energy Consumption**: Training large models is resource-intensive
5. **Generalization**: AI struggles with tasks outside training distribution

---

## Getting Started with AI

### Essential Skills
1. **Programming**: Python is most popular
2. **Mathematics**: Linear algebra, calculus, probability
3. **Statistics**: Understanding data and uncertainty
4. **Problem-Solving**: Breaking down complex problems

### Tools and Libraries
```python
# Core libraries
import numpy as np           # Numerical computing
import pandas as pd          # Data manipulation
import matplotlib.pyplot as plt  # Visualization

# Machine Learning
from sklearn import ...      # Scikit-learn

# Deep Learning
import tensorflow as tf      # TensorFlow
import torch                 # PyTorch

# NLP
import nltk                  # Natural Language Toolkit
import spacy                 # Industrial NLP
```

### Learning Path
1. Start with basic Python programming
2. Learn mathematics fundamentals
3. Study machine learning basics
4. Explore specialized areas (NLP, Computer Vision, etc.)
5. Work on projects
6. Read research papers
7. Participate in competitions (Kaggle)

---

## Practice Exercise

**Task**: Implement a simple pathfinding agent

Create an agent that finds the shortest path in a grid world with obstacles.

```python
# Grid: 0 = empty, 1 = obstacle, S = start, G = goal
grid = [
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]

# Implement A* or BFS to find path from (0,0) to (4,4)
```

---

## Key Takeaways

- AI is about creating intelligent machines
- Current AI is "narrow" - specialized for specific tasks
- Search algorithms are fundamental to AI problem-solving
- AI has numerous real-world applications
- Ethical considerations are crucial in AI development
- Mathematics and programming are essential AI skills

---

**Next Steps**: Move to intermediate level to learn about planning, reasoning, and advanced AI techniques.
