# Artificial Intelligence - Practice Questions

## Easy Difficulty

### Question 1
**What is Artificial Intelligence?**

**Answer:** The science of making machines capable of performing tasks that typically require human intelligence, such as learning, reasoning, problem-solving, and perception.

---

### Question 2
**What is the Turing Test?**

**Answer:** A test proposed by Alan Turing to determine if a machine can exhibit intelligent behavior indistinguishable from a human. If a human evaluator cannot tell whether they're conversing with a machine or human, the machine passes the test.

---

### Question 3
**What is the difference between Narrow AI and General AI?**

**Answer:**
- **Narrow AI (Weak AI):** Designed for specific tasks (e.g., chess, spam detection, voice assistants). Current state of AI.
- **General AI (Strong AI):** Human-level intelligence across all domains. Not yet achieved.

---

### Question 4
**Which search algorithm guarantees finding the shortest path?**

A) Depth-First Search
B) Breadth-First Search
C) Greedy Search
D) Random Search

**Answer:** B (Breadth-First Search)

---

### Question 5
**What is an intelligent agent in AI?**

**Answer:** Anything that perceives its environment through sensors and acts upon that environment through actuators to achieve its goals.

---

### Question 6
**What is heuristic in AI?**

**Answer:** A rule of thumb or educated guess that helps solve problems faster by guiding the search toward promising solutions, though it doesn't guarantee optimal solutions.

---

### Question 7
**What is the difference between DFS and BFS?**

**Answer:**
- **DFS (Depth-First Search):** Explores as deep as possible before backtracking. Uses stack (LIFO).
- **BFS (Breadth-First Search):** Explores all neighbors at current depth before moving deeper. Uses queue (FIFO).

---

### Question 8
**What is A* search algorithm?**

**Answer:** An informed search algorithm that finds the shortest path using: f(n) = g(n) + h(n), where g(n) is cost from start, h(n) is estimated cost to goal.

---

### Question 9
**What is the minimax algorithm used for?**

**Answer:** Game playing in two-player zero-sum games. It minimizes the maximum possible loss by assuming the opponent plays optimally.

---

### Question 10
**What is reinforcement learning?**

**Answer:** Learning through trial and error by receiving rewards or penalties for actions, gradually learning optimal behavior.

---

## Medium Difficulty

### Question 1
**Explain the A* search algorithm and its properties. What makes a heuristic admissible?**

**Answer:**

**A* Algorithm:**
- Combines benefits of uniform-cost search and greedy best-first search
- Uses evaluation function: f(n) = g(n) + h(n)
  - g(n): Actual cost from start to node n
  - h(n): Estimated cost from n to goal (heuristic)

**Algorithm:**
```python
def a_star(start, goal, heuristic):
    open_set = PriorityQueue()
    open_set.put((heuristic(start, goal), start))

    came_from = {}
    g_score = {start: 0}

    while not open_set.empty():
        current = open_set.get()[1]

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in get_neighbors(current):
            tentative_g = g_score[current] + cost(current, neighbor)

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                open_set.put((f_score, neighbor))

    return None  # No path found
```

**Properties:**

**Admissible Heuristic:**
- Never overestimates actual cost to goal
- h(n) ≤ h*(n) where h*(n) is true cost
- Guarantees A* finds optimal solution

**Consistent (Monotonic) Heuristic:**
- h(n) ≤ cost(n, n') + h(n') for all n, n'
- Triangle inequality
- Implies admissibility

**Example Heuristics:**

**Manhattan Distance (Grid):**
```python
def manhattan(pos, goal):
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
```

**Euclidean Distance:**
```python
def euclidean(pos, goal):
    return sqrt((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)
```

**Complexity:**
- Time: O(b^d) where b = branching factor, d = depth
- Space: O(b^d)
- With good heuristic: Much better in practice

---

### Question 2
**Implement the minimax algorithm with alpha-beta pruning for a game tree.**

**Answer:**

```python
def minimax_alpha_beta(state, depth, alpha, beta, is_maximizing):
    """
    Minimax with alpha-beta pruning

    Args:
        state: Current game state
        depth: How deep to search
        alpha: Best value for maximizer
        beta: Best value for minimizer
        is_maximizing: True if maximizing player's turn

    Returns:
        Best score for current player
    """
    # Base case: terminal state or depth limit
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

def best_move(state, depth):
    """Find best move for current player"""
    best_score = float('-inf')
    best_move = None

    for move in get_possible_moves(state):
        new_state = make_move(state, move)
        score = minimax_alpha_beta(new_state, depth - 1, float('-inf'), float('inf'), False)

        if score > best_score:
            best_score = score
            best_move = move

    return best_move

# Example: Tic-Tac-Toe
class TicTacToe:
    def __init__(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'

    def is_terminal(self):
        return self.check_winner() is not None or self.is_full()

    def check_winner(self):
        # Check rows
        for row in self.board:
            if row[0] == row[1] == row[2] != ' ':
                return row[0]

        # Check columns
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] != ' ':
                return self.board[0][col]

        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ':
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
            return self.board[0][2]

        return None

    def is_full(self):
        return all(cell != ' ' for row in self.board for cell in row)

    def evaluate(self):
        winner = self.check_winner()
        if winner == 'X':
            return 1
        elif winner == 'O':
            return -1
        else:
            return 0

    def get_possible_moves(self):
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == ' ':
                    moves.append((i, j))
        return moves

    def make_move(self, move, player):
        new_game = TicTacToe()
        new_game.board = [row[:] for row in self.board]
        new_game.board[move[0]][move[1]] = player
        return new_game
```

**Alpha-Beta Pruning Benefits:**
- Reduces nodes evaluated
- Best case: O(b^(d/2)) instead of O(b^d)
- No effect on final result
- More effective with good move ordering

---

### Question 3
**Explain Constraint Satisfaction Problems (CSP) and implement backtracking with forward checking.**

**Answer:**

**CSP Components:**
- **Variables:** X = {X₁, X₂, ..., Xₙ}
- **Domains:** D = {D₁, D₂, ..., Dₙ}
- **Constraints:** C restricts value combinations

**Example: Map Coloring**
```python
class CSP:
    def __init__(self, variables, domains, constraints):
        self.variables = variables
        self.domains = domains
        self.constraints = constraints

    def is_consistent(self, var, value, assignment):
        """Check if assignment is consistent with constraints"""
        for constraint in self.constraints:
            if var in constraint:
                other_var = constraint[0] if constraint[1] == var else constraint[1]
                if other_var in assignment:
                    if assignment[other_var] == value:
                        return False
        return True

    def forward_checking(self, var, value, assignment, domains):
        """Remove inconsistent values from neighbor domains"""
        for constraint in self.constraints:
            if var in constraint:
                other_var = constraint[0] if constraint[1] == var else constraint[1]
                if other_var not in assignment:
                    if value in domains[other_var]:
                        domains[other_var] = domains[other_var].copy()
                        domains[other_var].remove(value)
                        if not domains[other_var]:
                            return False  # Domain wipeout
        return True

    def select_unassigned_variable(self, assignment, domains):
        """MRV heuristic: Choose variable with fewest legal values"""
        unassigned = [v for v in self.variables if v not in assignment]
        return min(unassigned, key=lambda v: len(domains[v]))

    def order_domain_values(self, var, assignment, domains):
        """Least constraining value heuristic"""
        def count_conflicts(value):
            conflicts = 0
            for constraint in self.constraints:
                if var in constraint:
                    other_var = constraint[0] if constraint[1] == var else constraint[1]
                    if other_var not in assignment and value in domains[other_var]:
                        conflicts += 1
            return conflicts

        return sorted(domains[var], key=count_conflicts)

    def backtrack(self, assignment, domains):
        """Backtracking search with forward checking"""
        if len(assignment) == len(self.variables):
            return assignment

        var = self.select_unassigned_variable(assignment, domains)

        for value in self.order_domain_values(var, assignment, domains):
            if self.is_consistent(var, value, assignment):
                assignment[var] = value

                # Save current domains
                saved_domains = {v: d.copy() for v, d in domains.items()}

                # Forward checking
                if self.forward_checking(var, value, assignment, domains):
                    result = self.backtrack(assignment, domains)
                    if result is not None:
                        return result

                # Restore domains
                domains.update(saved_domains)
                del assignment[var]

        return None

# Example: Map Coloring
variables = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']
domains = {var: ['red', 'green', 'blue'] for var in variables}
constraints = [
    ('WA', 'NT'), ('WA', 'SA'),
    ('NT', 'SA'), ('NT', 'Q'),
    ('SA', 'Q'), ('SA', 'NSW'), ('SA', 'V'),
    ('Q', 'NSW'),
    ('NSW', 'V')
]

csp = CSP(variables, domains, constraints)
solution = csp.backtrack({}, domains)
print(solution)
```

**Improvements:**
- **MRV (Minimum Remaining Values):** Choose variable with fewest legal values
- **Degree Heuristic:** Choose variable involved in most constraints
- **LCV (Least Constraining Value):** Prefer values that rule out fewest choices for neighbors
- **Arc Consistency (AC-3):** More powerful than forward checking

---

### Question 4
**Explain Q-Learning and implement it for a simple grid world problem.**

**Answer:**

**Q-Learning:** Model-free reinforcement learning algorithm.

**Key Concepts:**
- **Q-value:** Q(s, a) = Expected future reward for taking action a in state s
- **Goal:** Learn optimal Q-values
- **Update Rule:**
  ```
  Q(s, a) ← Q(s, a) + α[r + γ max Q(s', a') - Q(s, a)]
  ```

**Components:**
- α: Learning rate
- γ: Discount factor
- ε: Exploration rate (ε-greedy)

**Implementation:**

```python
import numpy as np
import random

class QLearningAgent:
    def __init__(self, states, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.states = states
        self.actions = actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

        # Initialize Q-table
        self.Q = {}
        for state in states:
            self.Q[state] = {}
            for action in actions:
                self.Q[state][action] = 0.0

    def choose_action(self, state):
        """ε-greedy action selection"""
        if random.random() < self.epsilon:
            return random.choice(self.actions)  # Explore
        else:
            return self.get_best_action(state)  # Exploit

    def get_best_action(self, state):
        """Get action with highest Q-value"""
        return max(self.actions, key=lambda a: self.Q[state][a])

    def update(self, state, action, reward, next_state):
        """Q-learning update"""
        max_next_q = max(self.Q[next_state].values())
        current_q = self.Q[state][action]

        # Q-learning formula
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.Q[state][action] = new_q

    def decay_epsilon(self, decay_rate=0.995, min_epsilon=0.01):
        """Decay exploration rate"""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.goal = (size-1, size-1)
        self.obstacles = [(1, 1), (2, 2), (3, 1)]
        self.reset()

    def reset(self):
        self.position = (0, 0)
        return self.position

    def step(self, action):
        """Take action and return (next_state, reward, done)"""
        x, y = self.position

        # Actions: 0=up, 1=right, 2=down, 3=left
        if action == 0:  # Up
            x = max(0, x - 1)
        elif action == 1:  # Right
            y = min(self.size - 1, y + 1)
        elif action == 2:  # Down
            x = min(self.size - 1, x + 1)
        elif action == 3:  # Left
            y = max(0, y - 1)

        next_position = (x, y)

        # Check if valid move
        if next_position in self.obstacles:
            next_position = self.position  # Stay in place
            reward = -1
        elif next_position == self.goal:
            reward = 100
        else:
            reward = -0.1  # Small penalty for each step

        self.position = next_position
        done = (self.position == self.goal)

        return self.position, reward, done

    def get_states(self):
        states = []
        for x in range(self.size):
            for y in range(self.size):
                if (x, y) not in self.obstacles:
                    states.append((x, y))
        return states

# Training
env = GridWorld(size=5)
states = env.get_states()
actions = [0, 1, 2, 3]  # up, right, down, left

agent = QLearningAgent(states, actions, alpha=0.1, gamma=0.9, epsilon=0.2)

# Train for many episodes
num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    steps = 0

    while steps < 100:  # Max steps per episode
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)

        agent.update(state, action, reward, next_state)

        state = next_state
        total_reward += reward
        steps += 1

        if done:
            break

    agent.decay_epsilon()

    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Steps: {steps}")

# Test learned policy
print("\nLearned Policy:")
state = env.reset()
path = [state]

for _ in range(20):
    action = agent.get_best_action(state)
    next_state, reward, done = env.step(action)
    path.append(next_state)
    state = next_state
    if done:
        break

print(f"Path: {path}")
```

---

### Question 5
**Explain Bayesian Networks and implement inference using variable elimination.**

**Answer:**

**Bayesian Network:** Directed acyclic graph representing probabilistic dependencies.

**Components:**
- **Nodes:** Random variables
- **Edges:** Direct dependencies
- **CPDs:** Conditional Probability Distributions

**Example:**
```
Rain → Sprinkler → WetGrass
Rain → WetGrass
```

**Implementation:**

```python
import numpy as np
from itertools import product

class BayesianNetwork:
    def __init__(self):
        self.variables = {}
        self.cpds = {}
        self.parents = {}

    def add_variable(self, name, values):
        self.variables[name] = values
        self.parents[name] = []

    def add_cpd(self, variable, parents, cpd):
        """
        Add conditional probability distribution
        cpd: dictionary mapping parent values to probabilities
        """
        self.parents[variable] = parents
        self.cpds[variable] = cpd

    def get_probability(self, variable, value, parent_values):
        """Get P(variable=value | parents)"""
        if not self.parents[variable]:
            return self.cpds[variable][value]

        parent_tuple = tuple(parent_values[p] for p in self.parents[variable])
        return self.cpds[variable][parent_tuple][value]

    def enumerate_all(self, query_var, evidence):
        """Exact inference by enumeration"""
        hidden_vars = [v for v in self.variables if v not in evidence and v != query_var]

        result = {}
        for query_value in self.variables[query_var]:
            prob = 0

            # Sum over all possible assignments to hidden variables
            for hidden_assignment in product(*[self.variables[v] for v in hidden_vars]):
                hidden_dict = dict(zip(hidden_vars, hidden_assignment))
                full_assignment = {**evidence, **hidden_dict, query_var: query_value}

                # Compute joint probability
                joint_prob = 1.0
                for var in self.variables:
                    parent_vals = {p: full_assignment[p] for p in self.parents[var]}
                    joint_prob *= self.get_probability(var, full_assignment[var], parent_vals)

                prob += joint_prob

            result[query_value] = prob

        # Normalize
        total = sum(result.values())
        return {k: v/total for k, v in result.items()}

# Example: Rain → Sprinkler → WetGrass
bn = BayesianNetwork()

# Add variables
bn.add_variable('Rain', [False, True])
bn.add_variable('Sprinkler', [False, True])
bn.add_variable('WetGrass', [False, True])

# Add CPDs
# P(Rain)
bn.add_cpd('Rain', [], {
    False: 0.8,
    True: 0.2
})

# P(Sprinkler | Rain)
bn.add_cpd('Sprinkler', ['Rain'], {
    (False,): {False: 0.4, True: 0.6},
    (True,): {False: 0.99, True: 0.01}
})

# P(WetGrass | Sprinkler, Rain)
bn.add_cpd('WetGrass', ['Sprinkler', 'Rain'], {
    (False, False): {False: 1.0, True: 0.0},
    (False, True): {False: 0.1, True: 0.9},
    (True, False): {False: 0.1, True: 0.9},
    (True, True): {False: 0.01, True: 0.99}
})

# Query: P(Rain | WetGrass=True)
evidence = {'WetGrass': True}
result = bn.enumerate_all('Rain', evidence)
print(f"P(Rain | WetGrass=True) = {result}")
```

---

## Hard Difficulty

### Question 1
**Implement Monte Carlo Tree Search (MCTS) for game playing.**

**Answer:** See AI Advanced section (01_advanced_ai.md) for complete MCTS implementation.

---

### Question 2
**Design a multi-agent system with negotiation protocols.**

**Answer:** See AI Intermediate section for multi-agent system design and implementation.

---

### Question 3
**Implement probabilistic programming with Pyro for Bayesian inference.**

**Answer:** See AI Advanced section for probabilistic programming implementations.

---

### Question 4
**Create a causal inference system using do-calculus.**

**Answer:** See AI Advanced section for causal inference and structural causal models.

---

### Question 5
**Implement meta-learning with MAML (Model-Agnostic Meta-Learning).**

**Answer:** See AI Advanced section for complete MAML implementation and explanation.
