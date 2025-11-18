# Artificial Intelligence - Intermediate Level

## Planning

Planning is the process of finding a sequence of actions to achieve a goal.

### STRIPS Planning

**STRIPS** (Stanford Research Institute Problem Solver) is a classical planning formalism.

**Components**:
- **State**: Set of propositions (facts)
- **Actions**: Have preconditions and effects
- **Goal**: Desired state

**Example - Blocks World**:
```python
# Initial State
state = {
    'on(A, Table)',
    'on(B, Table)',
    'on(C, A)',
    'clear(B)',
    'clear(C)',
    'empty(hand)'
}

# Goal
goal = {
    'on(A, B)',
    'on(B, C)'
}

# Actions
actions = {
    'pickup': {
        'preconditions': ['on(X, Table)', 'clear(X)', 'empty(hand)'],
        'effects': ['holding(X)', '¬on(X, Table)', '¬clear(X)', '¬empty(hand)']
    },
    'putdown': {
        'preconditions': ['holding(X)'],
        'effects': ['on(X, Table)', 'clear(X)', 'empty(hand)', '¬holding(X)']
    },
    'stack': {
        'preconditions': ['holding(X)', 'clear(Y)'],
        'effects': ['on(X, Y)', 'clear(X)', 'empty(hand)', '¬holding(X)', '¬clear(Y)']
    }
}
```

### Forward State-Space Search

```python
from collections import deque

class PlanningProblem:
    def __init__(self, initial_state, goal_state, actions):
        self.initial = initial_state
        self.goal = goal_state
        self.actions = actions

    def is_goal(self, state):
        return all(prop in state for prop in self.goal)

    def get_successors(self, state):
        successors = []
        for action_name, action in self.actions.items():
            if self.is_applicable(state, action):
                new_state = self.apply_action(state, action)
                successors.append((new_state, action_name))
        return successors

    def is_applicable(self, state, action):
        return all(pre in state for pre in action['preconditions'])

    def apply_action(self, state, action):
        new_state = state.copy()
        for effect in action['effects']:
            if effect.startswith('¬'):
                new_state.discard(effect[1:])
            else:
                new_state.add(effect)
        return new_state

def forward_search(problem):
    queue = deque([(problem.initial, [])])
    visited = {frozenset(problem.initial)}

    while queue:
        state, plan = queue.popleft()

        if problem.is_goal(state):
            return plan

        for new_state, action in problem.get_successors(state):
            state_frozen = frozenset(new_state)
            if state_frozen not in visited:
                visited.add(state_frozen)
                queue.append((new_state, plan + [action]))

    return None
```

### Planning Graphs

**GraphPlan** algorithm uses planning graphs to find valid plans efficiently.

**Layers**:
- **Proposition Layer**: States at level i
- **Action Layer**: Applicable actions at level i
- **Mutex Relations**: Incompatible actions/propositions

---

## Constraint Satisfaction Problems (CSP)

Problems defined by variables, domains, and constraints.

### CSP Components

```python
# Example: Map Coloring
# Color map regions so no adjacent regions have same color

variables = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']
domains = {var: ['red', 'green', 'blue'] for var in variables}
constraints = [
    ('WA', 'NT'), ('WA', 'SA'),
    ('NT', 'SA'), ('NT', 'Q'),
    ('SA', 'Q'), ('SA', 'NSW'), ('SA', 'V'),
    ('Q', 'NSW'),
    ('NSW', 'V')
]

def is_consistent(var, value, assignment, constraints):
    for neighbor in get_neighbors(var, constraints):
        if neighbor in assignment and assignment[neighbor] == value:
            return False
    return True
```

### Backtracking Search

```python
def backtracking_search(csp):
    return backtrack({}, csp)

def backtrack(assignment, csp):
    if len(assignment) == len(csp.variables):
        return assignment

    var = select_unassigned_variable(assignment, csp)

    for value in order_domain_values(var, assignment, csp):
        if is_consistent(var, value, assignment, csp):
            assignment[var] = value

            result = backtrack(assignment, csp)
            if result is not None:
                return result

            del assignment[var]

    return None

def select_unassigned_variable(assignment, csp):
    # Minimum Remaining Values (MRV) heuristic
    unassigned = [v for v in csp.variables if v not in assignment]
    return min(unassigned, key=lambda v: count_legal_values(v, assignment, csp))

def order_domain_values(var, assignment, csp):
    # Least Constraining Value heuristic
    return sorted(
        csp.domains[var],
        key=lambda val: count_conflicts(var, val, assignment, csp)
    )
```

### Constraint Propagation

#### Forward Checking
After assigning variable, remove inconsistent values from neighbors' domains.

```python
def forward_checking(var, value, assignment, csp):
    for neighbor in get_neighbors(var, csp.constraints):
        if neighbor not in assignment:
            for val in csp.domains[neighbor]:
                if not is_consistent(neighbor, val, assignment):
                    csp.domains[neighbor].remove(val)
                    if not csp.domains[neighbor]:
                        return False  # Domain wipeout
    return True
```

#### Arc Consistency (AC-3)
```python
from collections import deque

def ac3(csp):
    queue = deque([(Xi, Xj) for Xi, Xj in csp.constraints])

    while queue:
        Xi, Xj = queue.popleft()

        if revise(csp, Xi, Xj):
            if not csp.domains[Xi]:
                return False

            for Xk in get_neighbors(Xi, csp.constraints):
                if Xk != Xj:
                    queue.append((Xk, Xi))

    return True

def revise(csp, Xi, Xj):
    revised = False
    for x in csp.domains[Xi]:
        if not any(is_consistent_value(x, y) for y in csp.domains[Xj]):
            csp.domains[Xi].remove(x)
            revised = True
    return revised
```

---

## Reasoning Under Uncertainty

### Probability Review

**Bayes' Theorem**:
```
P(A|B) = P(B|A) × P(A) / P(B)
```

**Example - Medical Diagnosis**:
```
P(Disease|Positive Test) =
    P(Positive Test|Disease) × P(Disease) / P(Positive Test)
```

### Bayesian Networks

Directed acyclic graph representing probabilistic dependencies.

```python
import pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define structure
model = BayesianNetwork([
    ('Rain', 'Sprinkler'),
    ('Rain', 'WetGrass'),
    ('Sprinkler', 'WetGrass')
])

# Define CPDs (Conditional Probability Distributions)
cpd_rain = TabularCPD(
    variable='Rain',
    variable_card=2,
    values=[[0.8], [0.2]]  # P(Rain=False), P(Rain=True)
)

cpd_sprinkler = TabularCPD(
    variable='Sprinkler',
    variable_card=2,
    values=[
        [0.6, 0.99],  # P(Sprinkler=False | Rain)
        [0.4, 0.01]   # P(Sprinkler=True | Rain)
    ],
    evidence=['Rain'],
    evidence_card=[2]
)

cpd_wetgrass = TabularCPD(
    variable='WetGrass',
    variable_card=2,
    values=[
        [1.0, 0.1, 0.1, 0.01],  # P(WetGrass=False | Sprinkler, Rain)
        [0.0, 0.9, 0.9, 0.99]   # P(WetGrass=True | Sprinkler, Rain)
    ],
    evidence=['Sprinkler', 'Rain'],
    evidence_card=[2, 2]
)

# Add CPDs to model
model.add_cpds(cpd_rain, cpd_sprinkler, cpd_wetgrass)

# Check model validity
assert model.check_model()

# Inference
inference = VariableElimination(model)

# Query: P(Rain | WetGrass=True)
result = inference.query(variables=['Rain'], evidence={'WetGrass': 1})
print(result)
```

### Hidden Markov Models (HMM)

Model for sequential data with hidden states.

```python
from hmmlearn import hmm
import numpy as np

# Define HMM
model = hmm.GaussianHMM(n_components=3, covariance_type="full")

# Training data
X = np.concatenate([
    np.random.randn(100, 1),
    np.random.randn(100, 1) + 5,
    np.random.randn(100, 1) - 5
])

# Train
model.fit(X)

# Predict hidden states
hidden_states = model.predict(X)

# Decode most likely sequence
log_prob, states = model.decode(X)

# Predict next observation
samples, states = model.sample(10)
```

---

## Multi-Agent Systems

Systems with multiple autonomous agents interacting.

### Agent Communication

**FIPA ACL (Agent Communication Language)**:
- **Inform**: Share information
- **Request**: Ask agent to perform action
- **Query**: Ask for information
- **Propose**: Suggest action
- **Accept/Reject**: Respond to proposals

### Cooperation Strategies

#### Centralized Planning
Single planner coordinates all agents.

#### Distributed Planning
Each agent plans independently and coordinates.

```python
class Agent:
    def __init__(self, agent_id, goals):
        self.id = agent_id
        self.goals = goals
        self.knowledge = {}

    def communicate(self, other_agent, message):
        other_agent.receive(message)

    def receive(self, message):
        # Process incoming message
        if message['type'] == 'inform':
            self.knowledge.update(message['content'])
        elif message['type'] == 'request':
            self.handle_request(message['content'])

    def negotiate(self, other_agent, resource):
        # Negotiation protocol
        pass
```

### Game Theory

#### Nash Equilibrium
Strategy profile where no player can improve by changing strategy alone.

```python
import numpy as np

# Prisoner's Dilemma
payoff_matrix = np.array([
    [(3, 3), (0, 5)],  # Cooperate
    [(5, 0), (1, 1)]   # Defect
])

# Find Nash Equilibrium
# (Defect, Defect) is Nash Equilibrium
```

---

## Reinforcement Learning

Learning through interaction with environment.

### Markov Decision Process (MDP)

**Components**:
- States (S)
- Actions (A)
- Transition function: P(s'|s,a)
- Reward function: R(s,a,s')
- Discount factor: γ

### Value Iteration

```python
import numpy as np

def value_iteration(states, actions, transitions, rewards, gamma=0.9, theta=0.01):
    """
    states: list of states
    actions: list of actions
    transitions: P[s][a][s'] = probability
    rewards: R[s][a][s'] = reward
    gamma: discount factor
    theta: convergence threshold
    """
    V = {s: 0 for s in states}

    while True:
        delta = 0
        for s in states:
            v = V[s]

            # Bellman update
            V[s] = max(
                sum(
                    transitions[s][a][s_prime] *
                    (rewards[s][a][s_prime] + gamma * V[s_prime])
                    for s_prime in states
                )
                for a in actions
            )

            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    # Extract policy
    policy = {}
    for s in states:
        policy[s] = max(
            actions,
            key=lambda a: sum(
                transitions[s][a][s_prime] *
                (rewards[s][a][s_prime] + gamma * V[s_prime])
                for s_prime in states
            )
        )

    return V, policy
```

### Q-Learning

Model-free reinforcement learning.

```python
import numpy as np
import random

class QLearningAgent:
    def __init__(self, states, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.Q = {s: {a: 0 for a in actions} for s in states}
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.actions = actions

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)  # Explore
        else:
            return max(self.actions, key=lambda a: self.Q[state][a])  # Exploit

    def update(self, state, action, reward, next_state):
        max_next_q = max(self.Q[next_state].values())
        self.Q[state][action] += self.alpha * (
            reward + self.gamma * max_next_q - self.Q[state][action]
        )

# Training loop
agent = QLearningAgent(states, actions)

for episode in range(1000):
    state = initial_state

    while not is_terminal(state):
        action = agent.choose_action(state)
        next_state, reward = environment.step(state, action)
        agent.update(state, action, reward, next_state)
        state = next_state
```

---

## Genetic Algorithms

Evolutionary approach to optimization.

```python
import random

class GeneticAlgorithm:
    def __init__(self, population_size, gene_length, fitness_fn):
        self.population_size = population_size
        self.gene_length = gene_length
        self.fitness_fn = fitness_fn

    def initialize_population(self):
        return [
            [random.randint(0, 1) for _ in range(self.gene_length)]
            for _ in range(self.population_size)
        ]

    def selection(self, population):
        # Tournament selection
        tournament = random.sample(population, k=3)
        return max(tournament, key=self.fitness_fn)

    def crossover(self, parent1, parent2):
        # Single-point crossover
        point = random.randint(1, self.gene_length - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def mutate(self, individual, mutation_rate=0.01):
        return [
            gene if random.random() > mutation_rate else 1 - gene
            for gene in individual
        ]

    def evolve(self, generations):
        population = self.initialize_population()

        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = [(ind, self.fitness_fn(ind)) for ind in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)

            # Print best individual
            best = fitness_scores[0]
            print(f"Generation {generation}: Best fitness = {best[1]}")

            # Create new population
            new_population = [best[0]]  # Elitism

            while len(new_population) < self.population_size:
                parent1 = self.selection(population)
                parent2 = self.selection(population)

                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                new_population.extend([child1, child2])

            population = new_population[:self.population_size]

        return max(population, key=self.fitness_fn)

# Example: Maximize number of 1s
def fitness(individual):
    return sum(individual)

ga = GeneticAlgorithm(population_size=100, gene_length=20, fitness_fn=fitness)
best = ga.evolve(generations=50)
print(f"Best solution: {best}")
```

---

## Fuzzy Logic

Reasoning with imprecise information.

```python
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define fuzzy variables
temperature = ctrl.Antecedent(np.arange(0, 101, 1), 'temperature')
humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
fan_speed = ctrl.Consequent(np.arange(0, 101, 1), 'fan_speed')

# Membership functions
temperature['cold'] = fuzz.trimf(temperature.universe, [0, 0, 50])
temperature['warm'] = fuzz.trimf(temperature.universe, [0, 50, 100])
temperature['hot'] = fuzz.trimf(temperature.universe, [50, 100, 100])

humidity['low'] = fuzz.trimf(humidity.universe, [0, 0, 50])
humidity['medium'] = fuzz.trimf(humidity.universe, [0, 50, 100])
humidity['high'] = fuzz.trimf(humidity.universe, [50, 100, 100])

fan_speed['slow'] = fuzz.trimf(fan_speed.universe, [0, 0, 50])
fan_speed['medium'] = fuzz.trimf(fan_speed.universe, [0, 50, 100])
fan_speed['fast'] = fuzz.trimf(fan_speed.universe, [50, 100, 100])

# Define rules
rule1 = ctrl.Rule(temperature['cold'] & humidity['low'], fan_speed['slow'])
rule2 = ctrl.Rule(temperature['warm'] & humidity['medium'], fan_speed['medium'])
rule3 = ctrl.Rule(temperature['hot'] | humidity['high'], fan_speed['fast'])

# Control system
fan_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
fan_sim = ctrl.ControlSystemSimulation(fan_ctrl)

# Compute output
fan_sim.input['temperature'] = 75
fan_sim.input['humidity'] = 60
fan_sim.compute()

print(f"Fan speed: {fan_sim.output['fan_speed']:.2f}")
```

---

## Practical Projects

1. **Sudoku Solver**: Use CSP with backtracking
2. **Pathfinding Visualizer**: Implement A*, Dijkstra, BFS
3. **Tic-Tac-Toe AI**: Minimax with alpha-beta pruning
4. **Robot Navigation**: Use planning algorithms
5. **Recommendation System**: Collaborative filtering
6. **Chatbot**: Rule-based or retrieval-based
7. **Game AI**: Genetic algorithm for game playing

---

**Next Steps**: Move to advanced level to learn about advanced reasoning, probabilistic programming, and cutting-edge AI techniques.
