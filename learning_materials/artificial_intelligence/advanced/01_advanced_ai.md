# Artificial Intelligence - Advanced Level

## Monte Carlo Tree Search (MCTS)

MCTS combines tree search with Monte Carlo random sampling. Used in AlphaGo and game AI.

### Algorithm Steps

1. **Selection**: Select most promising node using UCB1
2. **Expansion**: Add child node
3. **Simulation**: Play random game from new node
4. **Backpropagation**: Update statistics along path

```python
import math
import random

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.untried_actions = get_legal_actions(state)

    def ucb1(self, exploration_param=1.41):
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits +
                exploration_param * math.sqrt(math.log(self.parent.visits) / self.visits))

    def select_child(self):
        return max(self.children, key=lambda c: c.ucb1())

    def expand(self):
        action = self.untried_actions.pop()
        next_state = apply_action(self.state, action)
        child = MCTSNode(next_state, parent=self)
        self.children.append(child)
        return child

    def simulate(self):
        state = self.state
        while not is_terminal(state):
            action = random.choice(get_legal_actions(state))
            state = apply_action(state, action)
        return get_result(state)

    def backpropagate(self, result):
        self.visits += 1
        self.value += result
        if self.parent:
            self.parent.backpropagate(result)

def mcts(root_state, iterations=1000):
    root = MCTSNode(root_state)

    for _ in range(iterations):
        node = root

        # Selection
        while node.untried_actions == [] and node.children != []:
            node = node.select_child()

        # Expansion
        if node.untried_actions != []:
            node = node.expand()

        # Simulation
        result = node.simulate()

        # Backpropagation
        node.backpropagate(result)

    # Return best action
    return max(root.children, key=lambda c: c.visits).state
```

### AlphaGo/AlphaZero Architecture

Combines MCTS with deep neural networks.

```python
import torch
import torch.nn as nn

class AlphaZeroNetwork(nn.Module):
    def __init__(self, board_size, num_actions):
        super().__init__()

        # Shared representation
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(256) for _ in range(19)
        ])

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, num_actions)
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(256, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def _make_residual_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        x = self.conv_block(x)

        for block in self.residual_blocks:
            identity = x
            x = block(x) + identity
            x = nn.functional.relu(x)

        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value
```

---

## Probabilistic Programming

Express probabilistic models as programs.

### Pyro Example

```python
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

# Define probabilistic model
def model(observations=None):
    # Prior
    mu = pyro.sample("mu", dist.Normal(0., 10.))
    sigma = pyro.sample("sigma", dist.Uniform(0., 10.))

    # Likelihood
    with pyro.plate("data", len(observations) if observations is not None else 10):
        return pyro.sample("obs", dist.Normal(mu, sigma), obs=observations)

# Generate synthetic data
true_mu, true_sigma = 3.0, 1.0
data = dist.Normal(true_mu, true_sigma).sample((100,))

# Inference using NUTS (No U-Turn Sampler)
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
mcmc.run(data)

# Get posterior samples
posterior_samples = mcmc.get_samples()
print(f"Posterior mean of mu: {posterior_samples['mu'].mean():.2f}")
print(f"Posterior mean of sigma: {posterior_samples['sigma'].mean():.2f}")
```

### Variational Inference

```python
from pyro.infer import SVI, Trace_ELBO
from pyro import poutine
import pyro.optim as optim

# Define guide (variational distribution)
def guide(observations=None):
    mu_loc = pyro.param("mu_loc", torch.tensor(0.))
    mu_scale = pyro.param("mu_scale", torch.tensor(1.),
                         constraint=dist.constraints.positive)

    sigma_loc = pyro.param("sigma_loc", torch.tensor(1.),
                          constraint=dist.constraints.positive)

    pyro.sample("mu", dist.Normal(mu_loc, mu_scale))
    pyro.sample("sigma", dist.LogNormal(sigma_loc, 1.))

# Stochastic Variational Inference
svi = SVI(model, guide, optim.Adam({"lr": 0.01}), loss=Trace_ELBO())

# Train
for step in range(2000):
    loss = svi.step(data)
    if step % 200 == 0:
        print(f"Step {step}, Loss: {loss:.2f}")

# Get learned parameters
for name, value in pyro.get_param_store().items():
    print(f"{name}: {value.item():.2f}")
```

---

## Causal Inference

Understanding cause-and-effect relationships.

### Do-Calculus

```python
import dowhy
from dowhy import CausalModel
import pandas as pd
import numpy as np

# Generate data
np.random.seed(42)
n = 1000

# Causal model: Z -> X -> Y, Z -> Y
Z = np.random.normal(0, 1, n)
X = 2 * Z + np.random.normal(0, 0.5, n)
Y = 3 * X + 1.5 * Z + np.random.normal(0, 0.5, n)

data = pd.DataFrame({'Z': Z, 'X': X, 'Y': Y})

# Define causal graph
model = CausalModel(
    data=data,
    treatment='X',
    outcome='Y',
    common_causes=['Z']
)

# Identify causal effect
identified_estimand = model.identify_effect()

# Estimate causal effect
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression"
)

print(estimate)

# Refute estimate (robustness checks)
refutation = model.refute_estimate(
    identified_estimand,
    estimate,
    method_name="random_common_cause"
)
print(refutation)
```

### Structural Causal Models (SCM)

```python
class StructuralCausalModel:
    def __init__(self):
        self.variables = {}
        self.functions = {}
        self.parents = {}

    def add_variable(self, name, parents, function):
        self.variables[name] = None
        self.parents[name] = parents
        self.functions[name] = function

    def sample(self, n=1, interventions=None):
        if interventions is None:
            interventions = {}

        samples = {var: [] for var in self.variables}

        for _ in range(n):
            values = {}

            # Topological order
            for var in self.topological_sort():
                if var in interventions:
                    values[var] = interventions[var]
                else:
                    parent_values = [values[p] for p in self.parents[var]]
                    values[var] = self.functions[var](*parent_values)

                samples[var].append(values[var])

        return {var: np.array(vals) for var, vals in samples.items()}

    def topological_sort(self):
        # Simple topological sort
        visited = set()
        order = []

        def dfs(var):
            if var in visited:
                return
            visited.add(var)
            for parent in self.parents[var]:
                dfs(parent)
            order.append(var)

        for var in self.variables:
            dfs(var)

        return order

# Example: Z -> X -> Y
scm = StructuralCausalModel()
scm.add_variable('Z', [], lambda: np.random.normal(0, 1))
scm.add_variable('X', ['Z'], lambda z: 2*z + np.random.normal(0, 0.5))
scm.add_variable('Y', ['X', 'Z'], lambda x, z: 3*x + 1.5*z + np.random.normal(0, 0.5))

# Observational data
obs_data = scm.sample(n=1000)

# Interventional data (do(X=1))
int_data = scm.sample(n=1000, interventions={'X': 1})
```

---

## Meta-Learning

Learning to learn - adapting quickly to new tasks.

### Model-Agnostic Meta-Learning (MAML)

```python
import torch
import torch.nn as nn
import torch.optim as optim

class MAML:
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_optimizer = optim.Adam(model.parameters(), lr=outer_lr)

    def inner_update(self, task_data, task_labels, num_steps=5):
        # Clone model for inner loop
        temp_model = type(self.model)().to(next(self.model.parameters()).device)
        temp_model.load_state_dict(self.model.state_dict())

        # Inner loop optimization
        for _ in range(num_steps):
            predictions = temp_model(task_data)
            loss = nn.functional.cross_entropy(predictions, task_labels)

            # Manual gradient update
            grads = torch.autograd.grad(loss, temp_model.parameters())
            for param, grad in zip(temp_model.parameters(), grads):
                param.data -= self.inner_lr * grad

        return temp_model

    def meta_update(self, tasks):
        meta_loss = 0

        for support_data, support_labels, query_data, query_labels in tasks:
            # Inner loop
            adapted_model = self.inner_update(support_data, support_labels)

            # Outer loop
            predictions = adapted_model(query_data)
            task_loss = nn.functional.cross_entropy(predictions, query_labels)
            meta_loss += task_loss

        meta_loss /= len(tasks)

        # Update meta-parameters
        self.outer_optimizer.zero_grad()
        meta_loss.backward()
        self.outer_optimizer.step()

        return meta_loss.item()

# Usage
model = SimpleNN()
maml = MAML(model)

for episode in range(1000):
    tasks = sample_tasks(num_tasks=8)
    loss = maml.meta_update(tasks)
    if episode % 100 == 0:
        print(f"Episode {episode}, Meta-loss: {loss:.4f}")
```

---

## Neural Architecture Search (NAS)

Automatically design neural network architectures.

### DARTS (Differentiable Architecture Search)

```python
import torch
import torch.nn as nn

class MixedOp(nn.Module):
    """Mixed operation with architecture parameters"""
    def __init__(self, C, stride):
        super().__init__()
        self.ops = nn.ModuleList([
            nn.Identity(),
            nn.MaxPool2d(3, stride, 1),
            nn.AvgPool2d(3, stride, 1),
            nn.Conv2d(C, C, 3, stride, 1),
            nn.Conv2d(C, C, 5, stride, 2),
        ])
        self.alpha = nn.Parameter(torch.randn(len(self.ops)))

    def forward(self, x):
        weights = torch.softmax(self.alpha, dim=0)
        return sum(w * op(x) for w, op in zip(weights, self.ops))

class SearchCell(nn.Module):
    def __init__(self, C, num_nodes=4):
        super().__init__()
        self.num_nodes = num_nodes

        self.ops = nn.ModuleList()
        for i in range(num_nodes):
            for j in range(i + 2):  # Connect to all previous nodes
                self.ops.append(MixedOp(C, stride=1))

    def forward(self, x):
        states = [x, x]  # Initial states

        for i in range(self.num_nodes):
            s = sum(self.ops[offset + j](states[j])
                   for j, offset in enumerate(range(i * (i + 1) // 2, (i + 1) * (i + 2) // 2)))
            states.append(s)

        return torch.cat(states[2:], dim=1)

# Bi-level optimization
def train_darts(search_cell, train_loader, val_loader, epochs=50):
    # Architecture parameters
    arch_params = [p for p in search_cell.parameters() if p.requires_grad]

    # Weight optimizer
    w_optimizer = torch.optim.SGD(search_cell.parameters(), lr=0.025, momentum=0.9)

    # Architecture optimizer
    a_optimizer = torch.optim.Adam(arch_params, lr=3e-4)

    for epoch in range(epochs):
        # Update weights
        for data, target in train_loader:
            w_optimizer.zero_grad()
            loss = compute_loss(search_cell(data), target)
            loss.backward()
            w_optimizer.step()

        # Update architecture
        for data, target in val_loader:
            a_optimizer.zero_grad()
            loss = compute_loss(search_cell(data), target)
            loss.backward()
            a_optimizer.step()

    # Extract final architecture
    return get_best_architecture(search_cell)
```

---

## Explainable AI (XAI)

Making AI decisions interpretable.

### SHAP for Model Interpretation

```python
import shap
import xgboost
import numpy as np
import matplotlib.pyplot as plt

# Train model
X_train, y_train = load_data()
model = xgboost.XGBRegressor().fit(X_train, y_train)

# SHAP explainer
explainer = shap.Explainer(model)
shap_values = explainer(X_train)

# Global feature importance
shap.summary_plot(shap_values, X_train)

# Single prediction explanation
shap.waterfall_plot(shap_values[0])

# Dependence plot
shap.dependence_plot("feature_name", shap_values.values, X_train)

# Force plot
shap.force_plot(explainer.expected_value, shap_values[0].values, X_train.iloc[0])
```

### Counterfactual Explanations

```python
import dice_ml

# Train model
model = train_classifier(X_train, y_train)

# Create DiCE explainer
d = dice_ml.Data(dataframe=df, continuous_features=continuous_features,
                 outcome_name='outcome')
m = dice_ml.Model(model=model, backend='sklearn')
exp = dice_ml.Dice(d, m)

# Generate counterfactuals
query_instance = X_test.iloc[0]
cf = exp.generate_counterfactuals(query_instance, total_CFs=3, desired_class=1)

cf.visualize_as_dataframe()
```

---

## Federated Learning

Training models across decentralized data.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class FederatedLearning:
    def __init__(self, global_model, num_clients):
        self.global_model = global_model
        self.num_clients = num_clients
        self.client_models = [
            type(global_model)() for _ in range(num_clients)
        ]

    def federated_averaging(self, client_weights, client_sizes):
        """FedAvg algorithm"""
        total_size = sum(client_sizes)

        # Initialize global weights
        global_weights = {}
        for key in client_weights[0].keys():
            global_weights[key] = torch.zeros_like(client_weights[0][key])

        # Weighted average
        for client_weight, client_size in zip(client_weights, client_sizes):
            for key in global_weights.keys():
                global_weights[key] += (client_size / total_size) * client_weight[key]

        return global_weights

    def train_round(self, client_datasets, epochs_per_client=5):
        client_weights = []
        client_sizes = []

        # Train on each client
        for i, dataset in enumerate(client_datasets):
            # Copy global model to client
            self.client_models[i].load_state_dict(self.global_model.state_dict())

            # Local training
            optimizer = torch.optim.SGD(self.client_models[i].parameters(), lr=0.01)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)

            self.client_models[i].train()
            for epoch in range(epochs_per_client):
                for data, target in loader:
                    optimizer.zero_grad()
                    output = self.client_models[i](data)
                    loss = nn.functional.cross_entropy(output, target)
                    loss.backward()
                    optimizer.step()

            # Collect client model
            client_weights.append(self.client_models[i].state_dict())
            client_sizes.append(len(dataset))

        # Aggregate
        global_weights = self.federated_averaging(client_weights, client_sizes)
        self.global_model.load_state_dict(global_weights)

# Usage
global_model = SimpleNN()
fl = FederatedLearning(global_model, num_clients=10)

for round in range(100):
    client_datasets = distribute_data(train_data, num_clients=10)
    fl.train_round(client_datasets)

    # Evaluate
    accuracy = evaluate(global_model, test_loader)
    print(f"Round {round}, Accuracy: {accuracy:.4f}")
```

---

## Continual Learning

Learning from stream of tasks without forgetting.

### Elastic Weight Consolidation (EWC)

```python
import torch
import torch.nn as nn

class EWC:
    def __init__(self, model, dataset, lambda_=0.4):
        self.model = model
        self.lambda_ = lambda_
        self.fisher = self._compute_fisher(dataset)
        self.optimal_params = {n: p.clone().detach() for n, p in model.named_parameters()}

    def _compute_fisher(self, dataset):
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}

        self.model.eval()
        for data, target in dataset:
            self.model.zero_grad()
            output = self.model(data)
            loss = nn.functional.cross_entropy(output, target)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2) / len(dataset)

        return fisher

    def penalty(self):
        loss = 0
        for n, p in self.model.named_parameters():
            loss += (self.fisher[n] * (p - self.optimal_params[n]).pow(2)).sum()
        return self.lambda_ * loss

# Training with EWC
def train_with_ewc(model, tasks):
    ewc = None

    for task_id, (train_loader, test_loader) in enumerate(tasks):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        for epoch in range(10):
            model.train()
            for data, target in train_loader:
                optimizer.zero_grad()

                # Standard loss
                output = model(data)
                loss = nn.functional.cross_entropy(output, target)

                # Add EWC penalty
                if ewc is not None:
                    loss += ewc.penalty()

                loss.backward()
                optimizer.step()

        # Compute Fisher information for current task
        ewc = EWC(model, train_loader)

        print(f"Task {task_id} completed")
```

---

## Multi-Task Learning

Learning multiple related tasks simultaneously.

```python
class MultiTaskModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_tasks):
        super().__init__()

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Task-specific heads
        self.task_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_tasks)
        ])

    def forward(self, x, task_id=None):
        shared_features = self.shared(x)

        if task_id is not None:
            return self.task_heads[task_id](shared_features)
        else:
            return [head(shared_features) for head in self.task_heads]

# Training
def train_multitask(model, task_datasets, num_epochs=100):
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        total_loss = 0

        for task_id, (X, y) in enumerate(task_datasets):
            optimizer.zero_grad()

            predictions = model(X, task_id)
            loss = nn.functional.mse_loss(predictions, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
```

---

## Neuro-Symbolic AI

Combining neural networks with symbolic reasoning.

```python
class NeuralLogicMachine(nn.Module):
    """Simple example of neuro-symbolic reasoning"""

    def __init__(self, num_objects, num_relations, embedding_dim):
        super().__init__()

        self.object_embeddings = nn.Embedding(num_objects, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        self.logic_modules = nn.ModuleDict({
            'and': nn.Linear(embedding_dim * 2, embedding_dim),
            'or': nn.Linear(embedding_dim * 2, embedding_dim),
            'not': nn.Linear(embedding_dim, embedding_dim)
        })

    def forward(self, objects, relations, logic_operations):
        # Get embeddings
        obj_embs = self.object_embeddings(objects)
        rel_embs = self.relation_embeddings(relations)

        # Apply logic operations
        result = obj_embs
        for op, args in logic_operations:
            if op == 'and':
                result = self.logic_modules['and'](
                    torch.cat([result, args], dim=-1)
                )
            elif op == 'or':
                result = self.logic_modules['or'](
                    torch.cat([result, args], dim=-1)
                )
            elif op == 'not':
                result = self.logic_modules['not'](result)

        return result
```

---

## Key Takeaways

- MCTS combines tree search with sampling for game playing
- Probabilistic programming enables flexible probabilistic modeling
- Causal inference reveals cause-effect relationships
- Meta-learning enables quick adaptation to new tasks
- NAS automates neural architecture design
- XAI makes models interpretable and trustworthy
- Federated learning enables privacy-preserving distributed training
- Continual learning prevents catastrophic forgetting
- Neuro-symbolic AI combines learning with reasoning

---

**Next Steps**: Explore cutting-edge research, contribute to open-source projects, and work on advanced AI applications.
