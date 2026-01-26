# Models Module

This module contains neural network model architectures for deep RL agents.

## Structure

- `networks.py`: Neural network architectures (Q-networks, policy networks)
- `value_functions.py`: Value function approximators

## Usage

```python
from src.models.networks import QNetwork

q_network = QNetwork(
    state_dim=10,
    action_dim=3,
    hidden_dims=[64, 64]
)
```
