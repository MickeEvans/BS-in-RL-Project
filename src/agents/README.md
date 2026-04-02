# Agents Module

This module contains reinforcement learning agent implementations.

## Implemented Agents

- **Q-Learning**: Tabular Q-learning with epsilon-greedy exploration
- **DQN**: Deep Q-Network for continuous state spaces
- **Actor-Critic**: Policy gradient methods

## Structure

- `q_learning.py`: Q-learning agent
- `dqn.py`: Deep Q-Network agent
- `base_agent.py`: Base class for all agents

## Usage

```python
from src.agents.q_learning import QLearningAgent

agent = QLearningAgent(
    state_space_size=100,
    action_space_size=3,
    learning_rate=0.1,
    discount_factor=0.99,
    epsilon=0.1
)

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
```
