# Environment Module

This module contains the trading environment implementation for the reinforcement learning agents.

## Structure

- `market.py`: Black-Scholes market simulation
- `option_env.py`: OpenAI Gym-compatible environment for option hedging
- `transaction_costs.py`: Transaction cost models

## Usage

```python
from src.environment.option_env import OptionHedgingEnv

env = OptionHedgingEnv(
    initial_price=100,
    strike_price=100,
    maturity=1.0,
    volatility=0.2,
    risk_free_rate=0.05,
    transaction_cost_rate=0.01
)

obs = env.reset()
done = False
while not done:
    action = agent.select_action(obs)
    obs, reward, done, info = env.step(action)
```
