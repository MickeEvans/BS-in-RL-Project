# Utilities Module

This module contains utility functions and helpers.

## Structure

- `black_scholes.py`: Black-Scholes formula implementations
- `metrics.py`: Performance metrics and evaluation functions
- `plotting.py`: Visualization utilities
- `config.py`: Configuration management

## Usage

```python
from src.utils.black_scholes import calculate_call_price, calculate_delta

price = calculate_call_price(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
delta = calculate_delta(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
```
