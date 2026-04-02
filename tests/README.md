# Tests

This directory contains unit tests for the project.

## Running Tests

Run all tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=src tests/
```

Run specific test file:
```bash
pytest tests/test_environment.py
```

## Test Structure

- `test_environment/`: Tests for trading environment
- `test_agents/`: Tests for RL agents
- `test_models/`: Tests for neural network models
- `test_utils/`: Tests for utility functions

## Writing Tests

Follow these guidelines:

1. **Use pytest**: All tests should use pytest framework
2. **Test naming**: Test files should start with `test_`
3. **Test functions**: Test functions should start with `test_`
4. **Fixtures**: Use pytest fixtures for common setup
5. **Assertions**: Use clear, specific assertions
6. **Documentation**: Add docstrings to complex tests

Example:
```python
import pytest
from src.environment.option_env import OptionHedgingEnv

def test_environment_reset():
    """Test that environment resets correctly."""
    env = OptionHedgingEnv()
    obs = env.reset()
    assert obs is not None
    assert len(obs) == env.observation_space.shape[0]

def test_environment_step():
    """Test that environment step works correctly."""
    env = OptionHedgingEnv()
    env.reset()
    obs, reward, done, info = env.step(0)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
```
