# Quick Start Guide

This guide will help you get started with the BS-in-RL-Project quickly.

## Prerequisites

- Python 3.8+
- pip
- Git

## Installation

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/MickeEvans/BS-in-RL-Project.git
cd BS-in-RL-Project

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
# Run tests to verify everything is working
pytest tests/

# Start Jupyter to explore notebooks
jupyter notebook
```

## Project Workflow

### For New Contributors

1. **Read the documentation**
   - README.md (overview)
   - CONTRIBUTING.md (workflow)

2. **Explore the code**
   - Browse `src/` modules
   - Check example notebooks in `notebooks/`

3. **Set up your branch**
   ```bash
   git checkout -b feature/your-name-feature
   ```

4. **Make changes and test**
   ```bash
   # Make your changes
   # Run tests
   pytest tests/
   
   # Commit changes
   git add .
   git commit -m "Description of changes"
   git push origin feature/your-name-feature
   ```

5. **Create Pull Request**
   - Go to GitHub
   - Create PR from your branch
   - Request review from team

## Common Tasks

### Running Experiments

```python
# In a Jupyter notebook or Python script
from src.environment.option_env import OptionHedgingEnv
from src.agents.q_learning import QLearningAgent

# Create environment
env = OptionHedgingEnv()

# Create agent
agent = QLearningAgent()

# Train
for episode in range(1000):
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
```

### Adding New Features

1. Create feature branch: `git checkout -b feature/new-feature`
2. Implement your feature in appropriate module
3. Add tests in `tests/`
4. Update documentation if needed
5. Submit pull request

### Code Style

```bash
# Format code
black src/

# Check style
flake8 src/
```

## Directory Guide

- **src/**: Source code
  - `environment/`: Trading environment
  - `agents/`: RL agents
  - `models/`: Neural networks
  - `utils/`: Helper functions

- **notebooks/**: Jupyter notebooks for experiments
- **tests/**: Unit tests
- **data/**: Data files (not in git)
- **results/**: Experiment results
- **docs/**: Additional documentation

## Getting Help

- Check documentation in `docs/`
- Read example notebooks
- Ask questions via GitHub issues
- Contact team members

## Next Steps

1. Run example notebooks in `notebooks/`
2. Read through code in `src/`
3. Try implementing a simple agent
4. Run experiments and analyze results

Happy coding! 🚀
