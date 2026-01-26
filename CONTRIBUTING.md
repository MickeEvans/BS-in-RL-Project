# Contributing to BS-in-RL-Project

Thank you for contributing to our reinforcement learning project! This document provides guidelines for collaboration.

## 🔄 Workflow

We use a feature branch workflow for collaboration:

### 1. Setting Up Your Environment

```bash
# Clone the repository
git clone https://github.com/MickeEvans/BS-in-RL-Project.git
cd BS-in-RL-Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Working on a Feature

```bash
# Update your main branch
git checkout main
git pull origin main

# Create a new feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ... edit files ...

# Add and commit changes
git add .
git commit -m "Brief description of changes"

# Push to GitHub
git push origin feature/your-feature-name
```

### 3. Creating a Pull Request

1. Go to the repository on GitHub
2. Click "New Pull Request"
3. Select your feature branch
4. Write a clear description of your changes
5. Request review from team members
6. Address any review comments
7. Once approved, merge into main

## 📝 Commit Message Guidelines

Write clear, descriptive commit messages:

- **Good**: "Add Q-learning agent with epsilon-greedy exploration"
- **Good**: "Fix bug in transaction cost calculation"
- **Bad**: "Update code"
- **Bad**: "Changes"

Format:
```
Brief summary (50 chars or less)

More detailed explanation if needed. Explain what and why,
not how. Wrap at 72 characters.
```

## 🏗️ Code Style

- Follow PEP 8 style guide for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular
- Comment complex logic

Example:
```python
def calculate_option_payoff(spot_price, strike_price, option_type='call'):
    """
    Calculate the payoff of a European option at maturity.
    
    Args:
        spot_price (float): Current stock price
        strike_price (float): Strike price of the option
        option_type (str): 'call' or 'put'
    
    Returns:
        float: Option payoff
    """
    if option_type == 'call':
        return max(spot_price - strike_price, 0)
    else:
        return max(strike_price - spot_price, 0)
```

## 🧪 Testing

- Write tests for new functionality
- Run tests before submitting pull requests
- Ensure all tests pass: `pytest tests/`
- Maintain or improve code coverage

## 📁 Project Organization

- **src/**: Source code for the project
- **notebooks/**: Jupyter notebooks for experiments and analysis
- **tests/**: Unit tests
- **data/**: Data files (not committed to git)
- **results/**: Experiment results, plots, and reports
- **docs/**: Additional documentation

## 🚫 What Not to Commit

- Large data files (use `.gitignore`)
- Model checkpoints (unless small and important)
- Personal IDE configurations
- API keys or credentials
- Temporary files

## 🤝 Code Review Process

1. **Submit PR**: Create pull request with clear description
2. **Review**: Team members review within 1-2 days
3. **Discuss**: Address comments and questions
4. **Approve**: At least one approval required
5. **Merge**: Squash and merge into main

## 💡 Best Practices

### Branching Strategy

- `main`: Production-ready code
- `feature/*`: New features or enhancements
- `bugfix/*`: Bug fixes
- `experiment/*`: Experimental code

### Communication

- Use GitHub Issues to track bugs and features
- Comment on pull requests for discussions
- Keep team updated on major changes
- Ask questions if something is unclear

### Collaboration Tips

1. **Pull frequently**: Run `git pull` before starting work
2. **Commit often**: Make small, logical commits
3. **Push regularly**: Share your progress
4. **Test locally**: Ensure code works before pushing
5. **Document**: Update README and docs as needed

## 🐛 Reporting Bugs

When reporting bugs, include:
- Description of the issue
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment (Python version, OS, etc.)

## 💻 Development Tips

### Running Experiments

Keep track of experiments in notebooks:
```bash
jupyter notebook notebooks/
```

### Code Formatting

Use tools to maintain code quality:
```bash
# Format code
black src/

# Check style
flake8 src/

# Type checking (optional)
mypy src/
```

## ❓ Questions?

If you have questions:
1. Check existing documentation
2. Search closed issues
3. Open a new issue with the "question" label
4. Ask team members

## 📚 Resources

- [Git Documentation](https://git-scm.com/doc)
- [GitHub Flow Guide](https://guides.github.com/introduction/flow/)
- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [Writing Good Commit Messages](https://chris.beams.io/posts/git-commit/)

Thank you for contributing! 🎉
