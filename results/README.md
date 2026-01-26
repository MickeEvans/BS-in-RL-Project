# Results

This directory stores experiment results, plots, and analysis outputs.

## Structure

Organize results by experiment:
```
results/
├── experiment_1/
│   ├── plots/
│   ├── metrics.json
│   └── README.md
├── experiment_2/
│   ├── plots/
│   ├── metrics.json
│   └── README.md
└── README.md
```

## Best Practices

1. **Create subdirectories**: One folder per experiment
2. **Document experiments**: Add README.md describing the experiment
3. **Version control**: Commit important results and plots (but not large files)
4. **Naming convention**: Use descriptive names with dates (e.g., `q_learning_2026-01-26`)
5. **Metrics**: Save metrics in structured format (JSON, CSV)
6. **Plots**: Save plots as PNG or SVG for version control

## Example Experiment Documentation

```markdown
# Experiment: Q-Learning vs DQN Comparison

**Date**: 2026-01-26
**Author**: Your Name

## Objective
Compare performance of Q-learning and DQN agents on option hedging task.

## Configuration
- Episodes: 10,000
- Transaction cost: 0.01
- Strike price: 100
- Volatility: 0.2

## Results
- Q-learning final loss: $2.34
- DQN final loss: $1.89

## Conclusions
DQN performs better in continuous state spaces.
```
