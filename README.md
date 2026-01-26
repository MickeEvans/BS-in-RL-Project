# Black-Scholes Hedging with Reinforcement Learning

A reinforcement learning approach to hedging European call options in markets with transaction costs using Q-learning and deep reinforcement learning techniques.

## 📋 Project Overview

This project explores optimal hedging strategies for European call options in the presence of transaction costs. Traditional Black-Scholes delta hedging becomes suboptimal when transaction costs are introduced. We use reinforcement learning (specifically Q-learning) to learn optimal hedging policies that minimize risk while accounting for trading costs.

## 👥 Contributors

This is a collaborative project by three team members working on reinforcement learning applications in quantitative finance.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/MickeEvans/BS-in-RL-Project.git
cd BS-in-RL-Project
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
BS-in-RL-Project/
├── src/                    # Source code
│   ├── environment/        # Trading environment implementation
│   ├── agents/            # RL agents (Q-learning, DQN, etc.)
│   ├── models/            # Neural network models
│   └── utils/             # Utility functions
├── notebooks/             # Jupyter notebooks for experiments
├── tests/                 # Unit tests
├── data/                  # Data files (not tracked by git)
├── results/               # Experiment results and plots
├── docs/                  # Additional documentation
├── requirements.txt       # Python dependencies
├── CONTRIBUTING.md        # Contribution guidelines
└── README.md             # This file
```

## 🎯 Usage

### Training an Agent

```bash
python src/train.py --agent q_learning --episodes 10000
```

### Running Experiments

See the `notebooks/` directory for example experiments and analysis.

## 🧪 Testing

Run tests using pytest:
```bash
pytest tests/
```

## 📊 Key Features

- **Market Simulation**: Realistic Black-Scholes market with configurable transaction costs
- **RL Agents**: Multiple agent implementations (Q-learning, Deep Q-Networks)
- **Hedging Strategies**: Comparison with traditional delta hedging
- **Analysis Tools**: Comprehensive performance metrics and visualization

## 📚 Research Background

The project is based on research in:
- Black-Scholes option pricing model
- Delta hedging strategies
- Reinforcement learning for financial applications
- Q-learning and value-based methods

## 🤝 Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or discussions, please open an issue on GitHub.

## 🙏 Acknowledgments

- Reinforcement learning course materials
- Quantitative finance literature on option hedging
- Open-source RL libraries and frameworks
