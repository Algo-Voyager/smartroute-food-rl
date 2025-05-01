# Food Delivery Reinforcement Learning

A dynamic routing system for on-demand meal delivery leveraging reinforcement learning, capable of handling up to 10⁸ synthetic orders.

## Project Structure

- `data/`: Contains synthetic data generators and datasets
- `envs/`: RL environment implementation for delivery simulation
- `agents/`: RL models and baseline heuristic implementations
- `training/`: Training scripts and utilities
- `evaluation/`: Evaluation metrics and analysis tools
- `config/`: Configuration files for the environment and models
- `visualization/`: Scripts for visualizing results and simulation

## Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Generate synthetic data:
```bash
python data/generate_data.py
```

2. Train the RL model:
```bash
python training/train_ppo.py
```

3. Evaluate the model:
```bash
python evaluation/evaluate.py
```

4. Visualize results:
```bash
python visualization/visualize.py
```

## Features

- Synthetic dataset generation (restaurants, orders, drivers, road network)
- OpenAI Gym-compatible simulation environment
- PPO implementation for driver assignment and routing
- Baseline heuristics for comparison
- Comprehensive evaluation metrics and visualizations 