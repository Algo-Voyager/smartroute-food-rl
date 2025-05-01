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

3. Pause and resume training:
```bash
# Training automatically saves checkpoints
# To resume from the latest checkpoint:
python resume_training.py

# To resume from a specific checkpoint:
python resume_training.py --manual_checkpoint models/ppo_delivery_specific_checkpoint.zip
```

4. Evaluate the model:
```bash
python evaluation/evaluate.py
```

5. Visualize results:
```bash
python visualization/visualize.py
```

## Features

- Synthetic dataset generation (restaurants, orders, drivers, road network)
- OpenAI Gym-compatible simulation environment
- PPO implementation for driver assignment and routing
- Baseline heuristics for comparison
- Comprehensive evaluation metrics and visualizations
- Automatic checkpoint saving and ability to pause/resume training 