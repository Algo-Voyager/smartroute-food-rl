#!/usr/bin/env python3
"""
Training script for the PPO agent on the food delivery environment.

This script configures and trains a PPO agent using the DeliveryEnv
environment and the DeliveryRLAgent class.
"""

import os
import sys
import argparse
import torch as th
import numpy as np
import random

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.delivery_env import DeliveryEnv
from agents.rl_agent import DeliveryRLAgent, CustomCallback

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train PPO agent for food delivery")
    
    # Environment parameters
    parser.add_argument("--data_dir", type=str, default="../data",
                       help="Directory containing data files")
    parser.add_argument("--day_start", type=int, default=10,
                       help="Starting hour of the day (24h format)")
    parser.add_argument("--day_end", type=int, default=22,
                       help="Ending hour of the day (24h format)")
    parser.add_argument("--city_size", type=float, default=10.0,
                       help="City size in km")
    parser.add_argument("--use_road_network", action="store_true",
                       help="Whether to use road network for travel times")
    parser.add_argument("--max_orders", type=int, default=100,
                       help="Maximum number of orders in the environment at once")
    parser.add_argument("--max_drivers", type=int, default=50,
                       help="Maximum number of drivers in the environment")
    
    # Agent parameters
    parser.add_argument("--policy", type=str, default="MultiInputPolicy",
                       help="Policy network type (MlpPolicy or MultiInputPolicy)")
    parser.add_argument("--n_envs", type=int, default=8,
                       help="Number of parallel environments")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--n_steps", type=int, default=2048,
                       help="Number of steps per rollout")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Minibatch size")
    parser.add_argument("--ent_coef", type=float, default=0.01,
                       help="Entropy coefficient")
    parser.add_argument("--clip_range", type=float, default=0.2,
                       help="PPO clip range")
    
    # Training parameters
    parser.add_argument("--total_timesteps", type=float, default=1e7,
                       help="Total number of timesteps to train for")
    parser.add_argument("--checkpoint_freq", type=int, default=100000,
                       help="Frequency of checkpointing (in timesteps)")
    parser.add_argument("--eval_freq", type=int, default=50000,
                       help="Frequency of evaluation (in timesteps)")
    parser.add_argument("--log_dir", type=str, default="../logs",
                       help="Directory for logs")
    parser.add_argument("--model_dir", type=str, default="../models",
                       help="Directory for models")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to model to resume training from")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to run on (auto, cpu, cuda, cuda:0, etc.)")
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seeds
    set_seed(args.seed)
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Environment keyword arguments
    env_kwargs = {
        "data_dir": args.data_dir,
        "day_start": args.day_start,
        "day_end": args.day_end,
        "city_size": args.city_size,
        "use_road_network": args.use_road_network,
        "max_orders": args.max_orders,
        "max_drivers": args.max_drivers
    }
    
    # Agent keyword arguments
    policy_kwargs = {
        "net_arch": [256, 256, 256],
        "activation_fn": th.nn.ReLU
    }
    
    # Create agent
    agent = DeliveryRLAgent(
        env_class=DeliveryEnv,
        env_kwargs=env_kwargs,
        n_envs=args.n_envs,
        policy=args.policy,
        policy_kwargs=policy_kwargs,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        verbose=1,
        tensorboard_log=args.log_dir,
        device=args.device
    )
    
    # Resume training if specified
    if args.resume:
        print(f"Resuming training from {args.resume}")
        vec_normalize_path = os.path.join(os.path.dirname(args.resume), "vec_normalize.pkl")
        agent.load(args.resume, vec_normalize_path if os.path.exists(vec_normalize_path) else None)
    
    # Train agent
    print("Starting training...")
    agent.train(
        total_timesteps=args.total_timesteps,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq
    )
    print("Training complete!")

if __name__ == "__main__":
    main() 