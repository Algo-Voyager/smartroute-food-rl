#!/usr/bin/env python3
"""
Main script for the food delivery RL project.

This script ties together all components of the project:
1. Data generation
2. Environment setup
3. Agent training
4. Evaluation
5. Visualization

It can be used to run the entire pipeline or individual components.
"""

import os
import sys
import argparse
import json
import subprocess
import time

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Food Delivery RL Pipeline")
    
    # Pipeline stages
    parser.add_argument("--generate_data", action="store_true",
                       help="Generate synthetic data")
    parser.add_argument("--train", action="store_true",
                       help="Train RL agent")
    parser.add_argument("--evaluate", action="store_true",
                       help="Evaluate agents")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualizations")
    parser.add_argument("--all", action="store_true",
                       help="Run full pipeline")
    
    # Configuration
    parser.add_argument("--config", type=str, default="config/default_config.json",
                       help="Configuration file path")
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def generate_data(config):
    """Generate synthetic data."""
    print("Generating synthetic data...")
    
    data_config = config['data_generation']
    env_config = config['environment']
    
    cmd = [
        "python", "data/generate_data.py",
        "--output_dir", env_config['data_dir'],
        "--n_restaurants", str(data_config['n_restaurants']),
        "--n_drivers", str(data_config['n_drivers']),
        "--n_orders", str(data_config['n_orders']),
        "--city_size", str(env_config['city_size']),
        "--grid_size", str(data_config['grid_size']),
        "--day_start", str(env_config['day_start']),
        "--day_end", str(env_config['day_end'])
    ]
    
    # Convert command to strings
    cmd = [str(item) for item in cmd]
    
    subprocess.run(cmd)
    print("Data generation complete!")

def train_agent(config):
    """Train RL agent."""
    print("Training RL agent...")
    
    env_config = config['environment']
    train_config = config['training']
    
    cmd = [
        "python", "training/train_ppo.py",
        "--data_dir", env_config['data_dir'],
        "--day_start", str(env_config['day_start']),
        "--day_end", str(env_config['day_end']),
        "--city_size", str(env_config['city_size']),
        "--max_orders", str(env_config['max_orders']),
        "--max_drivers", str(env_config['max_drivers']),
        "--policy", train_config['policy'],
        "--n_envs", str(train_config['n_envs']),
        "--gamma", str(train_config['gamma']),
        "--learning_rate", str(train_config['learning_rate']),
        "--n_steps", str(train_config['n_steps']),
        "--batch_size", str(train_config['batch_size']),
        "--ent_coef", str(train_config['ent_coef']),
        "--clip_range", str(train_config['clip_range']),
        "--total_timesteps", str(train_config['total_timesteps']),
        "--checkpoint_freq", str(train_config['checkpoint_freq']),
        "--eval_freq", str(train_config['eval_freq'])
    ]
    
    if env_config.get('use_road_network', False):
        cmd.append("--use_road_network")
    
    # Convert command to strings
    cmd = [str(item) for item in cmd]
    
    subprocess.run(cmd)
    print("Training complete!")

def evaluate_agents(config):
    """Evaluate agents."""
    print("Evaluating agents...")
    
    env_config = config['environment']
    eval_config = config['evaluation']
    
    cmd = [
        "python", "evaluation/evaluate.py",
        "--data_dir", env_config['data_dir'],
        "--day_start", str(env_config['day_start']),
        "--day_end", str(env_config['day_end']),
        "--city_size", str(env_config['city_size']),
        "--max_orders", str(env_config['max_orders']),
        "--max_drivers", str(env_config['max_drivers']),
        "--n_episodes", str(eval_config['n_episodes']),
        "--model_path", eval_config['model_path'],
        "--vec_normalize_path", eval_config['vec_normalize_path'],
        "--output_dir", eval_config['output_dir']
    ]
    
    if env_config.get('use_road_network', False):
        cmd.append("--use_road_network")
    
    # Convert command to strings
    cmd = [str(item) for item in cmd]
    
    subprocess.run(cmd)
    print("Evaluation complete!")

def generate_visualizations(config):
    """Generate visualizations."""
    print("Generating visualizations...")
    
    env_config = config['environment']
    vis_config = config['visualization']
    
    cmd = [
        "python", "visualization/visualize.py",
        "--data_dir", env_config['data_dir'],
        "--output_dir", vis_config['output_dir'],
        "--city_size", str(env_config['city_size'])
    ]
    
    # Convert command to strings
    cmd = [str(item) for item in cmd]
    
    subprocess.run(cmd)
    print("Visualization complete!")

def main():
    """Main function to run the pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create necessary directories
    os.makedirs(config['environment']['data_dir'], exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs(config['evaluation']['output_dir'], exist_ok=True)
    os.makedirs(config['visualization']['output_dir'], exist_ok=True)
    
    # Run pipeline stages
    if args.all or args.generate_data:
        generate_data(config)
    
    if args.all or args.train:
        train_agent(config)
    
    if args.all or args.evaluate:
        evaluate_agents(config)
    
    if args.all or args.visualize:
        generate_visualizations(config)
    
    if not (args.all or args.generate_data or args.train or args.evaluate or args.visualize):
        print("No pipeline stages selected. Use --all or specify stages with --generate_data, --train, --evaluate, --visualize")

if __name__ == "__main__":
    main() 