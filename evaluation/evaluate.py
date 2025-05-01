#!/usr/bin/env python3
"""
Evaluation script for comparing RL agent with baseline heuristics.

This script evaluates the performance of different agents (PPO, nearest-driver,
fixed-cutoff) on the food delivery environment and generates comparative metrics.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.delivery_env import DeliveryEnv
from agents.baseline_agents import NearestDriverAgent, FixedCutoffAgent
from agents.rl_agent import DeliveryRLAgent

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate food delivery agents")
    
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
    
    # Evaluation parameters
    parser.add_argument("--n_episodes", type=int, default=10,
                       help="Number of episodes to evaluate")
    parser.add_argument("--model_path", type=str, default="../models/ppo_delivery_final.zip",
                       help="Path to trained RL model")
    parser.add_argument("--vec_normalize_path", type=str, default="../models/vec_normalize.pkl",
                       help="Path to VecNormalize statistics")
    parser.add_argument("--output_dir", type=str, default="../evaluation",
                       help="Directory for output files")
    parser.add_argument("--render", action="store_true",
                       help="Render the environment during evaluation")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    return parser.parse_args()

def evaluate_agent(env, agent, n_episodes=10, render=False):
    """
    Evaluate an agent on the environment.
    
    Args:
        env: Environment to evaluate on
        agent: Agent to evaluate
        n_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Reset metrics
    total_reward = 0
    total_deliveries = 0
    total_rejects = 0
    total_repositions = 0
    total_delivery_time = 0
    delivery_times = []
    late_deliveries = 0  # Deliveries over 30 minutes
    
    # Tracking for visualizations
    episode_rewards = []
    time_series = []  # (time, n_drivers, n_orders, n_deliveries, n_rejects)
    driver_timelines = {}  # {driver_id: [(start_time, end_time, status)]}
    
    for episode in tqdm(range(n_episodes), desc="Evaluating episodes"):
        # Reset environment
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_time_series = []
        
        # Initialize driver timelines for this episode
        curr_driver_timelines = {}
        
        while not (done or truncated):
            # Choose action
            if hasattr(agent, 'act'):
                action = agent.act(obs)
            else:
                # Handle case where agent is an RL model
                action = agent.predict(obs, deterministic=True)[0]
            
            # Take action
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            if render:
                env.render()
            
            # Record time series data
            episode_time_series.append((
                env.current_time,
                len(env.drivers),
                len(env.active_orders) + len(env.prepping_orders) + len(env.ready_orders),
                env.total_deliveries,
                env.total_rejects
            ))
            
            # Record driver timelines
            for driver_id, driver in env.drivers.items():
                if driver_id not in curr_driver_timelines:
                    curr_driver_timelines[driver_id] = []
                
                # If we have a previous record, check if status changed
                if curr_driver_timelines[driver_id]:
                    prev_start, _, prev_status = curr_driver_timelines[driver_id][-1]
                    if prev_status != driver['status']:
                        # Status changed, record end time for previous status
                        curr_driver_timelines[driver_id][-1] = (prev_start, env.current_time, prev_status)
                        # Start new status
                        curr_driver_timelines[driver_id].append((env.current_time, None, driver['status']))
                else:
                    # First record for this driver
                    curr_driver_timelines[driver_id].append((env.current_time, None, driver['status']))
        
        # Close out driver timelines
        for driver_id, timeline in curr_driver_timelines.items():
            for i, (start, end, status) in enumerate(timeline):
                if end is None:
                    timeline[i] = (start, env.current_time, status)
            
            # Add to overall driver timelines
            if driver_id not in driver_timelines:
                driver_timelines[driver_id] = []
            driver_timelines[driver_id].extend(timeline)
        
        # Accumulate metrics
        total_reward += episode_reward
        total_deliveries += env.total_deliveries
        total_rejects += env.total_rejects
        total_repositions += env.total_repositions
        total_delivery_time += env.total_delivery_time
        
        # Record delivery times for completed orders
        for order_id in env.completed_orders:
            order = env.orders[order_id]
            delivery_time_minutes = (order['delivery_time'] - order['order_time']) * 60
            delivery_times.append(delivery_time_minutes)
            
            if delivery_time_minutes > 30:
                late_deliveries += 1
        
        episode_rewards.append(episode_reward)
        time_series.extend(episode_time_series)
    
    # Calculate metrics
    avg_reward = total_reward / n_episodes
    avg_deliveries = total_deliveries / n_episodes
    avg_rejects = total_rejects / n_episodes
    avg_repositions = total_repositions / n_episodes
    
    completion_rate = total_deliveries / (total_deliveries + total_rejects) if (total_deliveries + total_rejects) > 0 else 0
    late_rate = late_deliveries / total_deliveries if total_deliveries > 0 else 0
    avg_delivery_time = total_delivery_time / total_deliveries if total_deliveries > 0 else 0
    
    # Return metrics
    metrics = {
        "avg_reward": avg_reward,
        "avg_deliveries": avg_deliveries,
        "avg_rejects": avg_rejects,
        "avg_repositions": avg_repositions,
        "completion_rate": completion_rate,
        "late_rate": late_rate,
        "avg_delivery_time": avg_delivery_time,
        "delivery_times": delivery_times,
        "episode_rewards": episode_rewards,
        "time_series": time_series,
        "driver_timelines": driver_timelines
    }
    
    return metrics

def plot_metrics(metrics_dict, output_dir):
    """
    Plot evaluation metrics.
    
    Args:
        metrics_dict: Dictionary with metrics for each agent
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set plot style
    sns.set(style="whitegrid")
    
    # Metrics comparison
    metrics_to_plot = ["avg_reward", "avg_deliveries", "avg_rejects", 
                      "completion_rate", "late_rate", "avg_delivery_time"]
    
    metrics_df = pd.DataFrame({
        agent_name: [metrics[metric] for metric in metrics_to_plot]
        for agent_name, metrics in metrics_dict.items()
    }, index=metrics_to_plot)
    
    # Plot metrics comparison
    plt.figure(figsize=(12, 8))
    metrics_df.plot(kind='bar')
    plt.title('Metrics Comparison between Agents')
    plt.ylabel('Value')
    plt.xlabel('Metric')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300)
    
    # Delivery time histograms
    plt.figure(figsize=(12, 8))
    for agent_name, metrics in metrics_dict.items():
        if metrics['delivery_times']:
            sns.histplot(metrics['delivery_times'], label=agent_name, alpha=0.5, bins=20)
    plt.axvline(x=30, color='r', linestyle='--', label='30 min threshold')
    plt.title('Delivery Time Distribution')
    plt.xlabel('Delivery Time (minutes)')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'delivery_time_histogram.png'), dpi=300)
    
    # Driver timeline Gantt chart (for the first episode of the RL agent only)
    if 'RL Agent' in metrics_dict:
        plot_driver_timeline(metrics_dict['RL Agent']['driver_timelines'], output_dir)
    
    # Time series plots
    for agent_name, metrics in metrics_dict.items():
        time_series = metrics['time_series']
        if time_series:
            times, n_drivers, n_orders, n_deliveries, n_rejects = zip(*time_series)
            
            plt.figure(figsize=(12, 8))
            plt.plot(times, n_drivers, label='Active Drivers')
            plt.plot(times, n_orders, label='Active Orders')
            plt.plot(times, n_deliveries, label='Cumulative Deliveries')
            plt.plot(times, n_rejects, label='Cumulative Rejects')
            plt.title(f'Time Series for {agent_name}')
            plt.xlabel('Time (hours)')
            plt.ylabel('Count')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'time_series_{agent_name.replace(" ", "_")}.png'), dpi=300)

def plot_driver_timeline(driver_timelines, output_dir):
    """
    Plot driver timeline Gantt chart.
    
    Args:
        driver_timelines: Dictionary with driver timelines
        output_dir: Directory to save plots
    """
    # Select a subset of drivers to avoid overcrowding the plot
    driver_subset = list(driver_timelines.keys())[:10]
    
    # Create data for Gantt chart
    gantt_data = []
    for i, driver_id in enumerate(driver_subset):
        timeline = driver_timelines[driver_id]
        for start, end, status in timeline:
            color = 'green' if status == 'free' else 'red'
            gantt_data.append({
                'Driver': f'Driver {driver_id}',
                'Start': start,
                'End': end,
                'Status': status,
                'Color': color
            })
    
    # Plot Gantt chart
    plt.figure(figsize=(15, 8))
    for entry in gantt_data:
        plt.barh(
            y=entry['Driver'],
            width=entry['End'] - entry['Start'],
            left=entry['Start'],
            color=entry['Color'],
            alpha=0.6,
            label=entry['Status'] if entry['Status'] + '_label' not in locals() else ""
        )
        if entry['Status'] + '_label' not in locals():
            locals()[entry['Status'] + '_label'] = True
    
    # Customize Gantt chart
    plt.title('Driver Timeline')
    plt.xlabel('Time (hours)')
    plt.ylabel('Driver')
    plt.grid(True, axis='x')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'driver_timeline.png'), dpi=300)

def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()
    
    # Create environment
    env_kwargs = {
        "data_dir": args.data_dir,
        "day_start": args.day_start,
        "day_end": args.day_end,
        "city_size": args.city_size,
        "use_road_network": args.use_road_network,
        "max_orders": args.max_orders,
        "max_drivers": args.max_drivers,
        "render_mode": "human" if args.render else None
    }
    
    env = DeliveryEnv(**env_kwargs)
    
    # Create agents
    nearest_driver_agent = NearestDriverAgent(env)
    fixed_cutoff_agent = FixedCutoffAgent(env)
    
    # Load RL agent if model path exists
    rl_agent = None
    if os.path.exists(args.model_path):
        # Create a DeliveryRLAgent instance
        rl_agent = DeliveryRLAgent(
            env_class=DeliveryEnv,
            env_kwargs=env_kwargs
        )
        
        # Load the trained model
        rl_agent.load(args.model_path, args.vec_normalize_path)
    
    # Evaluate agents
    results = {}
    
    print("Evaluating Nearest Driver Agent...")
    nearest_driver_metrics = evaluate_agent(env, nearest_driver_agent, args.n_episodes, args.render)
    results["Nearest Driver"] = nearest_driver_metrics
    
    print("Evaluating Fixed Cutoff Agent...")
    fixed_cutoff_metrics = evaluate_agent(env, fixed_cutoff_agent, args.n_episodes, args.render)
    results["Fixed Cutoff"] = fixed_cutoff_metrics
    
    if rl_agent is not None:
        print("Evaluating RL Agent...")
        rl_metrics = evaluate_agent(env, rl_agent.model, args.n_episodes, args.render)
        results["RL Agent"] = rl_metrics
    
    # Print summary
    print("\nEvaluation Summary:")
    print("-" * 50)
    
    for agent_name, metrics in results.items():
        print(f"\n{agent_name}:")
        print(f"  Average Reward: {metrics['avg_reward']:.2f}")
        print(f"  Average Deliveries: {metrics['avg_deliveries']:.2f}")
        print(f"  Completion Rate: {metrics['completion_rate']:.2%}")
        print(f"  Late Rate: {metrics['late_rate']:.2%}")
        print(f"  Average Delivery Time: {metrics['avg_delivery_time']:.2f} minutes")
    
    # Plot metrics
    plot_metrics(results, args.output_dir)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        agent_name: {
            'avg_reward': metrics['avg_reward'],
            'avg_deliveries': metrics['avg_deliveries'],
            'avg_rejects': metrics['avg_rejects'],
            'avg_repositions': metrics['avg_repositions'],
            'completion_rate': metrics['completion_rate'],
            'late_rate': metrics['late_rate'],
            'avg_delivery_time': metrics['avg_delivery_time']
        }
        for agent_name, metrics in results.items()
    })
    
    metrics_df.to_csv(os.path.join(args.output_dir, 'metrics.csv'))
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main() 