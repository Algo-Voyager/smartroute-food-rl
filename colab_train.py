#!/usr/bin/env python3
"""
Colab-optimized training script for food delivery RL.

This script provides:
1. Better error handling
2. More frequent progress updates
3. Guaranteed checkpoint creation
4. Reduced training time for Colab (5M steps)
5. Automatic model saving to Google Drive
"""

import os
import sys
import time
import json
import argparse
import datetime
import traceback
import subprocess
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Colab-optimized RL training")
    parser.add_argument("--config", type=str, default="config/default_config.json",
                       help="Configuration file path")
    parser.add_argument("--steps", type=int, default=5000000,
                       help="Total timesteps to train for")
    parser.add_argument("--envs", type=int, default=16,
                       help="Number of parallel environments")
    parser.add_argument("--checkpoint_freq", type=int, default=50000, 
                       help="Checkpoint frequency in timesteps")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--save_to_drive", action="store_true",
                       help="Save checkpoints to Google Drive")
    parser.add_argument("--drive_path", type=str, default="/content/drive/MyDrive/food_delivery_rl",
                       help="Path in Google Drive to save checkpoints")
    return parser.parse_args()

def ensure_directories():
    """Ensure all necessary directories exist."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("config", exist_ok=True)

def create_default_config():
    """Create a default configuration file if it doesn't exist."""
    config_path = "config/default_config.json"
    if os.path.exists(config_path):
        return
    
    print(f"Creating default config at {config_path}")
    
    default_config = {
        "data_generation": {
            "n_restaurants": 200,
            "n_drivers": 50,
            "n_orders": 10000,
            "grid_size": 20
        },
        "environment": {
            "data_dir": "data",
            "day_start": 10,
            "day_end": 22,
            "city_size": 10.0,
            "max_orders": 100,
            "max_drivers": 50,
            "use_road_network": True
        },
        "training": {
            "policy": "MultiInputPolicy",
            "n_envs": 16,
            "gamma": 0.99,
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 128,
            "ent_coef": 0.01,
            "clip_range": 0.2,
            "total_timesteps": 5000000,  # 5M steps for Colab
            "checkpoint_freq": 50000,    # More frequent checkpoints
            "eval_freq": 50000
        },
        "evaluation": {
            "n_episodes": 10,
            "model_path": "models/ppo_delivery_final.zip",
            "vec_normalize_path": "models/vec_normalize.pkl",
            "output_dir": "evaluation"
        },
        "visualization": {
            "output_dir": "visualization"
        }
    }
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=4)

def load_config(config_path):
    """Load configuration from file with error handling."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        print("Creating and using default config instead.")
        create_default_config()
        with open("config/default_config.json", 'r') as f:
            return json.load(f)

def update_config_with_args(config, args):
    """Update configuration with command line arguments."""
    if args.steps:
        config["training"]["total_timesteps"] = args.steps
    if args.envs:
        config["training"]["n_envs"] = args.envs
    if args.checkpoint_freq:
        config["training"]["checkpoint_freq"] = args.checkpoint_freq
    
    return config

def copy_to_drive(source_dir, drive_path):
    """Copy models to Google Drive for persistence."""
    # Check if Google Drive is mounted
    if not os.path.exists("/content/drive"):
        print("Google Drive not mounted. Skipping drive backup.")
        return False
    
    # Create the directory in Drive if it doesn't exist
    os.makedirs(drive_path, exist_ok=True)
    
    try:
        # Copy all files from the source directory to Drive
        cmd = f"cp -r {source_dir}/* {drive_path}/"
        subprocess.run(cmd, shell=True, check=True)
        print(f"✅ Successfully copied {source_dir} to Google Drive at {drive_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to copy to Google Drive: {e}")
        return False

def setup_environment():
    """Set up the Python environment and imports."""
    try:
        import gymnasium
        from stable_baselines3 import PPO
        print(f"Gymnasium version: {gymnasium.__version__}")
        print(f"Stable-Baselines3 available: {'PPO' in dir()}")
        return True
    except ImportError as e:
        print(f"Error importing libraries: {e}")
        print("Installing required packages...")
        subprocess.run("pip install gymnasium stable-baselines3", shell=True)
        return False

def force_checkpoint(model_dir="models", label="manual"):
    """Force the current training to save a checkpoint."""
    import glob
    # First, find the currently running Python process that's training
    try:
        # Find latest checkpoint to get info from
        checkpoints = glob.glob(f"{model_dir}/ppo_delivery_*.zip")
        if not checkpoints:
            print("No checkpoints found to force save.")
            return False
        
        # Create a forced checkpoint file that tells ProgressCallback to save
        force_file = f"{model_dir}/force_checkpoint_{label}.txt"
        with open(force_file, "w") as f:
            f.write(f"Requested at {datetime.datetime.now()}")
        
        print(f"✅ Requested forced checkpoint: {force_file}")
        return True
    except Exception as e:
        print(f"❌ Failed to force checkpoint: {e}")
        return False

def train_agent(config, resume_path=None, save_to_drive=False, drive_path=None):
    """Train RL agent with enhanced error handling and logging."""
    print("\n" + "=" * 80)
    print(" TRAINING RL AGENT ".center(80, "="))
    print("=" * 80 + "\n")
    
    env_config = config['environment']
    train_config = config['training']
    
    start_time = time.time()
    
    # Build command with proper quoting and error handling
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
    
    if resume_path:
        cmd.extend(["--resume", resume_path])
    
    # Convert command to strings
    cmd = [str(item) for item in cmd]
    
    # Print the command
    print("Running command:", " ".join(cmd))
    
    # Set up checkpoint monitoring
    last_checkpoint_time = time.time()
    checkpoint_interval = 3600  # Force a checkpoint at least every hour
    
    try:
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Set up tracking variables
        last_output_time = time.time()
        output_timeout = 600  # 10 minutes
        iteration_pattern = "iterations"
        fps_pattern = "fps"
        timesteps_pattern = "total_timesteps"
        
        # Process output line by line
        for line in process.stdout:
            # Print the line
            print(line, end='')
            
            # Update the last output time
            last_output_time = time.time()
            
            # Check if we need to force a checkpoint
            if time.time() - last_checkpoint_time > checkpoint_interval:
                print("\n" + "=" * 80)
                print(" FORCING HOURLY CHECKPOINT ".center(80, "="))
                print("=" * 80 + "\n")
                force_checkpoint(label=f"hourly_{int((time.time() - start_time) / 3600)}h")
                last_checkpoint_time = time.time()
            
            # If we have Google Drive savings enabled, periodically copy checkpoints
            if save_to_drive and drive_path and time.time() - last_checkpoint_time > 1800:  # Every 30 minutes
                print("\nBacking up checkpoints to Google Drive...")
                copy_to_drive("models", drive_path)
            
            # Check for training progress indicators
            if iteration_pattern in line or fps_pattern in line or timesteps_pattern in line:
                # Extract training statistics if available
                try:
                    if "time/iterations" in line:
                        iterations = int(line.split("|")[2].strip())
                        print(f"Training progress: {iterations} iterations completed")
                    if "time/fps" in line:
                        fps = float(line.split("|")[2].strip())
                        print(f"Training speed: {fps:.1f} FPS")
                    if "time/total_timesteps" in line:
                        timesteps = int(line.split("|")[2].strip())
                        progress = (timesteps / train_config['total_timesteps']) * 100
                        elapsed = time.time() - start_time
                        remaining = (elapsed / progress) * (100 - progress) if progress > 0 else float('inf')
                        print(f"Training progress: {timesteps}/{train_config['total_timesteps']} timesteps ({progress:.1f}%)")
                        print(f"Elapsed time: {elapsed/3600:.1f} hours, Estimated remaining: {remaining/3600:.1f} hours")
                except Exception:
                    pass  # If we can't parse progress, just continue
        
        # Wait for the process to complete
        process.wait()
        print(f"\nTraining completed with return code {process.returncode}")
        
        # Save final model to Google Drive if enabled
        if save_to_drive and drive_path:
            print("\nSaving final checkpoints to Google Drive...")
            copy_to_drive("models", drive_path)
        
        return process.returncode == 0
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Attempting to save checkpoint...")
        force_checkpoint(label="interruption")
        
        # Wait a moment for the checkpoint to be saved
        time.sleep(5)
        
        # Save to Drive if enabled
        if save_to_drive and drive_path:
            print("\nSaving checkpoints to Google Drive before termination...")
            copy_to_drive("models", drive_path)
        
        # Send termination signal
        try:
            process.terminate()
            print("Process terminated.")
        except:
            print("Could not terminate process.")
        
        return False
    except Exception as e:
        print(f"\n\nError during training: {e}")
        traceback.print_exc()
        
        # Try to save a checkpoint
        force_checkpoint(label="error")
        
        # Save to Drive if enabled
        if save_to_drive and drive_path:
            print("\nSaving checkpoints to Google Drive after error...")
            copy_to_drive("models", drive_path)
        
        return False

def main():
    # Parse arguments
    args = parse_args()
    
    # Set up directories
    ensure_directories()
    
    # Create default config if needed
    create_default_config()
    
    # Load and update configuration
    config = load_config(args.config)
    config = update_config_with_args(config, args)
    
    # Set up environment
    if not setup_environment():
        print("Environment setup failed. Please check dependencies.")
        return 1
    
    # Print configuration summary
    print("\n" + "=" * 80)
    print(" TRAINING CONFIGURATION ".center(80, "="))
    print("=" * 80)
    print(f"Total timesteps: {config['training']['total_timesteps']}")
    print(f"Parallel environments: {config['training']['n_envs']}")
    print(f"Checkpoint frequency: {config['training']['checkpoint_freq']} timesteps")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    if args.resume:
        print(f"Resuming from: {args.resume}")
    if args.save_to_drive:
        print(f"Saving to Google Drive: {args.drive_path}")
    print("=" * 80 + "\n")
    
    # Check for Google Drive if requested
    if args.save_to_drive:
        if not os.path.exists("/content/drive"):
            print("Google Drive not mounted but save_to_drive is enabled.")
            print("Run the following in a cell before continuing:")
            print("from google.colab import drive")
            print("drive.mount('/content/drive')")
            return 1
        
        # Create the directory in Google Drive
        os.makedirs(args.drive_path, exist_ok=True)
    
    # Train the agent
    success = train_agent(
        config, 
        resume_path=args.resume,
        save_to_drive=args.save_to_drive,
        drive_path=args.drive_path
    )
    
    if success:
        print("\n" + "=" * 80)
        print(" TRAINING COMPLETED SUCCESSFULLY ".center(80, "="))
        print("=" * 80 + "\n")
        return 0
    else:
        print("\n" + "=" * 80)
        print(" TRAINING FAILED OR WAS INTERRUPTED ".center(80, "="))
        print("=" * 80 + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 