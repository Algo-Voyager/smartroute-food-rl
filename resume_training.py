#!/usr/bin/env python3
import os
import sys
import glob
import json
import subprocess
import argparse

def find_latest_checkpoint():
    """Find the latest checkpoint file in the models directory."""
    # Check if models directory exists
    if not os.path.exists('models'):
        print("Models directory not found. Creating it.")
        os.makedirs('models', exist_ok=True)
        return None
    
    # First, check for checkpoint counter file
    counter_file = 'models/checkpoint_counter.json'
    if os.path.exists(counter_file):
        try:
            with open(counter_file, 'r') as f:
                counter_data = json.load(f)
                last_checkpoint = counter_data.get('last_checkpoint')
                if last_checkpoint and os.path.exists(last_checkpoint):
                    print(f"Found latest checkpoint from counter file: {last_checkpoint}")
                    print(f"Total training hours so far: {counter_data.get('hours_elapsed', 0)}")
                    return last_checkpoint
        except Exception as e:
            print(f"Warning: Could not load checkpoint counter: {e}")
    
    # Look for hourly checkpoints first
    hourly_checkpoints = glob.glob('models/ppo_delivery_hourly_*h.zip')
    if hourly_checkpoints:
        # Sort by hour number
        hourly_checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('h')[0]))
        latest_hourly = hourly_checkpoints[-1]
        print(f"Found latest hourly checkpoint: {latest_hourly}")
        return latest_hourly
    
    # Find all step-based checkpoint files
    checkpoints = glob.glob('models/ppo_delivery_*.zip')
    
    # Also check for final model
    final_model = 'models/ppo_delivery_final.zip'
    if os.path.exists(final_model):
        checkpoints.append(final_model)
    
    if not checkpoints:
        print("No checkpoints found in models/ directory")
        return None
    
    # Sort by timestep number (last element before .zip extension)
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if x.split('_')[-1].split('.')[0].isdigit() else float('inf'))
    
    return checkpoints[-1]

def main():
    parser = argparse.ArgumentParser(description="Resume training from latest checkpoint")
    parser.add_argument("--manual_checkpoint", type=str, help="Manually specify checkpoint path")
    args = parser.parse_args()
    
    checkpoint = args.manual_checkpoint if args.manual_checkpoint else find_latest_checkpoint()
    
    if not checkpoint:
        print("No checkpoint found. Starting fresh training.")
        cmd = ["python", "training/train_ppo.py"]
    else:
        print(f"Resuming from checkpoint: {checkpoint}")
        # Convert to absolute path if needed
        checkpoint = os.path.abspath(checkpoint)
        
        # Check if vec_normalize.pkl exists in the same directory
        vec_normalize = os.path.join(os.path.dirname(checkpoint), "vec_normalize.pkl")
        vec_normalize_arg = ""
        if os.path.exists(vec_normalize):
            print(f"Found VecNormalize file: {vec_normalize}")
        else:
            print("Warning: No VecNormalize file found!")
        
        cmd = ["python", "training/train_ppo.py", "--resume", checkpoint]
    
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
