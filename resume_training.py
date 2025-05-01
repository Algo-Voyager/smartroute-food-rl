#!/usr/bin/env python3
import os
import sys
import glob
import subprocess
import argparse

def find_latest_checkpoint():
    # Find all checkpoint files
    checkpoints = glob.glob('models/ppo_delivery_*.zip')
    if not checkpoints:
        print("No checkpoints found in models/ directory")
        return None
    
    # Sort by timestep number
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
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
        # Check if vec_normalize.pkl exists in the same directory
        vec_normalize = os.path.join(os.path.dirname(checkpoint), "vec_normalize.pkl")
        
        cmd = ["python", "training/train_ppo.py", "--resume", checkpoint]
    
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
