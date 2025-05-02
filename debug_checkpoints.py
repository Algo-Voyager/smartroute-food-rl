#!/usr/bin/env python3
"""
Debugging script for checkpoint issues in food delivery RL project.
This script checks for filesystem permissions, verifies checkpoint functionality,
and fixes common issues that might prevent checkpoint creation.
"""

import os
import sys
import glob
import time
import json
import traceback
import importlib
import subprocess
from datetime import datetime

def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def check_directory_permissions():
    """Check if the models directory exists and has proper permissions."""
    print_section("CHECKING DIRECTORY PERMISSIONS")
    
    # Check if models directory exists
    if not os.path.exists("models"):
        print("Models directory doesn't exist. Creating it...")
        try:
            os.makedirs("models", exist_ok=True)
            print("✅ Models directory created successfully.")
        except Exception as e:
            print(f"❌ ERROR creating models directory: {e}")
            return False
    else:
        print("✅ Models directory exists.")
    
    # Check permissions
    try:
        test_file = "models/test_permissions.txt"
        with open(test_file, "w") as f:
            f.write("Testing write permissions")
        
        print(f"✅ Successfully wrote to {test_file}")
        
        # Clean up
        os.remove(test_file)
        print(f"✅ Successfully removed {test_file}")
        return True
    except Exception as e:
        print(f"❌ ERROR with file operations in models directory: {e}")
        return False

def check_checkpoint_config():
    """Check configuration for checkpoint frequency."""
    print_section("CHECKING CHECKPOINT CONFIGURATION")
    
    # Check if config file exists
    config_path = "config/default_config.json"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found at {config_path}")
        return False
    
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        
        checkpoint_freq = config.get("training", {}).get("checkpoint_freq", None)
        if checkpoint_freq is None:
            print("❌ checkpoint_freq not found in config file")
            return False
        
        print(f"✅ Checkpoint frequency set to {checkpoint_freq} timesteps")
        
        # If checkpoint frequency is too high, suggest lowering it for testing
        if checkpoint_freq > 10000:
            print(f"⚠️  Checkpoint frequency of {checkpoint_freq} timesteps may take a long time.")
            print("   Consider temporarily lowering it to 1000-5000 for testing.")
        
        return True
    except Exception as e:
        print(f"❌ ERROR reading config file: {e}")
        return False

def find_existing_checkpoints():
    """Find any existing checkpoints."""
    print_section("CHECKING EXISTING CHECKPOINTS")
    
    checkpoints = glob.glob("models/ppo_delivery_*.zip")
    if not checkpoints:
        print("ℹ️  No existing checkpoints found.")
        return []
    
    print(f"✅ Found {len(checkpoints)} checkpoint(s):")
    for cp in sorted(checkpoints):
        mod_time = datetime.fromtimestamp(os.path.getmtime(cp))
        file_size = os.path.getsize(cp) / (1024 * 1024)  # Convert to MB
        print(f"   - {cp} ({file_size:.2f} MB, modified: {mod_time})")
    
    return checkpoints

def test_checkpoint_creation():
    """Test if stable-baselines3 can create a checkpoint."""
    print_section("TESTING CHECKPOINT CREATION")
    
    try:
        import stable_baselines3
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        print(f"✅ stable-baselines3 version: {stable_baselines3.__version__}")
        
        # Create a simple environment for testing
        def make_env():
            import gymnasium as gym
            return gym.make("CartPole-v1")
        
        env = DummyVecEnv([make_env])
        
        # Create a simple PPO model
        model = PPO("MlpPolicy", env, verbose=1)
        
        # Try to save a checkpoint
        test_checkpoint = "models/test_checkpoint.zip"
        print(f"Testing checkpoint creation at {test_checkpoint}...")
        model.save(test_checkpoint)
        
        if os.path.exists(test_checkpoint):
            print(f"✅ Test checkpoint successfully created!")
            
            # Clean up
            os.remove(test_checkpoint)
            print(f"✅ Test checkpoint removed.")
            return True
        else:
            print(f"❌ Failed to create test checkpoint.")
            return False
    except Exception as e:
        print(f"❌ ERROR during checkpoint test: {e}")
        traceback.print_exc()
        return False

def check_disk_space():
    """Check available disk space."""
    print_section("CHECKING DISK SPACE")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        
        # Convert to GB
        total_gb = total / (1024**3)
        used_gb = used / (1024**3)
        free_gb = free / (1024**3)
        
        print(f"Total disk space: {total_gb:.2f} GB")
        print(f"Used disk space: {used_gb:.2f} GB")
        print(f"Free disk space: {free_gb:.2f} GB")
        
        if free_gb < 1.0:
            print("❌ Less than 1 GB of free disk space available. This may cause checkpoint failures.")
            return False
        else:
            print(f"✅ Sufficient disk space available ({free_gb:.2f} GB).")
            return True
    except Exception as e:
        print(f"❌ ERROR checking disk space: {e}")
        return False

def monitor_checkpoint_creation(duration=300):
    """Monitor for checkpoint creation for a specified duration (in seconds)."""
    print_section(f"MONITORING FOR CHECKPOINT CREATION (for {duration}s)")
    
    # Get initial list of checkpoints
    initial_checkpoints = set(glob.glob("models/ppo_delivery_*.zip"))
    print(f"Starting with {len(initial_checkpoints)} existing checkpoints.")
    
    start_time = time.time()
    end_time = start_time + duration
    
    try:
        while time.time() < end_time:
            # Check for new checkpoints
            current_checkpoints = set(glob.glob("models/ppo_delivery_*.zip"))
            new_checkpoints = current_checkpoints - initial_checkpoints
            
            if new_checkpoints:
                print(f"\n✅ New checkpoint(s) detected:")
                for cp in sorted(new_checkpoints):
                    mod_time = datetime.fromtimestamp(os.path.getmtime(cp))
                    file_size = os.path.getsize(cp) / (1024 * 1024)  # Convert to MB
                    print(f"   - {cp} ({file_size:.2f} MB, modified: {mod_time})")
                initial_checkpoints = current_checkpoints
            
            # Print progress
            elapsed = time.time() - start_time
            remaining = duration - elapsed
            print(f"\rMonitoring for new checkpoints... {elapsed:.0f}s elapsed, {remaining:.0f}s remaining", end="")
            
            # Sleep for a bit
            time.sleep(5)
        
        print("\n\nMonitoring period ended.")
        return True
    except KeyboardInterrupt:
        print("\n\nMonitoring interrupted by user.")
        return False
    except Exception as e:
        print(f"\n\n❌ ERROR during monitoring: {e}")
        return False

def force_checkpoint_creation():
    """Force creation of a checkpoint by modifying ProgressCallback."""
    print_section("FORCING CHECKPOINT CREATION")
    
    try:
        # First check if we can find the training script
        if not os.path.exists("agents/rl_agent.py"):
            print("❌ agents/rl_agent.py not found. Cannot force checkpoint creation.")
            return False
        
        print("Attempting to force checkpoint creation...")
        print("1. Creating a temporary script to force a checkpoint...")
        
        with open("force_checkpoint.py", "w") as f:
            f.write("""#!/usr/bin/env python3
import os
import sys
import glob
import time
import torch

sys.path.insert(0, ".")

# Find latest checkpoint to get model path
checkpoints = glob.glob("models/ppo_delivery_*.zip")
if not checkpoints:
    print("No checkpoints found to resume from. Run training first.")
    sys.exit(1)

checkpoints.sort(key=lambda x: os.path.getmtime(x))
latest_checkpoint = checkpoints[-1]
print(f"Found latest checkpoint: {latest_checkpoint}")

# Try to manually save a copy of the model
try:
    print("Attempting to load and re-save model...")
    from stable_baselines3 import PPO
    
    # Load the model
    model = PPO.load(latest_checkpoint)
    
    # Save a manual checkpoint
    forced_checkpoint = "models/forced_checkpoint.zip"
    model.save(forced_checkpoint)
    
    if os.path.exists(forced_checkpoint):
        print(f"✅ Successfully created forced checkpoint at {forced_checkpoint}")
    else:
        print(f"❌ Failed to create forced checkpoint")
except Exception as e:
    print(f"❌ Error during forced checkpoint creation: {e}")
    import traceback
    traceback.print_exc()
""")
        
        # Make it executable
        os.chmod("force_checkpoint.py", 0o755)
        
        # Run the script
        print("\n2. Running the script to force checkpoint creation...")
        subprocess.run(["python", "force_checkpoint.py"])
        
        # Check if it worked
        if os.path.exists("models/forced_checkpoint.zip"):
            print("\n✅ Successfully forced checkpoint creation!")
            return True
        else:
            print("\n❌ Failed to force checkpoint creation.")
            return False
    except Exception as e:
        print(f"❌ ERROR during forced checkpoint creation: {e}")
        return False

def main():
    """Run all checks."""
    print_section("CHECKPOINT DEBUGGING UTILITY")
    print("This script diagnoses issues with checkpoint creation in the RL training system.")
    
    # Run all checks
    permissions_ok = check_directory_permissions()
    config_ok = check_checkpoint_config()
    existing_checkpoints = find_existing_checkpoints()
    disk_space_ok = check_disk_space()
    
    # Only test checkpoint creation if permissions are OK
    checkpoint_test_ok = test_checkpoint_creation() if permissions_ok else False
    
    # Summarize results
    print_section("SUMMARY")
    print(f"✓ Directory permissions: {'OK' if permissions_ok else 'FAILED'}")
    print(f"✓ Checkpoint configuration: {'OK' if config_ok else 'FAILED'}")
    print(f"✓ Existing checkpoints: {len(existing_checkpoints)}")
    print(f"✓ Disk space: {'OK' if disk_space_ok else 'INSUFFICIENT'}")
    print(f"✓ Checkpoint test: {'OK' if checkpoint_test_ok else 'FAILED'}")
    
    # Overall assessment
    if all([permissions_ok, config_ok, disk_space_ok, checkpoint_test_ok]):
        print("\n✅ All checks PASSED. Checkpoint system should be working properly.")
        
        if not existing_checkpoints:
            choice = input("\nNo existing checkpoints found. Would you like to:\n"
                          "1. Monitor for checkpoint creation (wait for 5 minutes)\n"
                          "2. Force checkpoint creation (if training is already running)\n"
                          "3. Exit\n"
                          "Enter choice [1-3]: ")
            
            if choice == "1":
                monitor_checkpoint_creation(300)  # 5 minutes
            elif choice == "2":
                force_checkpoint_creation()
        else:
            print("\nCheckpoints exist and everything seems to be working correctly.")
    else:
        print("\n❌ Some checks FAILED. Please address the issues above.")
        
        if not permissions_ok:
            print("\nTry running: chmod -R 755 models/")
        
        if not config_ok:
            print("\nCheck your configuration file and make sure checkpoint_freq is set properly.")
        
        if not disk_space_ok:
            print("\nFree up some disk space to ensure checkpoints can be saved.")
        
        if not checkpoint_test_ok:
            print("\nThere may be an issue with Stable-Baselines3 or its dependencies.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 