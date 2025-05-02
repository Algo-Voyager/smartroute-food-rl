#!/usr/bin/env python3
"""
Wrapper script to run training with logging to a file.
Output will be logged to 'training_log.txt' which can be monitored with:
    tail -f training_log.txt
"""

import os
import sys
import time
import datetime
import subprocess
import signal
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run training with logging")
    parser.add_argument("--log_file", type=str, default="training_log.txt",
                       help="Path to log file")
    parser.add_argument("--cmd", type=str, default="python main.py --train",
                       help="Command to run")
    parser.add_argument("--monitor", action="store_true",
                       help="Monitor log file with tail -f")
    return parser.parse_args()

def setup_logging(log_file):
    # Create log directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        
    # Create or clear the log file
    with open(log_file, 'w') as f:
        f.write(f"=== Training Log started at {datetime.datetime.now()} ===\n\n")
    
    return log_file

def run_with_logging(cmd, log_file):
    """Run a command and log output to a file."""
    print(f"Running command: {cmd}")
    print(f"Logging output to: {log_file}")
    print(f"You can monitor progress with: tail -f {log_file}")
    
    # Redirect output to the log file
    with open(log_file, 'a') as log:
        # Log the command
        log.write(f"Command: {cmd}\n")
        log.write(f"Started at: {datetime.datetime.now()}\n")
        log.write("-" * 80 + "\n\n")
        log.flush()
        
        # Start the process
        process = subprocess.Popen(
            cmd.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Read output and write to log file
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()  # Make sure it's written immediately for tail -f
        except KeyboardInterrupt:
            print(f"\nReceived keyboard interrupt. Terminating process...")
            process.send_signal(signal.SIGINT)  # Send SIGINT (Ctrl+C)
            time.sleep(2)  # Give it a moment to handle the signal
            if process.poll() is None:  # If still running
                print("Process not responding to SIGINT. Terminating...")
                process.terminate()
            
            log.write(f"\n=== Process interrupted at {datetime.datetime.now()} ===\n")
            return process.returncode
        
        # Wait for the process to complete
        process.wait()
        log.write(f"\n=== Process completed at {datetime.datetime.now()} with return code {process.returncode} ===\n")
        
        return process.returncode

def monitor_log(log_file):
    """Monitor the log file with tail -f."""
    tail_cmd = f"tail -f {log_file}"
    try:
        subprocess.run(tail_cmd, shell=True)
    except KeyboardInterrupt:
        print("\nStopped monitoring log file.")

def main():
    args = parse_args()
    log_file = setup_logging(args.log_file)
    
    returncode = run_with_logging(args.cmd, log_file)
    
    if args.monitor:
        monitor_log(log_file)
    
    return returncode

if __name__ == "__main__":
    sys.exit(main()) 