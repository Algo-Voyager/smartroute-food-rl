#!/usr/bin/env python3
"""
RL agent for the food delivery system using Stable-Baselines3 PPO.

This module provides wrapper classes and utility functions to train and run
RL agents for the food delivery environment.
"""

import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
import torch as th
import time

def make_env(env_id, env_kwargs, rank, seed=0):
    """
    Utility function for multiprocessed env.
    
    Args:
        env_id: environment ID string or class
        env_kwargs: keyword arguments for environment initialization
        rank: index of the subprocess
        seed: Random seed
        
    Returns:
        Initialized gym environment
    """
    def _init():
        if isinstance(env_id, str):
            env = gym.make(env_id, **env_kwargs)
        else:
            env = env_id(**env_kwargs)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    
    set_random_seed(seed)
    return _init

class DeliveryRLAgent:
    """
    Reinforcement Learning agent for food delivery routing.
    
    Uses Stable-Baselines3 PPO with MlpPolicy for training.
    """
    
    def __init__(
        self,
        env_class,
        env_kwargs,
        n_envs=8,
        policy="MlpPolicy",
        policy_kwargs=None,
        gamma=0.99,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./logs/",
        device="auto"
    ):
        """
        Initialize the RL agent.
        
        Args:
            env_class: Environment class
            env_kwargs: Keyword arguments for environment initialization
            n_envs: Number of parallel environments
            policy: Policy network type ("MlpPolicy" or "MultiInputPolicy")
            policy_kwargs: Keyword arguments for policy initialization
            gamma: Discount factor
            learning_rate: Learning rate
            n_steps: Number of steps per rollout
            batch_size: Minibatch size
            ent_coef: Entropy coefficient
            clip_range: PPO clip range
            verbose: Verbosity level
            tensorboard_log: Directory for tensorboard logs
            device: Device to run on ("auto", "cpu", "cuda", "cuda:0", etc.)
        """
        self.env_class = env_class
        self.env_kwargs = env_kwargs
        self.n_envs = n_envs
        self.tensorboard_log = tensorboard_log
        
        # Default policy network configuration if not provided
        if policy_kwargs is None:
            policy_kwargs = {
                "net_arch": [256, 256, 256],
                "activation_fn": th.nn.ReLU
            }
        
        # Create vectorized environment
        env = SubprocVecEnv([
            make_env(env_class, env_kwargs, i) for i in range(n_envs)
        ])
        
        # Normalize observations and rewards
        self.env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=gamma
        )
        
        # Create PPO agent
        self.model = PPO(
            policy=policy,
            env=self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            ent_coef=ent_coef,
            clip_range=clip_range,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device
        )
    
    def train(self, total_timesteps=1e7, checkpoint_freq=100000, eval_freq=50000, model_dir="./models"):
        """
        Train the RL agent.
        
        Args:
            total_timesteps: Total number of timesteps to train for
            checkpoint_freq: Frequency of checkpointing (in timesteps)
            eval_freq: Frequency of evaluation (in timesteps)
            model_dir: Directory to save models and checkpoints
            
        Returns:
            Trained model
        """
        # Create directory for checkpoints
        os.makedirs(model_dir, exist_ok=True)
        best_model_dir = os.path.join(model_dir, "best")
        os.makedirs(best_model_dir, exist_ok=True)
        
        # Create callbacks
        # Save an initial checkpoint after 10 steps to verify saving works
        initial_checkpoint_callback = CheckpointCallback(
            save_freq=10,
            save_path=model_dir,
            name_prefix="ppo_delivery_initial",
            save_vecnormalize=True,
            verbose=1
        )
        
        # Regular checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=model_dir,
            name_prefix="ppo_delivery",
            save_vecnormalize=True,
            verbose=1
        )
        
        # Progress tracking callback
        progress_callback = ProgressCallback(verbose=1)
        
        # Create evaluation environment
        eval_env = SubprocVecEnv([
            make_env(self.env_class, self.env_kwargs, i) for i in range(1)
        ])
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=self.model.gamma,
            training=False
        )
        
        # Copy normalization statistics from training environment
        eval_env.obs_rms = self.env.obs_rms
        eval_env.ret_rms = self.env.ret_rms
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=best_model_dir,
            log_path=self.tensorboard_log,
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )
        
        # Train the agent
        self.model.learn(
            total_timesteps=int(total_timesteps),
            callback=[initial_checkpoint_callback, checkpoint_callback, eval_callback, progress_callback]
        )
        
        # Save the final model
        final_model_path = os.path.join(model_dir, "ppo_delivery_final")
        self.model.save(final_model_path)
        
        # Save normalization parameters
        vec_normalize_path = os.path.join(model_dir, "vec_normalize.pkl")
        self.env.save(vec_normalize_path)
        
        return self.model
    
    def load(self, model_path, vec_normalize_path=None):
        """
        Load a pre-trained model.
        
        Args:
            model_path: Path to the model file
            vec_normalize_path: Path to the VecNormalize statistics
            
        Returns:
            Loaded model
        """
        self.model = PPO.load(model_path, self.env)
        
        if vec_normalize_path and os.path.exists(vec_normalize_path):
            self.env = VecNormalize.load(vec_normalize_path, self.env)
            self.env.training = False
            self.env.norm_reward = False
        
        return self.model
    
    def act(self, observation, deterministic=True):
        """
        Choose an action based on the current observation.
        
        Args:
            observation: Environment observation
            deterministic: Whether to use deterministic actions
            
        Returns:
            action: Action to take
        """
        # Convert observation to the format expected by the model
        if isinstance(observation, dict):
            # Create a batch of 1 observation
            obs_dict = {}
            for key, value in observation.items():
                if isinstance(value, np.ndarray):
                    # Add batch dimension if needed
                    if value.ndim == 1:
                        value = value.reshape(1, -1)
                    obs_dict[key] = value
                else:
                    obs_dict[key] = np.array([value])
            
            # Normalize observation
            obs = self.env.normalize_obs(obs_dict)
        else:
            # Normalize observation
            obs = self.env.normalize_obs(np.array([observation]))
        
        # Get action from model
        action, _ = self.model.predict(obs, deterministic=deterministic)
        
        # Return the first action (since we have a batch of 1)
        return action[0]

class CustomCallback(BaseCallback):
    """
    Custom callback for logging additional metrics during training.
    """
    
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
    
    def _on_step(self):
        # Access to environment metrics through self.locals["env"]
        env = self.locals["env"].envs[0]
        
        # Log metrics
        if hasattr(env, "total_deliveries"):
            self.logger.record("metrics/deliveries", env.total_deliveries)
        if hasattr(env, "total_rejects"):
            self.logger.record("metrics/rejects", env.total_rejects)
        if hasattr(env, "total_repositions"):
            self.logger.record("metrics/repositions", env.total_repositions)
        if hasattr(env, "total_delivery_time") and hasattr(env, "total_deliveries"):
            if env.total_deliveries > 0:
                avg_delivery_time = env.total_delivery_time / env.total_deliveries
                self.logger.record("metrics/avg_delivery_time", avg_delivery_time)
        
        return True 

class ProgressCallback(BaseCallback):
    """
    Callback to print training progress information.
    """
    def __init__(self, verbose=1):
        super(ProgressCallback, self).__init__(verbose)
        self.last_time = time.time()
        self.training_start_time = time.time()
        self.last_hourly_checkpoint_time = time.time()
        self.iteration_time = []
        self.last_saved_checkpoint = None
        
    def _on_training_start(self):
        print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.training_start_time = time.time()
        self.last_hourly_checkpoint_time = time.time()
        
    def _on_step(self):
        current_time = time.time()
        
        # Check if an hour has passed since the last hourly checkpoint
        if current_time - self.last_hourly_checkpoint_time >= 3600:  # 3600 seconds = 1 hour
            # Save an hourly checkpoint
            checkpoint_path = os.path.join(
                os.path.dirname(self.model.tensorboard_log),
                "models",
                f"ppo_delivery_hourly_{int(current_time - self.training_start_time)}s.zip"
            )
            self.model.save(checkpoint_path)
            
            # Save VecNormalize
            vec_normalize_path = os.path.join(
                os.path.dirname(self.model.tensorboard_log),
                "models",
                "vec_normalize.pkl"
            )
            self.training_env.save(vec_normalize_path)
            
            print(f"Hourly checkpoint saved: {checkpoint_path}")
            self.last_hourly_checkpoint_time = current_time
        
        if self.n_calls % 100 == 0:
            elapsed = current_time - self.last_time
            self.iteration_time.append(elapsed)
            
            if len(self.iteration_time) > 10:
                self.iteration_time.pop(0)
                
            avg_time = sum(self.iteration_time) / len(self.iteration_time)
            steps_per_second = 100 / avg_time
            
            # Calculate estimated time for next checkpoint
            if hasattr(self.model, 'num_timesteps') and hasattr(self.model, '_last_checkpoint_step'):
                steps_to_next_checkpoint = 240000 - (self.model.num_timesteps % 240000)
                time_to_next_checkpoint = steps_to_next_checkpoint / steps_per_second
                next_checkpoint_time = time.strftime('%H:%M:%S', time.gmtime(current_time + time_to_next_checkpoint))
                print(f"Next regular checkpoint in ~{time_to_next_checkpoint/60:.1f} minutes at ~{next_checkpoint_time}")
            
            # Calculate time until next hourly checkpoint
            time_to_hourly = 3600 - (current_time - self.last_hourly_checkpoint_time)
            next_hourly_time = time.strftime('%H:%M:%S', time.gmtime(current_time + time_to_hourly))
            print(f"Next hourly checkpoint in ~{time_to_hourly/60:.1f} minutes at ~{next_hourly_time}")
            
            print(f"Step: {self.n_calls}, "
                  f"FPS: {steps_per_second:.2f}, "
                  f"Elapsed: {elapsed:.2f}s, "
                  f"Total timesteps: {self.model.num_timesteps}")
            
            self.last_time = current_time
            
            # Look for newest checkpoint file
            import glob
            checkpoints = glob.glob(os.path.join(os.path.dirname(self.model.tensorboard_log), 
                                               "models/ppo_delivery_*.zip"))
            if checkpoints:
                newest_checkpoint = max(checkpoints, key=os.path.getctime)
                if newest_checkpoint != self.last_saved_checkpoint:
                    self.last_saved_checkpoint = newest_checkpoint
                    print(f"New checkpoint saved: {newest_checkpoint}")
        
        return True 