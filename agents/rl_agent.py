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
    
    def train(self, total_timesteps=1e7, checkpoint_freq=100000, eval_freq=50000):
        """
        Train the RL agent.
        
        Args:
            total_timesteps: Total number of timesteps to train for
            checkpoint_freq: Frequency of checkpointing (in timesteps)
            eval_freq: Frequency of evaluation (in timesteps)
            
        Returns:
            Trained model
        """
        # Create directory for checkpoints
        os.makedirs("./models", exist_ok=True)
        
        # Create callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path="./models/",
            name_prefix="ppo_delivery"
        )
        
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
            best_model_save_path="./models/best/",
            log_path="./logs/",
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )
        
        # Train the agent
        self.model.learn(
            total_timesteps=int(total_timesteps),
            callback=[checkpoint_callback, eval_callback]
        )
        
        # Save the final model
        self.model.save("./models/ppo_delivery_final")
        
        # Save normalization parameters
        self.env.save("./models/vec_normalize.pkl")
        
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