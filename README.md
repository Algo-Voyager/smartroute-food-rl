# Food Delivery Reinforcement Learning

A dynamic routing system for on-demand meal delivery leveraging reinforcement learning, capable of handling up to 10⁸ orders.

---

## Table of Contents

1. [Project Structure](#project-structure)  
2. [Setup](#setup)  
3. [Usage](#usage)  
   - [Generate Data](#generate-data)  
   - [Training](#training)  
   - [Pause & Resume Training](#pause--resume-training)  
   - [Evaluation](#evaluation)  
   - [Visualization](#visualization)  
4. [Running the Full Pipeline](#running-the-full-pipeline)  
5. [Utility Scripts](#utility-scripts)
   - [Checkpoint Debugging](#checkpoint-debugging)
   - [Logging and Monitoring](#logging-and-monitoring)
   - [Colab Integration](#colab-integration)
6. [Features](#features)  

---

## Project Structure

```
.
├── config/              # JSON config files (environment, training, evaluation)
├── data/                # Synthetic data generators and datasets
├── envs/                # RL environment implementations
├── agents/              # RL models and baseline heuristics
├── training/            # Training scripts and utilities
├── evaluation/          # Evaluation metrics and analysis tools
├── visualization/       # Scripts for plotting and simulation playback
├── logs/                # TensorBoard log directory
├── models/              # Saved checkpoints and normalization state
├── resume_training.py   # Script to resume from the latest or specified checkpoint
├── run_with_logging.py  # Script to run training with logging to a file
├── debug_checkpoints.py # Utility for diagnosing checkpoint issues
├── colab_train.py       # Optimized training script for Google Colab
├── requirements.txt     # Python dependencies
└── main.py              # Entry-point for full pipeline
```

---

## Setup

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yourusername/mtp-food-delivery-RL.git
   cd mtp-food-delivery-RL
   ```

2. **Create & activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Generate Data

```bash
python data/generate_data.py \
  --n_restaurants 200 \
  --n_drivers 50 \
  --n_orders 100000 \
  --grid_size 20
```

All parameters have sensible defaults; omit flags to use them.

### Training

```bash
python training/train_ppo.py \
  --config config/default_config.json
```

* Checkpoints (every 240,000 steps) go to `models/`.
* TensorBoard logs go to `logs/` (view with `tensorboard --logdir logs`).

You can override any hyperparameter on the CLI, for example:

```bash
python training/train_ppo.py \
  --n_envs 16 \
  --learning_rate 1e-4 \
  --total_timesteps 5e6
```

### Pause & Resume Training

* **Pause**: Ctrl+C
* **Resume (latest checkpoint)**:

  ```bash
  python resume_training.py
  ```
* **Resume (specific checkpoint)**:

  ```bash
  python resume_training.py \
    --manual_checkpoint models/ppo_delivery_480000.zip
  ```

### Evaluation

```bash
python evaluation/evaluate.py \
  --n_episodes 10 \
  --model_path models/ppo_delivery_final.zip \
  --vec_normalize_path models/vec_normalize.pkl \
  --output_dir evaluation/
```

### Visualization

```bash
python visualization/visualize.py \
  --input_dir evaluation/ \
  --output_dir visualization/
```

---

## Running the Full Pipeline

```bash
python main.py --all
```

This will sequentially perform:

1. Data generation
2. Model training
3. Evaluation
4. Visualization

---

## Utility Scripts

The project includes several utility scripts to help with training monitoring, debugging, and Colab integration.

### Checkpoint Debugging

The `debug_checkpoints.py` script diagnoses and fixes checkpoint creation issues:

```bash
python debug_checkpoints.py
```

This utility:
- Checks directory permissions and filesystem access
- Verifies checkpoint configuration
- Tests if the system can create checkpoints
- Monitors for new checkpoints in real time
- Can force a checkpoint creation if needed

Use this script when troubleshooting checkpoint saving issues or verifying that checkpoints are being created properly.

### Logging and Monitoring

Use `run_with_logging.py` to redirect all training output to a log file and monitor it in real time:

```bash
# In first terminal - run training with logging
python run_with_logging.py

# In second terminal - monitor the log file
tail -f training_log.txt
```

By default, it will run `python main.py --train` and log the output to `training_log.txt`. You can customize both the command and the log file:

```bash
python run_with_logging.py --cmd "python training/train_ppo.py --n_envs 32" --log_file "high_env_training.log"
```

### Colab Integration

For training in Google Colab, use the optimized training script:

```bash
python colab_train.py --save_to_drive
```

Features of Colab integration:
- More frequent checkpoints (every 50K steps vs 240K)
- Forced hourly checkpoints regardless of step count
- Automatic backup to Google Drive
- Enhanced error handling and progress reporting
- Optimized parameters for Colab's runtime constraints

Options include:
```bash
python colab_train.py \
  --steps 5000000 \
  --envs 16 \
  --checkpoint_freq 50000 \
  --save_to_drive \
  --drive_path "/content/drive/MyDrive/food_delivery_rl"
```

---

## Features

* **Synthetic data generator** for restaurants, orders, drivers, and road networks
* **OpenAI Gym–compatible** delivery simulation environment
* **PPO-based RL agent** with customizable policy and hyperparameters
* **Baseline heuristics** for performance comparison
* **Automatic checkpointing** (step-based + hourly) and seamless resume
* **TensorBoard integration** for real-time monitoring
* **Comprehensive evaluation** metrics and visualizations 
* **Debugging utilities** for checkpoint and training diagnostics
* **Google Colab integration** for cloud-based training 