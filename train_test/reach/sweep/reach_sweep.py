# train.py

import argparse
import gym
import rlbench.gym
from stable_baselines3 import PPO
import wandb
from wandb.integration.sb3 import WandbCallback

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--ent_coef", type=float, help="Entropy coefficient")
parser.add_argument("--clip_range", type=float, help="Clip range")
args = parser.parse_args()

# Create environment
env = gym.make('reach_target-state-v0', render_mode=None)

# Configuration
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 45000,
    "env_id": 'reach_target-state-v0',
    "ent_coef": args.ent_coef,
    "clip_range": args.clip_range
}

# Initialize W&B run
run = wandb.init(
    project="sweep_reach_target_PPO",
    config=config,
    sync_tensorboard=True,
    monitor_gym=True,
)

# Create PPO model
model = PPO(config["policy_type"], config["env_id"], verbose=1, tensorboard_log=f"runs/{run.id}", n_steps=150, ent_coef=config["ent_coef"], clip_range=config["clip_range"])

# Train model
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        model_save_path=f"models/{run.id}",
        verbose=2,
    )
)
