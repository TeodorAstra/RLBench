import gym
import rlbench.gym
from stable_baselines3 import PPO
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback
import datetime
import os
from stable_baselines3.common.monitor import Monitor
from custom_wandb_callback import CustomWandbCallback

# Create environment
env = gym.make('teodor_zone_pick_and_place-state-v0', render_mode="human")




config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 100000,
    "env_id": env,
    "n_steps": 200,
    "ent_coef": 0.01
}



# Log the artifact to the run


model = PPO(config["policy_type"], config["env_id"], verbose=1, n_steps=config["n_steps"], ent_coef=config["ent_coef"]) 
model.load("remove_zone_pick_and_place_20240410_1659", print_system_info=True)
#model = PPO(config["policy_type"], config["env_id"], verbose=1, tensorboard_log=f"runs/{run.id}", n_steps=config["n_steps"], ent_coef=config["ent_coef"])

# Create Wandb callback instance
model.learn(
    total_timesteps=config["total_timesteps"],
)






