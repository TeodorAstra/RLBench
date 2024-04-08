import gym
import rlbench.gym
from stable_baselines3 import DDPG
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback
import datetime
import os
from stable_baselines3.common.monitor import Monitor
from custom_wandb_callback import CustomWandbCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import numpy as np


# Create environment
env = gym.make('teodor_remove_from_zone-state-v0', render_mode=None)


task_code_path = "/home/teodor/Exjobb/Sim/RLBench/rlbench/tasks/teodor_remove_from_zone_v2.py"

# Check if the file exists
if not os.path.exists(task_code_path):
    raise FileNotFoundError(f"The reward function file '{task_code_path}' does not exist.")

# Read the contents of the reward function file
with open(task_code_path, "r") as rf:
    task_code = rf.read()

task_code_artifact = wandb.Artifact("task_code", type="code")

# Add the reward function file to the artifact
task_code_artifact.add_file(task_code_path)

n_actions = env.action_space.shape[-1]

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 1000000,
    "env_id": env,
    "n_steps": 200,
    "ent_coef": 0.01,
    "action_noise" : NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
}


run = wandb.init(
    project="teodor_remove_from_zone_v2_DDPG",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #monitor_gym=True,  # auto-upload the videos of agents playing the game
     #save_code=True,  # optional
)

# Log the artifact to the run
run.log_artifact(task_code_artifact)

#model = PPO(config["policy_type"], config["env_id"], verbose=1, n_steps=config["n_steps"], ent_coef=config["ent_coef"]) 
model = DDPG(config["policy_type"], config["env_id"], action_noise=config["action_noise"], verbose=1, tensorboard_log=f"runs/{run.id}")

# Create Wandb callback instance
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=CustomWandbCallback(
    #model_save_path=f"models/{run.id}",
    #verbose=2,
    ),
)


current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M")
filename = f"remove_from_zone_DDPG_v2{current_datetime}"
model.save(filename)

run.finish()

