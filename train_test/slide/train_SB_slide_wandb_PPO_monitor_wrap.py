import gym
#import gymnasium as gym
import rlbench.gym
from stable_baselines3 import PPO
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback
import datetime
import os
from stable_baselines3.common.monitor import Monitor
import shimmy
from gym.wrappers import Monitor as GymMonitor

# Create environment
env = gym.make('slide_block_to_target-state-v0', render_mode=None)
#env = Monitor(env, "./logs", info_keywords=("is_success",))
env = shimmy.GymV21CompatibilityV0(env=env)

#print("Action space after Monitor wrapper:", env.)
print(type(env.action_space))
env.reset()
print(env.step(env.action_space.sample()))

task_code_path = "/home/teodor/Exjobb/Sim/RLBench/rlbench/tasks/slide_block_to_target.py"

# Check if the file exists
if not os.path.exists(task_code_path):
    raise FileNotFoundError(f"The reward function file '{task_code_path}' does not exist.")

# Read the contents of the reward function file
with open(task_code_path, "r") as rf:
    task_code = rf.read()

task_code_artifact = wandb.Artifact("task_code", type="code")

# Add the reward function file to the artifact
task_code_artifact.add_file(task_code_path)


config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 2000000,
    "env_id": env,
    "n_steps": 180
}
run = wandb.init(
    project="slide_block_to_target_PPO",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    #save_code=True,  # optional
)

# Log the artifact to the run
run.log_artifact(task_code_artifact)
print(type(env.action_space))
model = PPO(config["policy_type"], config["env_id"], verbose=1, tensorboard_log=f"runs/{run.id}", n_steps=config["n_steps"]) #n_steps=180)
print(type(env.action_space))
# Create Wandb callback instance

model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
    model_save_path=f"models/{run.id}",
    verbose=2,
    ),
)

current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M")
filename = f"slide_block_to_target_custom_callback{current_datetime}"
model.save(filename)

run.finish()


