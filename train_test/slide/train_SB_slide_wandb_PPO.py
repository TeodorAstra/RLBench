import gym
import rlbench.gym
from stable_baselines3 import PPO
import wandb
from wandb.integration.sb3 import WandbCallback
import datetime
import os


# Create environment
env = gym.make('slide_block_to_target-state-v0', render_mode=None)

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
    "total_timesteps": 900,
    "env_id": env,
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

model = PPO(config["policy_type"], config["env_id"], verbose=1, tensorboard_log=f"runs/{run.id}", n_steps = 90) #n_steps=180)
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
)

current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M")
filename = f"slide_block_to_target_{current_datetime}"
model.save(filename)

run.finish()

# Load the trained agent
#model = A2C.load("A2C_grilling")

# Evaluate the agent
#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

