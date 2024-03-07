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
    "n_steps": 90
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

model = PPO(config["policy_type"], config["env_id"], verbose=1, tensorboard_log=f"runs/{run.id}", n_steps=config["n_steps"]) #n_steps=180)

def is_task_successful(info):
# Check if the episode terminated before a certain number of steps
    return info['steps'] < 30

"""
class CustomWandbCallback(WandbCallback):
    def __init__(self, *args, **kwargs):
        super(CustomWandbCallback, self).__init__(*args, **kwargs)

    def _on_step(self) -> bool:
        # Call parent method
        result = super()._on_step()

        # Access local variables using self.locals
        total_successes = sum(info.get('is_successful', False) for info in self.locals.get('info', []))
        success_rate = total_successes / (config["n_steps"]/30) # 30 is the max episode length of the task
        wandb.log({"Success Rate": success_rate})

        return result
"""

class CustomWandbCallback(WandbCallback):
    def __init__(self, n_steps=100, *args, **kwargs):
        super(CustomWandbCallback, self).__init__(*args, **kwargs)
        self.n_steps = config["n_steps"]
        self.successful_steps = []
    
    def _on_step(self) -> bool:
        # Call parent method
        result = super()._on_step()

        # Access local variables using self.locals
        info = self.locals.get('info', {})
        is_successful = info.get('is_successful', False)
        
        # Add the success/failure of the current step to the list
        self.successful_steps.append(is_successful)
        
        # Keep only the last n_steps in the list
        if len(self.successful_steps) > self.n_steps:
            self.successful_steps = self.successful_steps[-self.n_steps:]

        # Calculate success rate for the last n_steps
        success_rate = sum(self.successful_steps) / len(self.successful_steps) if self.successful_steps else 0
        wandb.log({"Success Rate (Last {} Steps)".format(self.n_steps): success_rate})

        return result

# Create Wandb callback instance

model.learn(
    total_timesteps=config["total_timesteps"],
    callback=CustomWandbCallback(
    model_save_path=f"models/{run.id}",
    verbose=2,
    ),
)

current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M")
filename = f"slide_block_to_target_custom_callback{current_datetime}"
model.save(filename)

run.finish()


