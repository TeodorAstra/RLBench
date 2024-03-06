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
        total_successes = sum(info.get('is_success') for info in self.locals.get('infos', []))
        success_rate = total_successes / len(self.locals.get('infos', []))
        wandb.log({"Success Rate": success_rate})

        return result
"""

class CustomWandbCallback(WandbCallback):
    def __init__(self, *args, **kwargs):
        super(CustomWandbCallback, self).__init__(*args, **kwargs)
        self.n_steps = config["n_steps"]
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Call parent method
        result = super()._on_step()

        # Access local variables using self.locals
        episode_lengths = self.locals.get('episode_lengths', [])
        self.episode_lengths.extend(episode_lengths)

        # Calculate the percentage of episodes shorter than 30 steps
        short_episodes = sum(1 for length in episode_lengths if length < 30)
        total_episodes = len(self.episode_lengths)
        percentage_short_episodes = (short_episodes / total_episodes) * 100

        # Log the percentage of short episodes
        wandb.log({"Percentage of Episodes Shorter Than 30 Steps": percentage_short_episodes})

        # Reset episode lengths every n_steps
        if self.num_timesteps % self.n_steps == 0:
            self.episode_lengths = []

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
filename = f"slide_block_to_target_{current_datetime}"
model.save(filename)

run.finish()


