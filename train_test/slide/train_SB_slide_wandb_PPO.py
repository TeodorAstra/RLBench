import gym
import rlbench.gym

from stable_baselines3 import PPO
import wandb
from wandb.integration.sb3 import WandbCallback


# Create environment
env = gym.make('slide_block_to_target-state-v0', render_mode=None)


config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 1000000,
    "env_id": env,
}
run = wandb.init(
    project="slide_block_to_target_PPO",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    # save_code=True,  # optional
)

model = PPO(config["policy_type"], config["env_id"], verbose=1, tensorboard_log=f"runs/{run.id}")
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
    
)

model.save("slide_block_to_target_PPO_shape_1m")

# Load the trained agent
#model = A2C.load("A2C_grilling")

# Evaluate the agent
#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

