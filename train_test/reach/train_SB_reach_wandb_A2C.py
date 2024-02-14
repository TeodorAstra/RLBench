import gym
import rlbench.gym

from stable_baselines3 import A2C
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import DummyVecEnv


# Create environment
env = gym.make('reach_target-state-v0', render_mode=None)

env = DummyVecEnv([lambda: env])  # Wrap the environment in a DummyVecEnv

# Tip from SB3: Normalize observations and rewards
env = VecNormalize(env, norm_obs=True, norm_reward=True)

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 100000,
    "env_id": env,
}
run = wandb.init(
    project="reach_target_A2C",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    # monitor_gym=True,  # auto-upload the videos of agents playing the game
    # save_code=True,  # optional
)

model = A2C(config["policy_type"], config["env_id"], verbose=1, tensorboard_log=f"runs/{run.id}")
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
    
)

model.save("reach_target_A2C_VecNorm")

wandb.finish()
# Load the trained agent
#model = A2C.load("A2C_grilling")

# Evaluate the agent
#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

