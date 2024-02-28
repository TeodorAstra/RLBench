import gym
import rlbench.gym

from stable_baselines3 import PPO
import wandb
from wandb.integration.sb3 import WandbCallback


# Create environment
env = gym.make('teodor_reach_target_two-state-v0', render_mode=None)
#env._max_episode_steps = 1000



config = {
    "policy_type": "MlpPolicy",
    #"total_timesteps": 22500,
    "total_timesteps": 90000,
    #"total_timesteps": 360000,
    "env_id": env,
}
run = wandb.init(
    project="reach_target_PPO",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    # save_code=True,  # optional
)

model = PPO(config["policy_type"], config["env_id"], verbose=1, tensorboard_log=f"runs/{run.id}", n_steps = 150, ent_coef=0.001)#, clip_range=0.4)
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
    
)

model.save("teodor_reach_big_ball_less_variations_90k")


#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

