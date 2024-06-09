import gym
import rlbench.gym
from stable_baselines3 import A2C
from stable_baselines3 import PPO

#from stable_baselines3.common.evaluation import evaluate_policy

# Create environment
env = gym.make('teodor_grasp_extract_random_pos-state-v0', render_mode="human")

# Load the trained agent
model = PPO.load("random_pos_PPO_20240525_0121", print_system_info=True)

# Evaluate the agent
#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)



obs = env.reset()

while True:

    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminate, _ = env.step(action)


    print(reward)

    if terminate:
        print('Episode Terminated. Resetting...')
        obs = env.reset()
