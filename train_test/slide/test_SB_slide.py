import gym
import rlbench.gym
from stable_baselines3 import A2C
from stable_baselines3 import PPO

#from stable_baselines3.common.evaluation import evaluate_policy

# Create environment
env = gym.make('slide_block_to_target-state-v0', render_mode='human')

# Load the trained agent
model = PPO.load("slide_block_to_target_20240229_1846", print_system_info=True)

# Evaluate the agent
#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

obs = env.reset()


while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminate, _ = env.step(action)
    print(reward)
    #env.render()  # Note: rendering increases step time.
    if terminate:
        print('Episode Terminated. Resetting...')
        obs = env.reset()
