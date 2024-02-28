import rlbench.gym
import gym
from stable_baselines3 import A2C
from stable_baselines3 import PPO

#from stable_baselines3.common.evaluation import evaluate_policy

# Create environment
env = gym.make('reach_target-state-v0', render_mode="human")

# Load the trained agent
model = PPO.load("reach_target_PPO_shaping_n_steps_150_ent_coef_0.01_clip_range_0.8_45k", print_system_info=True)



training_steps = 120
episode_length = 15
for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        obs = env.reset()
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminate, _ = env.step(action)
    print(reward)
    #env.render()  # Note: rendering increases step time.
    if terminate:
        print('Episode Terminated. Resetting...')
        obs = env.reset()

print('Done')
env.close()