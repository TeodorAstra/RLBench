import gym
import rlbench.gym
from stable_baselines3 import A2C

# Create environment
env = gym.make('teodor_reach_target-state-v0', render_mode='human')

# Load the trained agent
model = A2C.load("teodor_reach_target")

# Evaluate the agent
#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

training_steps = 120
episode_length = 40
for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        obs = env.reset()

    action, _states = model.predict(obs)
    #obs, reward, terminate, _ = env.step(env.action_space.sample())
    obs, reward, terminate, _ = env.step(action)
    #obs, reward, terminate, _ = env.step(action)
    print(obs)
    print(reward)
    print(terminate)
