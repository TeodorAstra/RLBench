import gym
import rlbench.gym

from stable_baselines3 import A2C



# Create environment
env = gym.make('reach_target-state-v0', render_mode=None)

# Instantiate the agent
model = A2C("MlpPolicy", env, verbose=1)
# Train the agent
model.learn(total_timesteps=10000)
# Save the agent
model.save("reach_target")

# Load the trained agent
#model = A2C.load("A2C_grilling")

# Evaluate the agent
#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
