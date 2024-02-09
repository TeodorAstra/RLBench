import gym
import rlbench.gym

from stable_baselines3 import A2C



# Create environment
env = gym.make('teodor_reach_target-state-v0', render_mode=None)

# Instantiate the agent
model = A2C("MlpPolicy", env, verbose=1)
# Train the agent
model.learn(total_timesteps=200000)
# Save the agent
model.save("teodor_reach_target_A2C")

# Load the trained agent
#model = A2C.load("A2C_grilling")

# Evaluate the agent
#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
