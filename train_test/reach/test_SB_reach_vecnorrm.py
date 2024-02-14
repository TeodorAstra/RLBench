import gym
import rlbench.gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
#from stable_baselines3.common.evaluation import evaluate_policy

# Create environment
env = gym.make('reach_target-state-v0', render_mode="human")

#env = DummyVecEnv([lambda: env])  # Wrap the environment in a DummyVecEnv
# Tip from SB3: Normalize observations and rewards
#env = VecNormalize(env, training=False, norm_obs=True, norm_reward=True)

# Load the trained agent
model = A2C.load("reach_target_A2C_VecNorm", print_system_info=True)

# Evaluate the agent
#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

obs = env.reset()

training_steps = 1000
episode_length = 100
for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        #obs = env.reset()

    action, _states = model.predict(obs, deterministic=True)
    #obs, reward, terminate, _ = env.step(env.action_space.sample())s
    #obs, reward, terminate, _ = env.step(action)
    obs, reward, terminate, _ = env.step(env.action_space.sample())
    #env.render()  # Note: rendering increases step time.
