import gym
import rlbench.gym
from stable_baselines3 import A2C
from stable_baselines3 import PPO

#from stable_baselines3.common.evaluation import evaluate_policy

# Create environment
env = gym.make('reach_target-state-v0', render_mode=None)

# Load the trained agent
model = PPO.load("reach_target_PPO_Rollback_square_reward", print_system_info=True)

# Evaluate the agent
#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

obs = env.reset()

"""
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(rewards)


"""    


training_steps = 45
episode_length = 15
for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        #obs = env.reset()

    #action, _states = model.predict(obs, deterministic=True) #Deterministic for A2C
    action, _states = model.predict(obs)
    #obs, reward, terminate, _ = env.step(action)
    obs, reward, terminate, _ = env.step(env.action_space.sample())

    print(reward)
    #env.render()  # Note: rendering increases step time.
