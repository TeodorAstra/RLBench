import rlbench.gym
import gym
from stable_baselines3 import A2C
from stable_baselines3 import PPO

#from stable_baselines3.common.evaluation import evaluate_policy

# Create environment
env = gym.make('teodor_reach_target-state-v0', render_mode="human")

# Load the trained agent
model = PPO.load("teodor_reach_big_ball_1xShaping&SuccessReward__StandardValues_360k", print_system_info=True)


obs = env.reset()

while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminate, _ = env.step(action)
    print(reward)
    #env.render()  # Note: rendering increases step time.
    if terminate:
        print('Episode Terminated. Resetting...')
        obs = env.reset()



#Defining training eps etc is not necesaary since its done int the gym

"""
training_steps = 150
episode_length = 40
for i in range(training_steps):
    if i % episode_length == 0:
        #print('Reset Episode')
        obs = env.reset()
    action, _states = model.predict(obs)
    obs, reward, terminate, _ = env.step(action)
    print(reward)
    #env.render()  # Note: rendering increases step time.
    if terminate:
        print('Episode Terminated. Resetting...')
        obs = env.reset()
"""

print('Done')
env.close()