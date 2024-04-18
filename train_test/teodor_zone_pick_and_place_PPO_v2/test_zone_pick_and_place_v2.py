import gym
import rlbench.gym
from stable_baselines3 import A2C
from stable_baselines3 import PPO

#from stable_baselines3.common.evaluation import evaluate_policy

# Create environment
env = gym.make('teodor_zone_pick_and_place_v2-state-v0', render_mode="human")

# Load the trained agent
model = PPO.load("remove_zone_pick_and_place_V2_20240418_0143_GripperJointP", print_system_info=True)

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
