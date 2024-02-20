import numpy as np
import gym

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget

class ImitationLearning(object):
    def __init__(self, env):
        self.env = env
        self.model = self._build_model()

    def _build_model(self):
        # Define a neural network architecture (MlpPolicy in this case)
        return PPO("MlpPolicy", self.env, verbose=1)

    def train(self, demos):
        # Convert demonstrations to the format expected by Stable Baselines
        obs = [demo.get_low_dim_data() for demo in demos]
        actions = [demo.get_action('joint_velocities') for demo in demos]

        # Train the policy network using PPO
        self.model.learn(total_timesteps=10000, log_interval=10, reset_num_timesteps=False, tb_log_name="ppo2_reach_target", callback=None)

    def predict_action(self, obs):
        return self.model.predict(obs)

    def behaviour_cloning_loss(self, ground_truth_actions, predicted_actions):
        return np.mean((ground_truth_actions - predicted_actions) ** 2)

live_demos = True
DATASET = '' if live_demos else 'PATH/TO/YOUR/DATASET'

obs_config = ObservationConfig()
obs_config.set_all(True)

env = Environment(
    action_mode=MoveArmThenGripper(
        arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()),
    obs_config=ObservationConfig(),
    headless=False)
env = gym.wrappers.TimeLimit(env, max_episode_steps=100)  # Limit the number of steps per episode
env = DummyVecEnv([lambda: env])

task = env.get_task(ReachTarget)

il = ImitationLearning(env)

demos = task.get_demos(2, live_demos=live_demos)  # -> List[List[Observation]]
demos = np.array(demos).flatten()

il.train(demos)

print('Done')
env.close()
