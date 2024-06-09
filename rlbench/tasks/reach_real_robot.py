from typing import List
from rlbench.backend.task import Task

import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition



class ReachRealRobot(Task):

    def init_task(self) -> None:
        self._sphere = Shape('Cuboid')
        self._target = ProximitySensor('Proximity_sensor')
        self.register_success_conditions([DetectedCondition(self.robot.arm.get_tip(), self._target)])
        
    def init_episode(self, index: int) -> List[str]:
        self._variation_index = index
        return ['slide the block to target',
                'slide the block onto the target',
                'push the block until it is sitting on top of the target',
                'slide the block towards the green target',
                'cover the target with the block by pushing the block in its'
                ' direction']

    def variation_count(self) -> int:
        return 1
    
    def is_static_workspace(self) -> bool:
        return True

    def get_low_dim_state(self) -> np.ndarray:
        # One of the few tasks that have a custom low_dim_state function.
        return np.concatenate([
            self._sphere.get_position()])
    
    """
    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, dict]:
        obs, reward, terminate = self.task.step(action)
        is_success = DetectedCondition(self._block, self._target).condition_met()[0]
        info = {'is_success': is_success}
        return self._extract_obs(obs), reward, terminate, {}

    def step(self) -> None:
        is_success = DetectedCondition(self._block, self._target).condition_met()[0]
        info = {'is_success': is_success}
        return info

   
    #Overriden step function to ad additional info regarding successfull tasks
    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, dict]:
        obs, reward, terminate = self.task.step(action)
        #is_successful = DetectedCondition(self._block, self._target).condition_met()[0]  # Call a method to determine if the task is successful
        is_successful, = self.task.success()
        info = {'is_successful': is_successful}
        return self._extract_obs(obs), reward, terminate, info
    
    """    
    def reward(self) -> float:
        #Distance rewards
        grip_to_target = -0.1*np.linalg.norm(
            self._sphere.get_position() - self.robot.arm.get_tip().get_position())
      
        complete = self.task_complete_reward()

        return grip_to_target + complete
        
    def task_complete_reward(self)->float:
        if (DetectedCondition(self.robot.arm.get_tip(), self._target).condition_met()[0]):
            return 1000
        else:
            return 0

        #Closer to target reward    
        """
        new_block_to_target = block_to_target
        closer_to_target_reward = 0
        if (new_block_to_target < self.old_block_to_target):
            closer_to_target_reward = 200
        self.old_block_to_target = new_block_to_target        
        """
    

        #Sub-goal Rewards
      

        """
        if self.success()[0]:
            return 1000
        """