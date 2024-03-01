from typing import List

import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition


class SlideBlockToTarget(Task):

    def init_task(self) -> None:
        self._block = Shape('block')
        self._target = ProximitySensor('success')
        self.register_success_conditions([
            DetectedCondition(self._block, self._target)])
        self.subgoal_achieved = False
        self.old_block_to_target = 0

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
            self._block.get_position(), self._target.get_position()])

    def reward(self) -> float:
        grip_to_block = -np.linalg.norm(
            self._block.get_position() - self.robot.arm.get_tip().get_position())
        block_to_target = np.linalg.norm(
            self._block.get_position() - self._target.get_position())
        
        block_velocity = np.linalg.norm(self._block.get_velocity()[0]) #Get linear velocity

        new_block_to_target = block_to_target

        closer_to_target_reward = 0
        if (new_block_to_target < self.old_block_to_target):
            closer_to_target_reward = 200
        
        self.old_block_to_target = new_block_to_target        


        block_velocity_reward = 0
        if block_velocity > 0:
            block_velocity_reward = 100

        #Introduce Sub-goal Rewards
        subgoal_reward = 0
        CLOSE_PROXIMITY = 0.15
        if not self.subgoal_achieved and (np.linalg.norm(self._block.get_position() - self.robot.arm.get_tip().get_position()) < CLOSE_PROXIMITY):
            subgoal_reward = 500  # Reward for achieving the subgoal
            self.subgoal_achieved = True  # Mark subgoal as achieved for this episode

        if self.success()[0]:
            return 1000

        """
        if DetectedCondition(self._block, self._target).condition_met()[0]:
            return 1000 # For successfull task
        """
        total_reward = (
            grip_to_block +
            -1*block_to_target +
            closer_to_target_reward +
            block_velocity_reward 
            #subgoal_reward
        )
        
        return total_reward
