from typing import Callable, List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.const import colors
from rlbench.backend.task import Task
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.conditions import DetectedCondition


class TeodorReachTarget(Task):

    def init_task(self) -> None:
        self.target = Shape('target')
        self.boundaries = Shape('boundary')
        success_sensor = ProximitySensor('success')
        self.register_success_conditions(
            [DetectedCondition(self.robot.arm.get_tip(), success_sensor)])

    def init_episode(self, index: int) -> List[str]:
        b = SpawnBoundary([self.boundaries])
        for ob in [self.target]:
            b.sample(ob, min_distance=0.2,
                     min_rotation=(0, 0, 0), max_rotation=(0, 0, 0))

        return ['reach the %s target']

    def variation_count(self) -> int:
        return len(colors)

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    
    def get_low_dim_state(self) -> np.ndarray:
        # One of the few tasks that have a custom low_dim_state function.
        return np.array(self.target.get_position())
    
        
    def is_static_workspace(self) -> bool:
        return True

    def reward(self) -> float:
        shaping_reward =  -np.linalg.norm(self.target.get_position() -
                               self.robot.arm.get_tip().get_position())

        if DetectedCondition(self.robot.arm.get_tip(), ProximitySensor('success')).condition_met()[0]:
            return 100 #Returns 10 for success

        return shaping_reward
    
        

