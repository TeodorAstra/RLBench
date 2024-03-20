from typing import List, Tuple
from rlbench.backend.task import Task
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.conditions import DetectedCondition

import numpy as np


class TeodorRemoveFromZone(Task):

    def init_task(self) -> None:
        # TODO: This is called once when a task is initialised.
        self.cube1 = Shape('cube1')
        self.cube2 = Shape('cube2')
        self.cube3 = Shape('cube3')

        self.spawn_boundary = Shape('spawn_boundary')
        self.in_zone_sensor = ProximitySensor('in_zone_sensor')

        zone_is_empty_condition = [DetectedCondition(self.cube1, self.in_zone_sensor, negated=True)]
        self.register_success_conditions(zone_is_empty_condition)


    def init_episode(self, index: int) -> List[str]:
        
        print("Start new ep")
        
        b = SpawnBoundary([self.spawn_boundary])

        b.sample(self.cube1, min_distance=0.2)
        b.sample(self.cube2, min_distance=0.2)
        b.sample(self.cube3, min_distance=0.2)
        
        return ['']

    def variation_count(self) -> int:
        # TODO: The number of variations for this task.
        return 1
    
    def is_static_workspace(self) -> bool:
        return True

    def get_low_dim_state(self) -> np.ndarray:
        # One of the few tasks that have a custom low_dim_state function.
        return np.concatenate([
            self.cube1.get_position(), self.cube2.get_position(), self.cube3.get_position()])

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    

    def zone_is_empty(self) -> bool:
        if (DetectedCondition(self.cube1, self.in_zone_sensor)
            and DetectedCondition(self.cube1, self.in_zone_sensor)
            and DetectedCondition(self.cube1, self.in_zone_sensor)):

            return False
        else:
            return False
        
    def reward(self) -> float:
        return 1
