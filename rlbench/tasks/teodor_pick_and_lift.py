from typing import List, Tuple
from rlbench.backend.task import Task
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.conditions import DetectedCondition, GraspedCondition

import numpy as np


class TeodorPickAndLift(Task):

    def init_task(self) -> None:
        # TODO: This is called once when a task is initialised.
        self.cube1 = Shape('cube1')
        #self.cube2 = Shape('cube2')
   
    

        self.register_graspable_objects([self.cube1])
      
        self.zone = Shape('zone')

        self.cylinder_sensor = ProximitySensor('cylinder')


        success = ([DetectedCondition(self.cube1, self.cylinder_sensor)])
        
        
        self.register_success_conditions(success)


    def init_episode(self, index: int) -> List[str]:


        self.has_been_grasped = {
            'cube1': False,

        }

        print("Start new ep")
        
        return ['']

    def variation_count(self) -> int:
        # TODO: The number of variations for this task.
        return 1
    
    def is_static_workspace(self) -> bool:
        return True

    def get_low_dim_state(self) -> np.ndarray:
        # One of the few tasks that have a custom low_dim_state function.
        return np.concatenate([
            self.cube1.get_position(), self.cylinder_sensor.get_position()])

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    
    """
    def zone_is_empty(self) -> bool:
        if (DetectedCondition(self.cube1, self.in_zone_sensor)
            and DetectedCondition(self.cube1, self.in_zone_sensor)
            and DetectedCondition(self.cube1, self.in_zone_sensor)):

            return False
        else:
            return False
    """        
    def reward(self) -> float:

            return self.cube_reward(self.cube1)

        
    
    def cube_reward(self, cube_id)->float:
        gripper_to_cube_positive = self.cube_distance_reward_postive(cube_id)
        cube_distance_from_cylinder = self.cube_distance_cylinder(cube_id)
        #exit_reward = self.exit_reward(cube_id)
        grasped_reward = self.grasped_reward()

        

        task_complete_reward = self.task_complete_reward()

        total_reward = (gripper_to_cube_positive + 
                        grasped_reward + 
                        cube_distance_from_cylinder +
                        #exit_reward +
                        task_complete_reward)
        
        return total_reward



    def zone_distance_reward(self)->float:
        if DetectedCondition(self.robot.arm.get_tip(), self.in_zone_sensor).condition_met()[0]:
            return  0 #No negative shaping if in zone
        else:
            return -np.linalg.norm(
                self.zone.get_position() - self.robot.arm.get_tip().get_position())
        
    def cube_distance_reward(self)->float:
       return -np.linalg.norm(
                self.cube1.get_position() - self.robot.arm.get_tip().get_position())
    
    def cube_distance_reward_postive(self, cube_id)->float:
       distance = np.linalg.norm(
                cube_id.get_position() - self.robot.arm.get_tip().get_position())
       reward = 10/(100*distance + 1)

       return reward
    
    def cube_distance_cylinder(self, cube_id)->float:
        if GraspedCondition(self.robot.gripper, cube_id).condition_met()[0]:
            distance = np.linalg.norm(
                        cube_id.get_position() - self.cylinder_sensor.get_position())
            reward = 100/(100*distance + 1)
        else:
           reward = 0
    
        return reward
    
    
    def cube_distance_from_center_while_grasped_reward(self, cube_id)->float:
        if GraspedCondition(self.robot.gripper, cube_id).condition_met()[0]:
            cube_name = cube_id.get_name()
            self.has_been_grasped[cube_name] = True
            #print("CUBE GRASPED")
            return 100*np.linalg.norm(
                    cube_id.get_position() - self.zone.get_position())
        else:
            return 0
   
    
    def exit_reward(self, cube_id)-> float:
        cube_name = cube_id.get_name()
        if self.inside_zone[cube_name]:
            if(DetectedCondition(cube_id,self.in_zone_sensor, negated=True).condition_met()[0]): #Negated meaning its no longer in the sensor
                self.inside_zone[cube_name] = False
                return 500
            else:
                return 0
        else:
            return 0
    
    def task_complete_reward(self)->float:
        if (DetectedCondition(self.cube1, self.cylinder_sensor).condition_met()[0]):
           # GraspedCondition(self.robot.gripper, self.cube1).condition_met()[0]):
            return 1000
        else:
            return 0
        
    def close_tip_reward(self):
        print(self.robot.gripper.get_open_amount())

    def grasped_reward(self)->float:
        if GraspedCondition(self.robot.gripper, self.cube1).condition_met()[0]:
            print("CUBE GRASPED")
            return 10
        else:
            return 0
        
    def cube_is_outside(self, cube_id)->float:
        cube_name = cube_id.get_name()
        if(self.has_been_grasped[cube_name] and 
           DetectedCondition(cube_id,self.in_zone_sensor, negated=True).condition_met()[0]):
            return 50
        else:
            return 0

    def grasped(self)->float:
        if GraspedCondition(self.robot.gripper, self.cube1).condition_met()[0]:
            return 10
        else:
            return 0