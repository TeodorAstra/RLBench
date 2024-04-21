from typing import List, Tuple
from rlbench.backend.task import Task
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.conditions import DetectedCondition, GraspedCondition

import numpy as np


class TeodorExtractWithDistractors(Task):

    def init_task(self) -> None:
        # TODO: This is called once when a task is initialised.
        self.cube1 = Shape('cube1')
   
    

        self.register_graspable_objects([self.cube1])
      
        self.zone = Shape('zone')

        
        self.in_zone_sensor = ProximitySensor('in_zone_sensor')

        zone_is_empty_condition = ([DetectedCondition(self.cube1, self.in_zone_sensor, negated=True), GraspedCondition(self.robot.gripper, self.cube1)])
        self.register_success_conditions(zone_is_empty_condition)


    def init_episode(self, index: int) -> List[str]:

        self.outside_zone = {
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
            self.cube1.get_position(), self.in_zone_sensor.get_position()])

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

        #self.close_tip_reward()

        #negative reward for grippers disntance to zone
        #gripper_to_zone = self.zone_distance_reward()

        #gripper_to_cube = self.cube_distance_reward()

        gripper_to_cube_positive = self.cube_distance_reward_postive()

        cube_distance_from_center_grasped = self.cube_distance_from_center_while_grasped_reward()
        
        #reward for end effector movement in zone. This to encourage interaction with the cubes
        #gripper_movement_in_zone = self.gripper_movement_in_zone()
        
        grasped_in_zone_reward = self.grasped_in_zone_reward()

        grasped_reward = self.grasped_reward()
        #rewards movement of cube 
        #v_r_1 = self.movement_reward(self.cube1)
        #v_r_2 = self.movement_reward(self.cube2)

        #velocity_reward = v_r_1#+v_r_2

        #rewards for cube exiting zone
        e_r_1 = self.exit_reward(self.cube1)
        #e_r_2 = self.exit_reward(self.cube2)

        exit_reward = e_r_1#+e_r_2

        task_complete_reward = self.task_complete_reward()

        #reward for completed task
        total_reward = (gripper_to_cube_positive + 
                        #grasped_reward + 
                        cube_distance_from_center_grasped +
                        task_complete_reward)
             

        #print(total_reward)
        #print(self.outside_zone)

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
    
    def cube_distance_reward_postive(self)->float:
       distance = np.linalg.norm(
                self.cube1.get_position() - self.robot.arm.get_tip().get_position())
       reward = 10/(100*distance + 1)

       return reward
    
    
    def cube_distance_from_center_while_grasped_reward(self)->float:
        if GraspedCondition(self.robot.gripper, self.cube1).condition_met()[0]:
            return 100*np.linalg.norm(
                    self.cube1.get_position() - self.zone.get_position())
        else:
            return 0
    
        
    def gripper_movement_in_zone(self)->float:
        if DetectedCondition(self.robot.arm.get_tip(), self.in_zone_sensor).condition_met()[0]:
            gripper_velocity = np.linalg.norm(self.robot.arm.get_tip().get_velocity()[0])
            if gripper_velocity > 0:
                return 0.1 
            else:
                return 0
        else:
            return 0
        
    def movement_reward(self, cube_id)-> float:
        block_velocity = np.linalg.norm(cube_id.get_velocity()[0]) #Get linear velocity
        cube_name = cube_id.get_name()
        #print(cube_name)

        block_velocity_reward = 0
        if (block_velocity > 0 and 
            not self.outside_zone[cube_name] and 
            DetectedCondition(self.robot.arm.get_tip(), self.in_zone_sensor).condition_met()[0]): #Should not get velocity reward ones the cube has exited the zone
            block_velocity_reward = 1

        return block_velocity_reward
    
    def exit_reward(self, cube_id)-> float:
        cube_name = cube_id.get_name()
        if not self.outside_zone[cube_name]:
            if(DetectedCondition(cube_id,self.in_zone_sensor, negated=True).condition_met()[0]): #Negated meaning its no longer in the sensor
                self.outside_zone[cube_name] = True
                return 100
            else:
                return 0
        else:
            return 0
    
    def task_complete_reward(self)->float:
        if (DetectedCondition(self.cube1, self.in_zone_sensor, negated=True).condition_met()[0] and 
            GraspedCondition(self.robot.gripper, self.cube1).condition_met()[0]): #and not 
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

    def grasped_in_zone_reward(self)->float:
        if DetectedCondition(self.robot.arm.get_tip(), self.in_zone_sensor).condition_met()[0] and GraspedCondition(self.robot.gripper, self.cube1).condition_met()[0]:
            return 500
        else:
            return 0