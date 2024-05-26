from typing import List, Tuple
from rlbench.backend.task import Task
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.dummy import Dummy
from rlbench.backend.conditions import DetectedCondition, GraspedCondition
from pyrep.objects.force_sensor import ForceSensor
from rlbench.backend.spawn_boundary import SpawnBoundary

import random

import numpy as np


class TeodorExtractScrambledV3(Task):

    def init_task(self) -> None:
        # TODO: This is called once when a task is initialised.
        self.cube1 = Shape('cube1')
        self.distractor1 = Shape('distractor1')
        self.distractor2 = Shape('distractor2')
        self.distractor3 = Shape('distractor3')
        self.distractor4 = Shape('distractor4')
        self.distractor5 = Shape('distractor5')
        self.spawn_boundary = Shape('spawn_boundary')
        
       

        self.register_graspable_objects([self.cube1])
      
        #self.zone = Shape('zone')

        
        self.in_zone_sensor = ProximitySensor('in_zone_sensor')
        

        
        

        zone_is_empty_condition = ([DetectedCondition(self.cube1, self.in_zone_sensor, negated=True), 
                                    GraspedCondition(self.robot.gripper, self.cube1)])
        self.register_success_conditions(zone_is_empty_condition)


    def init_episode(self, index: int) -> List[str]:

        self.outside_zone = {
            'cube1': False,
            'distractor1' : False,
            'distractor2' : False,
            'distractor3' : False,
            'distractor4' : False,
            'distractor5' : False
        }

        rand= random.randint(1, 3)

        if(rand == 1):
            pos =[0.25, 0.025, 0.777]

        elif(rand == 2):
            pos =[0.22, -0.1, 0.777]

        elif(rand == 3):
            pos =[0.25, 0.1, 0.777]


        self.cube1.set_position(pos, None, reset_dynamics=True)

        
        print("Start new ep")
        
        b = SpawnBoundary([self.spawn_boundary])
        b.sample(self.distractor1, min_distance=0.05)
        b.sample(self.distractor2, min_distance=0.05)
        b.sample(self.distractor3, min_distance=0.05)
        b.sample(self.distractor4, min_distance=0.05)
        b.sample(self.distractor5, min_distance=0.05)

        
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
    """
        return np.concatenate([
            self.cube1.get_position(), self.in_zone_sensor.get_position(),
            self.distractor1.get_position(),
            self.distractor2.get_position(),
            self.distractor3.get_position(),
            self.distractor4.get_position(),
            self.distractor5.get_position(),
            #self.target.get_position()
            ])
    """        
            #self.tip_1.get_position(), self.tip_2.get_position()])
            
            

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

  
        gripper_to_cube = self.cube_distance_reward()

        #tips_to_zones = self.grip_zone_distance_reward() 
        """
        distractor1 = self.exit_punishment(self.distractor1)
        distractor2 = self.exit_punishment(self.distractor2)
        distractor3 = self.exit_punishment(self.distractor3)
        distractor4 = self.exit_punishment(self.distractor4)
        distractor5 = self.exit_punishment(self.distractor5)

        distractor_punishment = (distractor1 +
                                 distractor2 +
                                 distractor3 +
                                 distractor4 +
                                 distractor5)
        """
       
        #cube_distance_final_pos = self.block_final_pos()

        #distance_grasped = self.cube_distance_from_final_pos_while_grasped_reward()
        distance_grasped = self.cube_distance_from_center_while_grasped_reward()

        #grasped_reward = self.grasped_reward()
 
        task_complete_reward = self.task_complete_reward()



        #reward for completed task
        total_reward = (gripper_to_cube +
                        distance_grasped +
                        #distractor_punishment +
                        task_complete_reward)
             

        #print(total_reward)
        #print(self.outside_zone)

        return total_reward
    
    
        
   

 
    
   
    
        
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
            print("GRASPED")
            reward = 20 + 100*np.linalg.norm(self.cube1.get_position() - self.in_zone_sensor.get_position())
            return reward
        else:
            return 0

    def cube_distance_from_final_pos_while_grasped_reward(self)->float:
        if GraspedCondition(self.robot.gripper, self.cube1).condition_met()[0]:
            dist = 100*np.linalg.norm(self.cube1.get_position() - self.final_pos.get_position())
            reward = 100/(dist + 1)
            return reward
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
        
    def exit_punishment(self, cube_id)->float:
            cube_name = cube_id.get_name()
            if not self.outside_zone[cube_name]:
                if(DetectedCondition(cube_id,self.in_zone_sensor, negated=True).condition_met()[0]): #Negated meaning its no longer in the sensor
                    self.outside_zone[cube_name] = True
                    return -50
                else:
                    return 0
            else:
                return 0  
    
    def task_complete_reward(self)->float:
        if (DetectedCondition(self.cube1, self.in_zone_sensor, negated = True).condition_met()[0] and
            GraspedCondition(self.robot.gripper, self.cube1).condition_met()[0]):
            return 10000
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