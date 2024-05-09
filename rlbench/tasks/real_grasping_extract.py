from typing import List, Tuple
from rlbench.backend.task import Task
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.dummy import Dummy
from rlbench.backend.conditions import DetectedCondition, GraspedCondition
from pyrep.objects.force_sensor import ForceSensor

import numpy as np


class RealGraspingExtract(Task):

    def init_task(self) -> None:
        # TODO: This is called once when a task is initialised.
        self.cube1 = Shape('cube1')
   
        self.center_tip = Dummy('Panda_tip')

        self.register_graspable_objects([self.cube1])
      
        #self.zone = Shape('zone')

        
        self.in_zone_sensor = ProximitySensor('in_zone_sensor')
        self.target_1 = ProximitySensor('target_1')
        self.target_2 = ProximitySensor('target_2')
        #self.final_pos = ProximitySensor('final_pos')

        #self.tip_1 = ForceSensor('Panda_gripper_touchSensor0')
        #self.tip_2 = ForceSensor('Panda_gripper_touchSensor1')

        #self.tip_1 = Shape('tip_placement_1')
        #self.tip_2 = Shape('tip_placement_2')

        self.tip_1 = Shape('Panda_rightfinger_force_contact')
        self.tip_2 = Shape('Panda_leftfinger_force_contact')
        

        zone_is_empty_condition = ([DetectedCondition(self.cube1, self.in_zone_sensor, negated=True), 
                                    GraspedCondition(self.robot.gripper, self.cube1)])
        self.register_success_conditions(zone_is_empty_condition)


    def init_episode(self, index: int) -> List[str]:

        
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
            self.cube1.get_position(), self.in_zone_sensor.get_position(), 
            self.target_1.get_position(), self.target_2.get_position()])
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

        tips_in_zone = self.tips_in_zone_reward()
       
        #cube_distance_final_pos = self.block_final_pos()

        #distance_grasped = self.cube_distance_from_final_pos_while_grasped_reward()
        distance_grasped = self.cube_distance_from_center_while_grasped_reward()

        #grasped_reward = self.grasped_reward()
 
        task_complete_reward = self.task_complete_reward()



        #reward for completed task
        total_reward = (#tips_to_zones +
                        gripper_to_cube +
                        tips_in_zone +
                        #cube_distance_final_pos +
                        distance_grasped +
                        #grasped_reward +
                        task_complete_reward)
             

        #print(total_reward)
        #print(self.outside_zone)

        return total_reward
    
    def grip_zone_distance_reward(self)->float:
            target_1_dist = -np.linalg.norm(self.target_1.get_position() - self.tip_1.get_position())
            target_2_dist = -np.linalg.norm(self.target_2.get_position() - self.tip_2.get_position())

            total = target_1_dist + target_2_dist
            return total
        
    def tips_in_zone_reward_old(self)->float:
        reward = 0
        if DetectedCondition(self.tip_1, self.target_1).condition_met()[0]:
            reward = reward + 10
            print("Tip 1 in zone")
        if DetectedCondition(self.tip_2, self.target_2).condition_met()[0]:
            reward = reward + 10
            print("Tip 2 in zone")
        return reward

    def tips_in_zone_reward(self)->float:
        if ((DetectedCondition(self.tip_1, self.target_1).condition_met()[0] and 
            DetectedCondition(self.tip_2, self.target_2).condition_met()[0]) or
            (DetectedCondition(self.tip_1, self.target_2).condition_met()[0] and 
            DetectedCondition(self.tip_2, self.target_1).condition_met()[0])):
            reward = 20
            print("Good grasp pos")
            return reward
        else:
            return 0
    
    def block_final_pos(self)->float:
            dist = -np.linalg.norm(self.cube1.get_position() - self.final_pos.get_position())
            total = 10*dist

            return total

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
            print("GRASPED")
            return 100*np.linalg.norm(
                    self.cube1.get_position() - self.in_zone_sensor.get_position())
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
        if (DetectedCondition(self.cube1, self.in_zone_sensor, negated = True).condition_met()[0] and
            GraspedCondition(self.robot.gripper, self.cube1).condition_met()[0]):
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