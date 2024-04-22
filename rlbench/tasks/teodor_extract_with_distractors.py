from typing import List, Tuple
from rlbench.backend.task import Task
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.conditions import DetectedCondition, GraspedCondition

import numpy as np


class TeodorExtractWithDistractors(Task):

    def init_task(self) -> None:
        # TODO: This is called once when a task is initialised.
        self.cube1 = Shape('cube1')
        self.distractor1 = Shape('distractor1')
        self.distractor2 = Shape('distractor2')
        self.distractor3 = Shape('distractor3')
        self.distractor4 = Shape('distractor4')
        self.distractor5 = Shape('distractor5')

        self.target = Dummy('Target_pos')

   
    

        self.register_graspable_objects([self.cube1])
      
        self.zone = Shape('zone')

        
        self.in_zone_sensor = ProximitySensor('in_zone_sensor')

        complete_task_condition = ([DetectedCondition(self.cube1, self.in_zone_sensor, negated=True), 
                                    GraspedCondition(self.robot.gripper, self.cube1),
                                    DetectedCondition(self.distractor1, self.in_zone_sensor),
                                    DetectedCondition(self.distractor2, self.in_zone_sensor), 
                                    DetectedCondition(self.distractor3, self.in_zone_sensor), 
                                    DetectedCondition(self.distractor4, self.in_zone_sensor),
                                    DetectedCondition(self.distractor5, self.in_zone_sensor)])
        self.register_success_conditions(complete_task_condition)


    def init_episode(self, index: int) -> List[str]:

        self.outside_zone = {
            'cube1': False,
            'distractor1' : False,
            'distractor2' : False,
            'distractor3' : False,
            'distractor4' : False,
            'distractor5' : False
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
            self.cube1.get_position(), self.in_zone_sensor.get_position(),
            self.distractor1.get_position(),
            self.distractor2.get_position(),
            self.distractor3.get_position(),
            self.distractor4.get_position(),
            self.distractor5.get_position(),
            self.target.get_position()
            ])

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

        gripper_to_cube_positive = self.cube_distance_reward_postive()

        #cube_distance_from_center_grasped = self.cube_distance_from_center_while_grasped_reward()

        task_complete_reward = self.task_complete_reward()

        distracor1 = self.exit_punishment(self.distractor1)
        distracor2 = self.exit_punishment(self.distractor2)
        distracor3 = self.exit_punishment(self.distractor3)
        distracor4 = self.exit_punishment(self.distractor4)
        distracor4 = self.exit_punishment(self.distractor5)

        cube_distance_to_target_grasped = self.cube_distance_to_target_while_grasped_reward()

        distractor_punishment = (distracor1 +
                                 distracor2 +
                                 distracor3 +
                                 distracor4)

        total_reward = (gripper_to_cube_positive + 
                        #cube_distance_from_center_grasped +
                        cube_distance_to_target_grasped +
                        distractor_punishment +
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
    
    def cube_distance_to_target_while_grasped_reward(self)->float:
        if GraspedCondition(self.robot.gripper, self.cube1).condition_met()[0]:
            distance = np.linalg.norm(
            self.cube1.get_position() - self.target.get_position())
            reward = 100/(100*distance + 1)
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
        
    def exit_punishment(self, cube_id)->float:
            cube_name = cube_id.get_name()
            if not self.outside_zone[cube_name]:
                if(DetectedCondition(cube_id,self.in_zone_sensor, negated=True).condition_met()[0]): #Negated meaning its no longer in the sensor
                    self.outside_zone[cube_name] = True
                    return -100
                else:
                    return 0
            else:
                return 0
            
    def task_complete_reward(self)->float:
        if (DetectedCondition(self.cube1, self.in_zone_sensor, negated=True).condition_met()[0] and
            GraspedCondition(self.robot.gripper, self.cube1).condition_met()[0] and 
            DetectedCondition(self.distractor1, self.in_zone_sensor).condition_met()[0] and
            DetectedCondition(self.distractor2, self.in_zone_sensor).condition_met()[0] and
            DetectedCondition(self.distractor3, self.in_zone_sensor).condition_met()[0] and
            DetectedCondition(self.distractor4, self.in_zone_sensor).condition_met()[0] and
            DetectedCondition(self.distractor5, self.in_zone_sensor).condition_met()[0] ):
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