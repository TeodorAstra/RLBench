from gym.envs.registration import register
import rlbench.backend.task as task
import os
from rlbench.utils import name_to_task_class
from rlbench.gym.rlbench_env import RLBenchEnv

TASKS = [t for t in os.listdir(task.TASKS_PATH)
         if t != '__init__.py' and t.endswith('.py')]

for task_file in TASKS:
    task_name = task_file.split('.py')[0]
    task_class = name_to_task_class(task_name)
    register(
        id='%s-state-v0' % task_name,
        entry_point='rlbench.gym:RLBenchEnv',
        kwargs={
            'task_class': task_class,
            'observation_mode': 'state'
        },
        max_episode_steps=15
    )
    register(
        id='%s-vision-v0' % task_name,
        entry_point='rlbench.gym:RLBenchEnv',
        kwargs={
            'task_class': task_class,
            'observation_mode': 'vision'
        }
    )


#____________________CUSTOM_ max_episode_steps_______________________________
    

register(
        id='slide_block_to_target-state-v0',
        entry_point='rlbench.gym:RLBenchEnv',
        kwargs={
            'task_class': name_to_task_class('slide_block_to_target'),
            'observation_mode': 'state'
        },
        max_episode_steps=30
    )

register(
        id='teodor_remove_from_zone-state-v0',
        entry_point='rlbench.gym:RLBenchEnv',
        kwargs={
            'task_class': name_to_task_class('teodor_remove_from_zone'),
            'observation_mode': 'state'
        },
        max_episode_steps=40
    )
register(
        id='teodor_remove_from_zone-vision-v0',
        entry_point='rlbench.gym:RLBenchEnv',
        kwargs={
            'task_class': name_to_task_class('teodor_remove_from_zone'),
            'observation_mode': 'vision'
        },
        max_episode_steps=30
    )

register(
        id='teodor_remove_from_zone_v2-state-v0',
        entry_point='rlbench.gym:RLBenchEnv',
        kwargs={
            'task_class': name_to_task_class('teodor_remove_from_zone'),
            'observation_mode': 'state'
        },
        max_episode_steps=40
    )

register(
        id='teodor_zone_pick_and_place-state-v0',
        entry_point='rlbench.gym:RLBenchEnv',
        kwargs={
            'task_class': name_to_task_class('teodor_zone_pick_and_place'),
            'observation_mode': 'state'
        },
        max_episode_steps=50
    )

register(
        id='teodor_zone_pick_and_place_v2-state-v0',
        entry_point='rlbench.gym:RLBenchEnv',
        kwargs={
            'task_class': name_to_task_class('teodor_zone_pick_and_place_v2'),
            'observation_mode': 'state'
        },
        max_episode_steps=150
    )

register(
        id='teodor_pick_and_lift-state-v0',
        entry_point='rlbench.gym:RLBenchEnv',
        kwargs={
            'task_class': name_to_task_class('teodor_pick_and_lift'),
            'observation_mode': 'state'
        },
        max_episode_steps=80
    )

register(
        id='teodor_extract_with_distractors-state-v0',
        entry_point='rlbench.gym:RLBenchEnv',
        kwargs={
            'task_class': name_to_task_class('teodor_extract_with_distractors'),
            'observation_mode': 'state'
        },
        max_episode_steps=80
    )

register(
        id='teodor_object_iso-state-v0',
        entry_point='rlbench.gym:RLBenchEnv',
        kwargs={
            'task_class': name_to_task_class('teodor_object_iso'),
            'observation_mode': 'state'
        },
        max_episode_steps=200
    )

register(
        id='teodor_extract_with_distractors_scrambled-state-v0',
        entry_point='rlbench.gym:RLBenchEnv',
        kwargs={
            'task_class': name_to_task_class('teodor_extract_with_distractors_scrambled'),
            'observation_mode': 'state'
        },
        max_episode_steps=80
    )

register(
        id='real_grasping-state-v0',
        entry_point='rlbench.gym:RLBenchEnv',
        kwargs={
            'task_class': name_to_task_class('real_grasping'),
            'observation_mode': 'state'
        },
        max_episode_steps=60
    )

register(
        id='real_grasping_extract-state-v0',
        entry_point='rlbench.gym:RLBenchEnv',
        kwargs={
            'task_class': name_to_task_class('real_grasping_extract'),
            'observation_mode': 'state'
        },
        max_episode_steps=60
    )

register(
        id='real_grasping_extract_v2-state-v0',
        entry_point='rlbench.gym:RLBenchEnv',
        kwargs={
            'task_class': name_to_task_class('real_grasping_extract_v2'),
            'observation_mode': 'state'
        },
        max_episode_steps=60
    )
register(
        id='teodor_grasp_extract_random_pos-state-v0',
        entry_point='rlbench.gym:RLBenchEnv',
        kwargs={
            'task_class': name_to_task_class('teodor_grasp_extract_random_pos'),
            'observation_mode': 'state'
        },
        max_episode_steps=60
    )  