import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from envs.DualArmCup_env import DualArmEnv
from envs.DualArmDoorway_env import DualArmPuzzleEnv
from envs.DualArmObject_env import DualArmObjectEnv
import zonopyrobots as robots2
import numpy as np
from tqdm import tqdm
import torch
import pickle

def collect_inter_arm_distance(num_sub_datasets=80, num_data=32000, manifold_only=True):
    for i_dataset in range(num_sub_datasets):
        env = DualArmEnv(robot=robot.urdf,
                            timestep_discretization=1,
                            step_type='direct',
                            check_self_collision=True,
                            verbose_self_collision=True,
                            seed=i_dataset)
        filename = f'seed{i_dataset}num_data{num_data}{"_manifold" if manifold_only else ""}.pkl'
        subdataset_qpos = np.zeros((num_data, 12))
        subdataset_distances = np.zeros(num_data)

        for i in tqdm(range(num_data)):
            distance = env.distance_sampler(consider_self_collision=False, manifold_only=manifold_only)
            env_to_qpos_mapping = np.array([0, 3, 4, 7, 9, 10, 1, 2, 5, 6, 8, 11], dtype=np.int32)
            qpos = env.qpos[env_to_qpos_mapping]
            subdataset_qpos[i] = qpos.copy()
            subdataset_distances[i] = distance
            
            
        with open(os.path.join(dataset_dir, filename), 'wb') as f:
            data = {
                'qpos': torch.from_numpy(subdataset_qpos).float(),
                'distances': torch.from_numpy(subdataset_distances).float()
            }
            pickle.dump(data, f)
    print(f"{num_sub_datasets} sub-datasets with each having {num_data} generated.")
    

def collect_puzzle_arm_distance(num_sub_datasets=80, num_data=32000, manifold_only=True):
    for i_dataset in range(num_sub_datasets):
        env = DualArmPuzzleEnv(robot=robot.urdf,
                            timestep_discretization=1,
                            step_type='direct',
                            check_self_collision=False,
                            verbose_self_collision=False,
                            seed=i_dataset)
        filename = f'seed{i_dataset}num_data{num_data}{"_manifold" if manifold_only else ""}.pkl'
        subdataset_qpos = np.zeros((num_data, 12))
        subdataset_distances = np.zeros(num_data)

        for i in tqdm(range(num_data)):
            distance = env.distance_sampler(consider_self_collision=False, manifold_only=manifold_only)
            env_to_qpos_mapping = np.array([0, 3, 4, 7, 9, 10, 1, 2, 5, 6, 8, 11], dtype=np.int32)
            qpos = env.qpos[env_to_qpos_mapping]
            subdataset_qpos[i] = qpos.copy()
            subdataset_distances[i] = distance
            
            
        with open(os.path.join(dataset_dir, filename), 'wb') as f:
            data = {
                'qpos': torch.from_numpy(subdataset_qpos).float(),
                'distances': torch.from_numpy(subdataset_distances).float()
            }
            pickle.dump(data, f)
    print(f"{num_sub_datasets} sub-datasets with each having {num_data} generated.")
    
    
def collect_object_arm_distance(num_sub_datasets=80, num_data=32000, manifold_only=True):
    for i_dataset in range(num_sub_datasets):
        env = DualArmObjectEnv(robot=robot.urdf,
                            timestep_discretization=1,
                            step_type='direct',
                            check_self_collision=False,
                            verbose_self_collision=False,
                            seed=i_dataset)
        filename = f'seed{i_dataset}num_data{num_data}{"_manifold" if manifold_only else ""}.pkl'
        subdataset_qpos = np.zeros((num_data, 12))
        subdataset_distances = np.zeros(num_data)

        for i in tqdm(range(num_data)):
            distance = env.distance_sampler(consider_self_collision=False, manifold_only=manifold_only)
            env_to_qpos_mapping = np.array([0, 3, 4, 7, 9, 10, 1, 2, 5, 6, 8, 11], dtype=np.int32)
            qpos = env.qpos[env_to_qpos_mapping]
            subdataset_qpos[i] = qpos.copy()
            subdataset_distances[i] = distance
            
            
        with open(os.path.join(dataset_dir, filename), 'wb') as f:
            data = {
                'qpos': torch.from_numpy(subdataset_qpos).float(),
                'distances': torch.from_numpy(subdataset_distances).float()
            }
            pickle.dump(data, f)
    print(f"{num_sub_datasets} sub-datasets with each having {num_data} generated.")
    

if __name__ == '__main__':
    cup_constraint = False
    if cup_constraint:
        robots2.DEBUG_VIZ = False
        robot = robots2.ZonoArmRobot.load(os.path.join(os.getcwd(),'envs/arm_urdfs/ur5/dual_ur5_sphere_gripper.urdf'), create_joint_occupancy=False)

        # self_collision_test()
        num_sub_datasets = 80
        check_self_collision = False
        num_data = 32000
        
        dataset_dir = f"UR5_datasets_and_training/UR5_cup_d0.6_mesh_distances_{'sc_' if check_self_collision else ''}dataset"
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        collect_inter_arm_distance(manifold_only=True)
        
    

    puzzle_constraint = False
    if puzzle_constraint:
        robots2.DEBUG_VIZ = False
        robot = robots2.ZonoArmRobot.load(os.path.join(os.getcwd(),'envs/arm_urdfs/ur5_doorway/dual_ur5_no_gripper.urdf'), create_joint_occupancy=False)

        # self_collision_test()
        num_sub_datasets = 80
        check_self_collision = False
        num_data = 32000


        dataset_dir = f"UR5_datasets_and_training/UR5_doorway_mesh_distances_{'sc_' if check_self_collision else ''}dataset"
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        collect_puzzle_arm_distance(manifold_only=False)
        collect_puzzle_arm_distance(manifold_only=True)
        
        
    object_constraint = False
    if object_constraint:
        robots2.DEBUG_VIZ = False
        robot = robots2.ZonoArmRobot.load(os.path.join(os.getcwd(),'envs/arm_urdfs/ur5_object/dual_ur5_sphere_gripper.urdf'), create_joint_occupancy=False)

        # self_collision_test()
        num_sub_datasets = 80
        check_self_collision = False
        num_data = 32000
        
        dataset_dir = f"UR5_datasets_and_training/UR5_object_mesh_distances_{'sc_' if check_self_collision else ''}dataset"
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        collect_object_arm_distance(manifold_only=False)
        collect_object_arm_distance(manifold_only=True)
    
    
    
    

    
    