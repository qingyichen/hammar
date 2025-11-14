import os
import torch
import pickle
import matplotlib.pyplot as plt

def combine_datasets(num_sub_datasets = 80, num_data = 32000, dataset_dir = 'UR5_datasets_and_training/UR5_d0.6_mesh_distances_dataset', suffix='_manifold_only'):
    alldataset_qpos = torch.zeros((num_sub_datasets * num_data, 12))
    alldataset_distances = torch.zeros(num_sub_datasets * num_data)

    for i_dataset in range(num_sub_datasets):
        filename = f'seed{i_dataset}num_data{num_data}_manifold.pkl'
        with open(os.path.join(dataset_dir, filename), 'rb') as f:
            data = pickle.load(f)
            print(data['qpos'][0])
            alldataset_qpos[i_dataset * num_data: (i_dataset+1) * num_data] = data['qpos']
            alldataset_distances[i_dataset * num_data: (i_dataset+1) * num_data] = data['distances']
    with open(os.path.join(dataset_dir, f'seed{0}-{num_sub_datasets-1}num_data{num_sub_datasets * num_data}_manifold.pkl'), 'wb') as f:
        data = {
            'qpos': alldataset_qpos,
            'distances': alldataset_distances,
        }
        pickle.dump(data, f)
        
    print("Finished combining sub-datasets.")
    
def distances_histograms(dataset_filename='UR5_datasets_and_training/UR5_d0.6_mesh_distances_dataset/seed0-79num_data2560000.pkl'):
    with open(dataset_filename, 'rb') as f:
        data = pickle.load(f)
        distances = data['distances']
    plt.hist(distances)
    plt.xlabel('Distances [m]')
    # plt.ylabel('Number of Occurrences')
    plt.savefig('distance_hist.png', dpi=600)
    
def combine_two_datasets():
    with open(f'UR5_datasets_and_training/UR5{suffix}_mesh_distances_dataset/seed0-79num_data2560000.pkl','rb') as f:
        data1 = pickle.load(f)
    with open(f'UR5_datasets_and_training/UR5{suffix}_mesh_distances_dataset/seed0-79num_data2560000_manifold.pkl', 'rb') as f:
        data2 = pickle.load(f)
    dataset_dir = f'UR5_datasets_and_training/UR5{suffix}_mesh_distances_dataset'
    with open(os.path.join(dataset_dir, f'mixed_dataset.pkl'), 'wb') as f:
        data = {
            'qpos': torch.cat([data1['qpos'], data2['qpos']], dim=0),
            'distances': torch.cat([data1['distances'], data2['distances']], dim=0),
        }
        pickle.dump(data, f)

if __name__ == '__main__':
    suffix = '_object'
    # combine_datasets(dataset_dir=f'UR5_datasets_and_training/UR5{suffix}_mesh_distances_dataset', suffix=suffix)
    # distances_histograms(dataset_filename=f'UR5_datasets_and_training/UR5{suffix}_mesh_distances_dataset/seed0-79num_data2560000_manifold.pkl')
    combine_two_datasets()
    
    
    