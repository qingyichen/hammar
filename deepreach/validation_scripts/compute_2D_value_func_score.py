# Enable import from parent package
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Arial'

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import modules, dataio

import torch
import numpy as np
import math
from torch.utils.data import DataLoader
import configargparse
import scipy.io as spio
from sklearn.metrics import recall_score, precision_score, f1_score

def ground_truth_fn(t, coords):
        results = 0.5 * np.arccos(np.clip(2 * coords[:,1], a_min=-1., a_max=1.0)) - t 
        return results

def load_model(logging_root='./hji_logs/logs_circle_particle/experiment_mc_seed0'):
    # Setting to plot
    ckpt_path = f'{logging_root}/checkpoints/model_final.pth'
    activation = 'sine'

    # Initialize and load the model
    model = modules.Constrained_Particle_ICNet(in_features=3, out_features=1, type=activation, mode='mlp',
                                final_layer_factor=1., hidden_features=512, num_hidden_layers=3)
    model.cuda()
    checkpoint = torch.load(ckpt_path, weights_only=True)
    try:
        model_weights = checkpoint['model']
    except:
        model_weights = checkpoint
        model.load_state_dict(model_weights)
        model.eval()
        
    return model

def val_fn_score(models, ground_truth_fn, times=None, threshold=0.01):
    sidelen = 500
    mgrid_coords = dataio.get_mgrid(sidelen)

    # Create base coords template
    mgrid_counter = 0
    extent = 0.6
    # for printing
    numbers = []

    base_coords = torch.zeros(sidelen ** 2, 3)
    for i in range(1,3):
        base_coords[:,i] = mgrid_coords[:,mgrid_counter] * extent
        mgrid_counter += 1
    if times is None:
        times = [np.pi/8, np.pi/4, np.pi/8 * 3]
    times_names = ['T-$\pi$/8', 'T-$\pi$/4', 'T-3$\pi$/8']
    
    for t in (times):
        coords = base_coords.clone()
        coords[:,0] = t
        coords = coords.cuda()
        # --- Ground truth ---
        np_coords = coords.cpu().numpy().reshape(-1, 3)
        on_manifold_mask = np.abs(np.linalg.norm(np_coords[..., 1:], axis=-1) - 0.5) < 0.01
        np_coords = np_coords[on_manifold_mask]
        gt_valfunc = ground_truth_fn(t, np_coords)
        gt_labels = (gt_valfunc < 0.).astype(np.int32).flatten()

        for i_model, model in enumerate(models):
            coords = torch.tensor(np_coords, dtype=torch.float32).cuda()
            model_in = {'coords': coords}
            with torch.no_grad():
                model_out = model(model_in)['model_out']
            valfunc = model_out.detach().cpu().numpy() 
            valfunc = valfunc.flatten()
            labels = (valfunc < threshold).astype(np.int32).flatten()
                        
            print(f"Time {t}, Model {i_model}: Accuracy={(labels==gt_labels).mean()}, Recall={recall_score(gt_labels, labels)}, Precision={precision_score(gt_labels, labels)}, F1 score={f1_score(gt_labels, labels)}")
            numbers += [f"{100*(labels==gt_labels).mean():.1f}", f"{100*recall_score(gt_labels, labels):.1f}", f"{100*precision_score(gt_labels, labels):.1f}", f"{100*f1_score(gt_labels, labels):.1f}"] 
    
        # print numbers to latex table
        # string = ""
        # for i in range(len(numbers)):
        #     string += numbers[i]
        #     if i % 4 == 3:
        #         string += "\\\\\midrule\n"
        #     else:
        #         string += " & "
        # print(string)  

if __name__ == '__main__':
    model_mc = load_model('./hji_logs/logs_circle_particle/experiment_mc_seed0')
    model_no_mc = load_model('./hji_logs/logs_circle_particle/experiment_no_mc_seed0')
    models = [model_mc, model_no_mc]

    times = None #[0., 0.5, 1.0]  # three rows
    val_fn_score(models, ground_truth_fn, times=times, threshold=0.03)
    