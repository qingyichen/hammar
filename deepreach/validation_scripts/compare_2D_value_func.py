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

def val_fn_compare(models, ground_truth_fn, times=None, level=0.001):
    """
    Plot ground truth, model1, and model2 in a 3x3 grid.
    
    models: [model1, model2]
    ground_truth_fn: function(time, coords) -> array of shape (sidelen, sidelen)
    times: list of 3 time values (for 3 rows)
    """
    sidelen = 1000
    mgrid_coords = dataio.get_mgrid(sidelen)

    # Create base coords template
    mgrid_counter = 0
    extent = 0.6
    vextent = 0.25
    scale = 1.25
    fontsize = 20
    labelsize = 16

    base_coords = torch.zeros(sidelen ** 2, 3)
    for i in range(1,3):
        base_coords[:,i] = mgrid_coords[:,mgrid_counter] * extent
        mgrid_counter += 1
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    if times is None:
        times = [np.pi/8, np.pi/4, np.pi/8 * 3]
    times_names = ['T-$\pi$/8', 'T-$\pi$/4', 'T-3$\pi$/8']

    for row, t in enumerate(times):
        coords = base_coords.clone()
        coords[:,0] = t
        coords = coords.cuda()

        # --- Ground truth ---
        np_coords = coords.cpu().numpy().reshape(-1, 3)
        gt_valfunc = ground_truth_fn(t, np_coords) / scale
        off_manifold_mask = np.abs(np.linalg.norm(np_coords[..., 1:], axis=-1) - 0.5) > 0.01
        off_manifold_value = -vextent * 0.75
        gt_valfunc[off_manifold_mask] = off_manifold_value
        gt_valfunc = gt_valfunc.reshape((sidelen, sidelen))
        im = axes[row,0].imshow(gt_valfunc.T, alpha=0.8, cmap='coolwarm_r', origin='lower',
                                vmin=-vextent, vmax=vextent, extent=(-extent, extent, -extent, extent))
        axes[row,0].set_title(f"Ground Truth (t={times_names[row]})", fontsize=fontsize)
        axes[row,0].tick_params(labelleft=True, labelbottom=False, labelsize=labelsize)
        if row == 2:
            axes[row,0].set_xlabel("x1", fontsize=fontsize)
            axes[row,0].tick_params(labelleft=True, labelbottom=True, labelsize=labelsize)
        axes[row,0].set_ylabel("x2", fontsize=fontsize)

        # --- Models ---
        for col, model in enumerate(models):
            model_in = {'coords': coords}
            with torch.no_grad():
                model_out = model(model_in)['model_out']
            valfunc = model_out.detach().cpu().numpy() / scale
            valfunc[off_manifold_mask] = off_manifold_value
            valfunc = valfunc.reshape((sidelen, sidelen))

            im = axes[row,col+1].imshow(valfunc.T, alpha=0.8, cmap='coolwarm_r', origin='lower',
                                        vmin=-vextent, vmax=vextent, extent=(-extent, extent, -extent, extent))
            axes[row,col+1].tick_params(labelleft=False, labelbottom=False)
            if col == 0:
                axes[row,col+1].set_title(f"Manifold-Constrained HJR (t={times_names[row]})", fontsize=fontsize)
            else:
                axes[row,col+1].set_title(f"Unconstrained HJR (t={times_names[row]})", fontsize=fontsize)
            if row == 2:
                axes[row,col+1].set_xlabel("x1", fontsize=fontsize)
                axes[row,col+1].tick_params(labelbottom=True, labelsize=labelsize)
            
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    model_mc = load_model('./hji_logs/logs_circle_particle/experiment_mc_seed0')
    model_no_mc = load_model('./hji_logs/logs_circle_particle/experiment_no_mc_seed0')
    models = [model_mc, model_no_mc]

    def ground_truth_fn(t, coords):
        results = 0.5 * np.arccos(np.clip(2 * coords[:,1], a_min=-1., a_max=1.0)) - t  # Example level set function
        results[results < 0] = 0
        return results
    times = None #[0., 0.5, 1.0]  # three rows
    fig = val_fn_compare(models, ground_truth_fn, times=times)
    fig.savefig("comparison_grid.png", dpi=300)