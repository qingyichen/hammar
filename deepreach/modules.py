import torch
from torch import nn
import numpy as np
from collections import OrderedDict
import math
import pytorch_kinematics as pk
import os, sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from deepreach.utils import line_segment_distances
from torch.autograd import grad


class BatchLinear(nn.Linear):
    '''A linear layer'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


class FCBlock(nn.Module):
    '''A fully connected neural network.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
                         'tanh':(nn.Tanh(), init_weights_xavier, None),
                         'selu':(nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus':(nn.Softplus(), init_weights_normal, None),
                         'elu':(nn.ELU(inplace=True), init_weights_elu, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(nn.Sequential(
            BatchLinear(in_features, hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(
                BatchLinear(hidden_features, hidden_features), nl
            ))

        if outermost_linear:
            self.net.append(nn.Sequential(BatchLinear(hidden_features, out_features)))
        else:
            self.net.append(nn.Sequential(
                BatchLinear(hidden_features, out_features), nl
            ))

        self.net = nn.Sequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = self.net(coords)
        return output


class SingleBVPNet(nn.Module):
    '''A canonical representation network for a BVP.'''

    def __init__(self, out_features=1, type='sine', in_features=2,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, **kwargs):
        super().__init__()
        self.mode = mode
        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True, nonlinearity=type)
        print(self)

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Enables us to compute gradients w.r.t. coordinates
        coords_org = model_input['coords'].clone().detach().requires_grad_(True)
        coords = coords_org

        output = self.net(coords)
        return {'model_in': coords_org, 'model_out': output}


class Constrained_Particle_ICNet(SingleBVPNet):
    def __init__(self, out_features=1, type='sine', in_features=3,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, **kwargs):
        super().__init__(out_features, type, in_features, mode, hidden_features, num_hidden_layers, **kwargs)
        
    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Enables us to compute gradients w.r.t. coordinates
        coords_org = model_input['coords'].clone().detach().requires_grad_(True)
        coords = coords_org

        output = self.net(coords)
        return {'model_in': coords_org, 'model_out': output}

        
class Cup_UR5ICNet(SingleBVPNet):
    def __init__(self, out_features=1, type='sine', in_features=13, mode='mlp', hidden_features=256, num_hidden_layers=3, 
                 bc_model=None, use_symmetry=False, num_links=12, max_joint_velocity=0.5,
                 provide_initial_condition=False, compute_ee_gradient=False, 
                 **kwargs):
        super().__init__(out_features, type, in_features, mode, hidden_features, num_hidden_layers, **kwargs)
        
        self.num_links = num_links
        self.max_joint_velocity = max_joint_velocity
        
        self.use_symmetry = use_symmetry
        self.provide_initial_condition = provide_initial_condition
        
        self.bc_model = bc_model
        self.compute_ee_gradient = compute_ee_gradient
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        robot1_urdf_path = 'envs/arm_urdfs/ur5/ur5_1.urdf'
        self.chain1 = pk.build_serial_chain_from_urdf(open(robot1_urdf_path).read(), f"ee_link").to(device=self.device)
        robot2_urdf_path = 'envs/arm_urdfs/ur5/ur5_2.urdf'
        self.chain2 = pk.build_serial_chain_from_urdf(open(robot2_urdf_path).read(), f"ee_link").to(device=self.device)
    
    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        
        symmetry_mask = None
        if self.use_symmetry:
            # symmetry: V(q11, q12, ..., q21, q22, ...) = V(pi-q11, -q12, ..., pi-q21, -q22, ...)
            states = model_input['coords'][...,1:]
            symmetry_mask = torch.logical_or(states[..., 6] < -torch.pi / 2, states[..., 6] > torch.pi / 2)                
            temp = model_input['coords'][symmetry_mask]
            temp *= -1
            temp[..., 1] += torch.pi
            temp[..., 7] += torch.pi
            temp[..., 1] = (temp[..., 1] + torch.pi) % (2 * torch.pi) - torch.pi
            temp[..., 7] = (temp[..., 7] + torch.pi) % (2 * torch.pi) - torch.pi
            model_input['coords'][symmetry_mask] = temp
        
        coords_org = model_input['coords'].clone().detach().requires_grad_(True)
        coords = coords_org
        
        if self.provide_initial_condition:
            states = coords_org[...,1:]
            boundary_condition = self.bc_model(states)
            projection_matrix1 = None
            projection_matrix2 = None
            A = None
            B = None
            if self.compute_ee_gradient:
                q1 = states[...,0:6].view(-1, 6)
                fk1 = self.chain1.forward_kinematics(q1, end_only=True)
                ee1 = fk1.get_matrix()[..., :3, 2] # z-orientation
                q2 = states[...,6:].view(-1, 6)
                fk2 = self.chain2.forward_kinematics(q2, end_only=True)
                ee2 = fk2.get_matrix()[..., :3, 2] # z-orientation
                
                dee1dq1 = torch.zeros(ee1.shape + (q1.shape[-1],), device=q1.device)
                dee2dq2 = torch.zeros(ee2.shape + (q2.shape[-1],), device=q2.device)
                    
                for i in range(2):
                    # z-axis upward constraint
                    y1 = ee1[:,i]
                    y2 = ee2[:,i]
                    dee1dq1[:,i,:] = grad(y1, q1, grad_outputs=torch.ones_like(y1), retain_graph=True)[0]
                    dee2dq2[:,i,:] = grad(y2, q2, grad_outputs=torch.ones_like(y2), retain_graph=True)[0]
                dee1dq1 = dee1dq1.unsqueeze(0)
                dee2dq2 = dee2dq2.unsqueeze(0)
                
                A = dee1dq1[..., :2, :]
                A_T = torch.transpose(A, -1, -2)
                projection_matrix1 = torch.eye(6, device=dee1dq1.device).expand(dee1dq1.shape[0], dee1dq1.shape[1], 6, 6)
                projection_matrix1 = projection_matrix1 - A_T @ torch.linalg.inv(A @ A_T) @ A
                
                B = dee2dq2[..., :2, :]
                B_T = torch.transpose(B, -1, -2)
                projection_matrix2 = torch.eye(6, device=dee2dq2.device).expand(dee2dq2.shape[0], dee2dq2.shape[1], 6, 6)
                projection_matrix2 = projection_matrix2 - B_T @ torch.linalg.inv(B @ B_T) @ B

            output = boundary_condition + self.net(coords)           
            return {'model_in': coords_org, 'model_out': output, 'symmetry_mask': symmetry_mask, 'dee1dq1_projection_matrix': projection_matrix1, 'dee2dq2_projection_matrix': projection_matrix2, 'A': A, 'B': B}
        else:
            output = self.net(coords)
            return {'model_in': coords_org, 'model_out': output, 'symmetry_mask': symmetry_mask, 'dee1dq1_projection_matrix': projection_matrix1, 'dee2dq2_projection_matrix': projection_matrix2, 'A': A, 'B': B}
        

class Doorway_UR5ICNet(SingleBVPNet):
    def __init__(self, out_features=1, type='sine', in_features=13, mode='mlp', hidden_features=256, num_hidden_layers=3, 
                 bc_model=None, use_symmetry=False, num_links=12, max_joint_velocity=0.5,
                 provide_initial_condition=False, compute_ee_gradient=False, 
                 **kwargs):
        super().__init__(out_features, type, in_features, mode, hidden_features, num_hidden_layers, **kwargs)
        
        self.num_links = num_links
        self.max_joint_velocity = max_joint_velocity
        
        self.use_symmetry = use_symmetry
        self.provide_initial_condition = provide_initial_condition
        
        self.bc_model = bc_model
        self.compute_ee_gradient = compute_ee_gradient
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        robot1_urdf_path = 'envs/arm_urdfs/ur5_doorway/ur5_1.urdf'
        self.chain1 = pk.build_serial_chain_from_urdf(open(robot1_urdf_path).read(), f"ee_link").to(device=self.device)
        robot2_urdf_path = 'envs/arm_urdfs/ur5_doorway/ur5_2.urdf'
        self.chain2 = pk.build_serial_chain_from_urdf(open(robot2_urdf_path).read(), f"ee_link").to(device=self.device)
    
    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        
        symmetry_mask = None
        if self.use_symmetry:
            # symmetry: V(q11, q12, ..., q21, q22, ...) = V(pi-q11, -q12, ..., pi-q21, -q22, ...)
            states = model_input['coords'][...,1:]
            symmetry_mask = torch.logical_or(states[..., 6] < -torch.pi / 2, states[..., 6] > torch.pi / 2)                
            temp = model_input['coords'][symmetry_mask]
            temp *= -1
            temp[..., 1] += torch.pi
            temp[..., 7] += torch.pi
            temp[..., 1] = (temp[..., 1] + torch.pi) % (2 * torch.pi) - torch.pi
            temp[..., 7] = (temp[..., 7] + torch.pi) % (2 * torch.pi) - torch.pi
            model_input['coords'][symmetry_mask] = temp
        
        coords_org = model_input['coords'].clone().detach().requires_grad_(True)
        coords = coords_org
        
        if self.provide_initial_condition:
            states = coords_org[...,1:]
            boundary_condition = self.bc_model(states)
            projection_matrix1 = None
            projection_matrix2 = None
            if self.compute_ee_gradient:
                q1 = states[...,0:6].view(-1, 6)
                fk1 = self.chain1.forward_kinematics(q1, end_only=True)
                ee1 = fk1.get_matrix()[..., :3, 0] # x-orientation
                q2 = states[...,6:].view(-1, 6)
                fk2 = self.chain2.forward_kinematics(q2, end_only=True)
                ee2 = fk2.get_matrix()[..., :3, 0] # x-orientation
                
                dee1dq1 = torch.zeros(ee1.shape[:-1] + (4, q1.shape[-1],), device=q1.device)
                dee2dq2 = torch.zeros(ee2.shape[:-1] + (4, q2.shape[-1],), device=q2.device)
                # 4 because orientation 2, z-position 1, x-y position 1
                
                z1 = fk1.get_matrix()[..., 2, 3] # z-position
                z2 = fk2.get_matrix()[..., 2, 3] # z-position
                xy1 = fk1.get_matrix()[..., 0, 3] + fk1.get_matrix()[..., 1, 3] # x-y position
                xy2 = fk2.get_matrix()[..., 0, 3] - fk2.get_matrix()[..., 1, 3] # x-y position
                
                for i in range(2):
                    # x-axis downward constraint
                    y1 = ee1[:,i]
                    y2 = ee2[:,i]
                    dee1dq1[:,i,:] = grad(y1, q1, grad_outputs=torch.ones_like(y1), retain_graph=True)[0]
                    dee2dq2[:,i,:] = grad(y2, q2, grad_outputs=torch.ones_like(y2), retain_graph=True)[0]
                dee1dq1[:,2,:] = grad(z1, q1, grad_outputs=torch.ones_like(z1), retain_graph=True)[0]
                dee2dq2[:,2,:] = grad(z2, q2, grad_outputs=torch.ones_like(z2), retain_graph=True)[0]
                dee1dq1[:,3,:] = grad(xy1, q1, grad_outputs=torch.ones_like(xy1), retain_graph=True)[0]
                dee2dq2[:,3,:] = grad(xy2, q2, grad_outputs=torch.ones_like(xy2), retain_graph=True)[0]
                dee1dq1 = dee1dq1.unsqueeze(0)
                dee2dq2 = dee2dq2.unsqueeze(0)
                
                A = dee1dq1
                A_T = torch.transpose(A, -1, -2)
                projection_matrix1 = torch.eye(6, device=dee1dq1.device).expand(dee1dq1.shape[0], dee1dq1.shape[1], 6, 6)
                projection_matrix1 = projection_matrix1 - A_T @ torch.linalg.inv(A @ A_T) @ A
                
                B = dee2dq2
                B_T = torch.transpose(B, -1, -2)
                projection_matrix2 = torch.eye(6, device=dee2dq2.device).expand(dee2dq2.shape[0], dee2dq2.shape[1], 6, 6)
                projection_matrix2 = projection_matrix2 - B_T @ torch.linalg.inv(B @ B_T) @ B

            output = boundary_condition + self.net(coords)           
            return {'model_in': coords_org, 'model_out': output, 'symmetry_mask': symmetry_mask, 'dee1dq1_projection_matrix': projection_matrix1, 'dee2dq2_projection_matrix': projection_matrix2}
        else:
            output = self.net(coords)
            return {'model_in': coords_org, 'model_out': output, 'symmetry_mask': symmetry_mask, 'dee1dq1_projection_matrix': projection_matrix1, 'dee2dq2_projection_matrix': projection_matrix2}
        
        

class Object_UR5ICNet(SingleBVPNet):
    def __init__(self, out_features=1, type='sine', in_features=13, mode='mlp', hidden_features=256, num_hidden_layers=3, 
                 bc_model=None, use_symmetry=False, num_links=12, max_joint_velocity=0.5,
                 provide_initial_condition=False, compute_ee_gradient=False, 
                 **kwargs):
        super().__init__(out_features, type, in_features, mode, hidden_features, num_hidden_layers, **kwargs)
        
        self.num_links = num_links
        self.max_joint_velocity = max_joint_velocity
        
        self.use_symmetry = use_symmetry
        self.provide_initial_condition = provide_initial_condition
        
        self.bc_model = bc_model
        self.compute_ee_gradient = compute_ee_gradient
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        robot1_urdf_path = 'envs/arm_urdfs/ur5_object/ur5_1.urdf'
        self.chain1 = pk.build_serial_chain_from_urdf(open(robot1_urdf_path).read(), f"ee_link").to(device=self.device)
        robot2_urdf_path = 'envs/arm_urdfs/ur5_object/ur5_2.urdf'
        self.chain2 = pk.build_serial_chain_from_urdf(open(robot2_urdf_path).read(), f"ee_link").to(device=self.device)
    
    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        
        symmetry_mask = None
        if self.use_symmetry:
            # symmetry: V(q11, q12, ..., q21, q22, ...) = V(pi-q11, -q12, ..., pi-q21, -q22, ...)
            states = model_input['coords'][...,1:]
            symmetry_mask = torch.logical_or(states[..., 6] < -torch.pi / 2, states[..., 6] > torch.pi / 2)                
            temp = model_input['coords'][symmetry_mask]
            temp *= -1
            temp[..., 1] += torch.pi
            temp[..., 7] += torch.pi
            temp[..., 1] = (temp[..., 1] + torch.pi) % (2 * torch.pi) - torch.pi
            temp[..., 7] = (temp[..., 7] + torch.pi) % (2 * torch.pi) - torch.pi
            model_input['coords'][symmetry_mask] = temp
        
        coords_org = model_input['coords'].clone().detach().requires_grad_(True)
        coords = coords_org
        
        if self.provide_initial_condition:
            states = coords_org[...,1:]
            boundary_condition = self.bc_model(states)
            projection_matrix1 = None
            projection_matrix2 = None
            if self.compute_ee_gradient:
                q1 = states[...,0:6].view(-1, 6)
                fk1 = self.chain1.forward_kinematics(q1, end_only=True)
                R1 = fk1.get_matrix()[..., :3, :3] # orientation
                q2 = states[...,6:].view(-1, 6)
                fk2 = self.chain2.forward_kinematics(q2, end_only=True)
                R2 = fk2.get_matrix()[..., :3, :3] # orientation
                
                R_d = torch.tensor([[ 0., -1.,  0.],
                    [ 0.,  0.,  1.],
                    [-1.,  0.,  0.]], device=q1.device, dtype=q1.dtype)
                M1 = (R_d.transpose(-1, -2) @ R1 - R1.transpose(-1, -2) @ R_d) / 2
                M2 = (R_d.transpose(-1, -2) @ R2 - R2.transpose(-1, -2) @ R_d) / 2
                e1 = torch.stack([M1[...,2,1], M1[...,0,2], M1[...,1,0]], dim=-1)
                e2 = torch.stack([M2[...,2,1], M2[...,0,2], M2[...,1,0]], dim=-1)
                
                dee1dq1 = torch.zeros(e1.shape + (q1.shape[-1],), device=q1.device)
                dee2dq2 = torch.zeros(e2.shape + (q2.shape[-1],), device=q2.device)
                
                for i in range(3):
                    # orientation constraint
                    y1 = e1[:,i]
                    y2 = e2[:,i]
                    dee1dq1[:,i,:] = grad(y1, q1, grad_outputs=torch.ones_like(y1), retain_graph=True)[0]
                    dee2dq2[:,i,:] = grad(y2, q2, grad_outputs=torch.ones_like(y2), retain_graph=True)[0]
                    
                dee1dq1 = dee1dq1.unsqueeze(0)
                dee2dq2 = dee2dq2.unsqueeze(0)
                                
                A = dee1dq1
                A_T = torch.transpose(A, -1, -2)
                projection_matrix1 = torch.eye(6, device=dee1dq1.device).expand(dee1dq1.shape[0], dee1dq1.shape[1], 6, 6)
                projection_matrix1 = projection_matrix1 - A_T @ torch.linalg.pinv(A @ A_T) @ A
                
                B = dee2dq2
                B_T = torch.transpose(B, -1, -2)
                projection_matrix2 = torch.eye(6, device=dee2dq2.device).expand(dee2dq2.shape[0], dee2dq2.shape[1], 6, 6)
                projection_matrix2 = projection_matrix2 - B_T @ torch.linalg.pinv(B @ B_T) @ B

            output = boundary_condition + self.net(coords)           
            return {'model_in': coords_org, 'model_out': output, 'symmetry_mask': symmetry_mask, 'dee1dq1_projection_matrix': projection_matrix1, 'dee2dq2_projection_matrix': projection_matrix2}
        else:
            output = self.net(coords)
            return {'model_in': coords_org, 'model_out': output, 'symmetry_mask': symmetry_mask, 'dee1dq1_projection_matrix': projection_matrix1, 'dee2dq2_projection_matrix': projection_matrix2}
        
        




########################
# Initialization methods
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # grab from upstream pytorch branch and paste here for now
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def init_weights_trunc_normal(m):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            fan_in = m.weight.size(1)
            fan_out = m.weight.size(0)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            mean = 0.
            # initialize with the same behavior as tf.truncated_normal
            # "The generated values follow a normal distribution with specified mean and
            # standard deviation, except that values whose magnitude is more than 2
            # standard deviations from the mean are dropped and re-picked."
            _no_grad_trunc_normal_(m.weight, mean, std, -2 * std, 2 * std)


def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_selu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)
