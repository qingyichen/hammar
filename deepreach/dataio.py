import numpy as np
import torch
from torch.utils.data import Dataset
import pytorch_kinematics as pk

def wrap_joint(configs):
    return (configs + torch.pi) % (2 * torch.pi) - torch.pi

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


def to_uint8(x):
    return (255. * x).astype(np.uint8)


def to_numpy(x):
    return x.detach().cpu().numpy()


def gaussian(x, mu=[0, 0], sigma=1e-4, d=2):
    x = x.numpy()
    if isinstance(mu, torch.Tensor):
        mu = mu.numpy()

    q = -0.5 * ((x - mu) ** 2).sum(1)
    return torch.from_numpy(1 / np.sqrt(sigma ** d * (2 * np.pi) ** d) * np.exp(q / sigma)).float()


class ReachabilityUR5CupSource(Dataset):
    def __init__(self, numpoints, 
        collisionR=0.0, max_joint_velocity=1.0, num_links=12,
        bc_model=None, device='cpu', pretrain=False, tMin=0.0, tMax=0.6, counter_start=0, counter_end=100e3, 
        pretrain_iters=2000, num_src_samples=1000, seed=0, use_symmetry=False, manifold_constrained=False, manifold_tolerance=0.03):
        super().__init__()
        torch.manual_seed(0)
        
        self.bc_model = bc_model.to(device)
        self.device = device

        self.pretrain = pretrain
        self.numpoints = numpoints
        
        self.max_joint_velocity = max_joint_velocity
        self.collisionR = collisionR

        self.num_states = num_links
        self.num_links = num_links

        self.tMax = tMax
        self.tMin = tMin

        self.N_src_samples = num_src_samples

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end 

        self.use_symmetry = use_symmetry
        self.manifold_constrained = manifold_constrained
        self.manifold_tolerance = manifold_tolerance
        if self.manifold_constrained:
            robot1_urdf_path = 'envs/arm_urdfs/ur5/ur5_1.urdf'
            self.chain1 = pk.build_serial_chain_from_urdf(open(robot1_urdf_path).read(), f"ee_link").to(device=self.device)
            
        torch.manual_seed(seed)

    def __len__(self):
        return 1

    def orientation_error(self, q, target_R, chain):
            T = chain.forward_kinematics(q, end_only=True).get_matrix()
            R = T[..., :3, 2]
            loss = torch.sum(torch.square(R - target_R), dim=-1)
            
            return loss            
        
    def gradient_descent(self, q, chain, target, stop_threshold=0.01):
        optimizer = torch.optim.SGD([q], lr=0.1)
        for _ in range(1000):
            optimizer.zero_grad()
            loss = self.orientation_error(q, target, chain)
            if loss.mean() < stop_threshold:
                return q
            loss = loss.sum()
            loss.backward()
            optimizer.step()
        return q


    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions

        # uniformly sample domain and include coordinates where source is non-zero 
        coords = torch.zeros(self.numpoints, self.num_states, device=self.device).uniform_(-1, 1) * torch.pi
        
        if self.pretrain:
            # only sample in time around the initial condition
            time = torch.ones(self.numpoints, 1, device=self.device) * start_time
            coords = torch.cat((time, coords), dim=1)
        else:
            # slowly grow time values from start time
            # this currently assumes start_time = 0 and max time value is tMax
            time = self.tMin + torch.zeros(self.numpoints, 1, device=self.device).uniform_(0, (self.tMax-self.tMin) * (self.counter / self.full_count))
            coords = torch.cat((time, coords), dim=1)

            # make sure we always have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time
        
        if self.use_symmetry:
            coords = coords[torch.logical_and(coords[..., 7] >= -torch.pi / 2, coords[..., 7] <= torch.pi / 2)]

        if self.manifold_constrained:
            q1 = coords[...,1:1+self.num_links//2].view(-1, self.num_links//2).requires_grad_()
            expected_z_orientation_column = torch.tensor([0., 0., 1.], device=self.device).expand(q1.shape[0], 3)
            noise = (torch.rand_like(expected_z_orientation_column) - 0.5) * 2 * self.manifold_tolerance
            expected_z_orientation_column = noise + expected_z_orientation_column
            q1 = wrap_joint(self.gradient_descent(q1, target=expected_z_orientation_column, chain=self.chain1).detach())
            coords[..., 1:1+self.num_links//2] = q1
        with torch.no_grad():
            states = coords[..., 1:]
            boundary_values = self.bc_model(states)
            
        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords[:, 0, None] == start_time)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False
        model_input = {'coords': coords}

        return model_input, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}
    
    
class ReachabilityUR5DoorwaySource(Dataset):
    def __init__(self, numpoints, 
        collisionR=0.0, max_joint_velocity=1.0, num_links=12,
        bc_model=None, device='cpu', pretrain=False, tMin=0.0, tMax=0.6, counter_start=0, counter_end=100e3, 
        pretrain_iters=2000, num_src_samples=1000, seed=0, use_symmetry=False, manifold_constrained=False, manifold_tolerance=0.03):
        super().__init__()
        torch.manual_seed(0)
        
        self.bc_model = bc_model.to(device)
        self.device = device

        self.pretrain = pretrain
        self.numpoints = numpoints
        
        self.max_joint_velocity = max_joint_velocity
        self.collisionR = collisionR

        self.num_states = num_links
        self.num_links = num_links

        self.tMax = tMax
        self.tMin = tMin

        self.N_src_samples = num_src_samples

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end 

        self.use_symmetry = use_symmetry
        self.manifold_constrained = manifold_constrained
        self.manifold_tolerance = manifold_tolerance
        if self.manifold_constrained:
            robot1_urdf_path = 'envs/arm_urdfs/ur5/ur5_1.urdf'
            self.chain1 = pk.build_serial_chain_from_urdf(open(robot1_urdf_path).read(), f"ee_link").to(device=self.device)
            
        torch.manual_seed(seed)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions

        # uniformly sample domain and include coordinates where source is non-zero 
        coords = torch.zeros(self.numpoints, self.num_states, device=self.device).uniform_(-1, 1) * torch.pi
        
        if self.pretrain:
            # only sample in time around the initial condition
            time = torch.ones(self.numpoints, 1, device=self.device) * start_time
            coords = torch.cat((time, coords), dim=1)
        else:
            # slowly grow time values from start time
            # this currently assumes start_time = 0 and max time value is tMax
            time = self.tMin + torch.zeros(self.numpoints, 1, device=self.device).uniform_(0, (self.tMax-self.tMin) * (self.counter / self.full_count))
            coords = torch.cat((time, coords), dim=1)

            # make sure we always have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time
        
        if self.use_symmetry:
            coords = coords[torch.logical_and(coords[..., 7] >= -torch.pi / 2, coords[..., 7] <= torch.pi / 2)]
            
        with torch.no_grad():
            states = coords[..., 1:]
            boundary_values = self.bc_model(states)
            
        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords[:, 0, None] == start_time)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False
        model_input = {'coords': coords}

        return model_input, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}
    

class ReachabilityGoalReachingConstrainedParticle(Dataset):
    def __init__(self, numpoints, 
        goalR=0.005, velocity=1.0, pretrain=False, tMin=0.0, tMax=1.6, counter_start=0, counter_end=100e3, 
        pretrain_iters=2000, num_src_samples=1000, seed=0, bound=1.0, constarint_circleR=0.5, constarint_tolerance=0.01, validation=False,
        sample_only_on_manifold=True):
        super().__init__()
        torch.manual_seed(0)

        self.pretrain = pretrain
        self.numpoints = numpoints
        
        self.max_velocity_u = velocity
        self.goalR = goalR
        self.goal = torch.tensor([constarint_circleR, 0.0])
        self.num_states = 2
        self.constarint_circleR = constarint_circleR
        self.constarint_tolerance = constarint_tolerance
        self.bound = bound

        self.tMax = tMax
        self.tMin = tMin

        self.N_src_samples = num_src_samples

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end 

        self.validation = validation
        self.sample_only_on_manifold = sample_only_on_manifold

        # Set the seed
        torch.manual_seed(seed)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions

        # sample uniformly in the box
        coords = torch.zeros(self.numpoints, self.num_states).uniform_(-self.bound, self.bound)

        if self.pretrain:
            # only sample in time around the initial condition
            time = torch.ones(self.numpoints, 1) * start_time
            coords = torch.cat((time, coords), dim=1)
        else:
            # slowly grow time values from start time
            # this currently assumes start_time = 0 and max time value is tMax
            time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin) * (self.counter / self.full_count))
            coords = torch.cat((time, coords), dim=1)

            # make sure we always have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time

        # set up the initial value function
        boundary_values = torch.norm(coords[..., 1:3] - self.goal, dim=-1, keepdim=True) - self.goalR        
        
        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:  
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords[:, 0, None] == start_time)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False
        model_input = {'coords': coords}

        if not self.validation:
            return model_input, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}
        else:
            return model_input, boundary_values
        


class ReachabilityUR5ObjectSource(Dataset):
    def __init__(self, numpoints, 
        collisionR=0.0, max_joint_velocity=1.0, num_links=12,
        bc_model=None, device='cpu', pretrain=False, tMin=0.0, tMax=0.6, counter_start=0, counter_end=100e3, 
        pretrain_iters=2000, num_src_samples=1000, seed=0, use_symmetry=False, manifold_constrained=False, manifold_tolerance=0.03):
        super().__init__()
        torch.manual_seed(0)
        
        self.bc_model = bc_model.to(device)
        self.device = device

        self.pretrain = pretrain
        self.numpoints = numpoints
        
        self.max_joint_velocity = max_joint_velocity
        self.collisionR = collisionR

        self.num_states = num_links
        self.num_links = num_links

        self.tMax = tMax
        self.tMin = tMin

        self.N_src_samples = num_src_samples

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end 

        self.use_symmetry = use_symmetry
        self.manifold_constrained = manifold_constrained
        self.manifold_tolerance = manifold_tolerance
        if self.manifold_constrained:
            robot1_urdf_path = 'envs/arm_urdfs/ur5/ur5_1.urdf'
            self.chain1 = pk.build_serial_chain_from_urdf(open(robot1_urdf_path).read(), f"ee_link").to(device=self.device)
            
        torch.manual_seed(seed)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions

        # uniformly sample domain and include coordinates where source is non-zero 
        coords = torch.zeros(self.numpoints, self.num_states, device=self.device).uniform_(-1, 1) * torch.pi
        
        if self.pretrain:
            # only sample in time around the initial condition
            time = torch.ones(self.numpoints, 1, device=self.device) * start_time
            coords = torch.cat((time, coords), dim=1)
        else:
            # slowly grow time values from start time
            # this currently assumes start_time = 0 and max time value is tMax
            time = self.tMin + torch.zeros(self.numpoints, 1, device=self.device).uniform_(0, (self.tMax-self.tMin) * (self.counter / self.full_count))
            coords = torch.cat((time, coords), dim=1)

            # make sure we always have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time
        
        if self.use_symmetry:
            coords = coords[torch.logical_and(coords[..., 7] >= -torch.pi / 2, coords[..., 7] <= torch.pi / 2)]
            
        with torch.no_grad():
            states = coords[..., 1:]
            boundary_values = self.bc_model(states)
            
        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords[:, 0, None] == start_time)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False
        model_input = {'coords': coords}

        return model_input, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}

    
if __name__ == '__main__':
    pass