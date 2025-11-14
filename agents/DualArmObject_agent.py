import torch
import numpy as np
from torch.autograd import grad
import pytorch_kinematics as pk
import os, sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import cyipopt

DEBUG = False

def wrap_joint(configs):
    return (configs + torch.pi) % (2 * torch.pi) - torch.pi

class DualArmObjectAgent():
    def __init__(self, 
                 max_joint_velocity=0.5,
                 num_links=6,
                 val_func_model=None,
                 verbose=False,
                 device='cpu') -> None:
        self.device = device
        self.max_joint_velocity = max_joint_velocity
        self.model = val_func_model
        self.state_dim = num_links
        self.verbose = verbose
        
        robot1_urdf_path = 'envs/arm_urdfs/ur5_object/ur5_1.urdf'
        self.chain1 = pk.build_serial_chain_from_urdf(open(robot1_urdf_path).read(), f"ee_link").to(device=self.device)
        robot2_urdf_path = 'envs/arm_urdfs/ur5_object/ur5_2.urdf'
        self.chain2 = pk.build_serial_chain_from_urdf(open(robot2_urdf_path).read(), f"ee_link").to(device=self.device)
        self.num_success_solutions = 0
        self.num_failed_solutions = 0
     

    def plan(self, agent_state, other_agent_state, goal_state, step_time=0.1, safe_time=0.3, buffer=0.0, planner_mode='hji', arm1=True, manifold_tolerance=0.005, only_self_constrained=False):
        use_simple_planner = (planner_mode == 'simple') or (planner_mode == 'very_simple') # whether to consider collision with other agent
        # manifold constraint: orientation constraint
        self.arm1 = arm1
        num_constarints = int(not use_simple_planner) + 1 + 3 # 1 for collision avoidance, 1 for action constraint, 3 for manifold constraint
        if not isinstance(agent_state, torch.Tensor):
            agent_state = torch.tensor(agent_state, device=self.device)
        if not isinstance(other_agent_state, torch.Tensor):
            other_agent_state = torch.tensor(other_agent_state, device=self.device)
        if not isinstance(goal_state, torch.Tensor):
            goal_state = torch.tensor(goal_state, device=self.device)
            
        # update adversary positions such that each agent will act as if they are adversary
        if (not use_simple_planner):
            adversary_positions = other_agent_state
            adversary_predicted_actions = self.predict_adversary_plans(self_state=agent_state, 
                                                                       adversary_agent_state=other_agent_state,
                                                                       only_self_constrained=only_self_constrained)
            adversary_positions += adversary_predicted_actions.flatten() * step_time
            adversary_positions = wrap_joint(adversary_positions)
        else:
            adversary_positions = None
        nlp_obj = NNArm_NLP(
            num_links=self.state_dim,
            start_position=agent_state,
            goal_position=goal_state,
            max_joint_velocity=self.max_joint_velocity,
            step_time=step_time,
            safe_time=safe_time,
            val_func_model=self.model,
            device=self.device,
            adversary_positions=adversary_positions,
            fk_chain=self.chain1 if arm1 else self.chain2,
            use_simple_planner=use_simple_planner,
            arm1=arm1
        )

        nlp = cyipopt.Problem(
            n=self.state_dim,
            m=num_constarints,
            problem_obj=nlp_obj,
            lb=[-self.max_joint_velocity]*self.state_dim,
            ub=[self.max_joint_velocity]*self.state_dim,
            cl=([buffer] if not use_simple_planner else []) +  [-1e20] + [-manifold_tolerance] * 3,
            cu=([1e20] if not use_simple_planner else []) +  [self.max_joint_velocity ** 2] + [manifold_tolerance] * 3, 
        )
        # manifold constraints: [1] z-orientation, [2] puzzle height, [3] puzzle groove
        nlp.add_option('sb', 'yes')
        nlp.add_option('print_level', 0)
        nlp.add_option('tol', 1e-3)
        nlp.add_option('max_iter', 15)
  
        # if optimization fails (when problem infeasible or problem exceeds time limit)
        # use the most conservative action
        initial_guess = torch.zeros(self.state_dim)
        if self.model is not None:
            temp = self.model.compute_ee_gradient
            self.model.compute_ee_gradient = False
            k_opt, self.info = nlp.solve(initial_guess.flatten().cpu().numpy())
            self.model.compute_ee_gradient = temp
            if self.info['status'] == 0:
                self.num_success_solutions += 1
            else:
                self.num_failed_solutions += 1
        else:
            k_opt, self.info = nlp.solve(initial_guess.flatten().cpu().numpy())
            if self.info['status'] == 0:
                self.num_success_solutions += 1
            else:
                self.num_failed_solutions += 1
            
        if self.info['status'] == 0 and DEBUG:
            print(f"Succ to solve with result ={k_opt}, NLP_OBJ CONS: {nlp_obj.cons}")
        

        if (not use_simple_planner) and (self.info['status'] != 0): 
            # import pdb; pdb.set_trace()
            if DEBUG:
                print(f"**Failed to solve, NLP_OBJ CONS: {nlp_obj.cons}")
            # return torch.tensor(k_opt) * 0.
            return self.make_safe_plans(
                start_position=agent_state,
                step_time=step_time,
                arm1=arm1,
                goal=goal_state,
                manifold_tolerance=manifold_tolerance
            ).flatten()
        
        return torch.tensor(k_opt)
    
    def predict_adversary_plans(self, self_state, adversary_agent_state, time=0.1, only_self_constrained=False):
        coords = torch.zeros(1, self.state_dim * 2 + 1, device=self.device)
        coords[:,0] = time
        coords[:,1:1+self.state_dim] = self_state
        coords[:,1+self.state_dim:] = adversary_agent_state
        self.currenrt_state = coords.clone()
        self.currenrt_state_values, self.current_state_grads = self.predict_value_function(coords, compute_grad=True)
        adversary_plans = -self.current_state_grads[:,1+self.state_dim:] / torch.linalg.norm(self.current_state_grads[:,1+self.state_dim:], dim=-1, keepdim=True) * self.max_joint_velocity
        self.safe_plan = self.current_state_grads[:,1:1+self.state_dim] / torch.linalg.norm(self.current_state_grads[:,1:1+self.state_dim], dim=-1, keepdim=True) * self.max_joint_velocity
        if DEBUG:
            print(f"current values: {self.currenrt_state_values}")
        return adversary_plans

    
    def make_safe_plans(self, start_position=None, step_time=0.1, arm1=True, manifold_tolerance=0.01, goal=None):
        # pick the safeset action according to the value function
        num_constraints = 1 + 3
        safe_plan = self.current_state_grads[:,1:1+self.state_dim] / torch.linalg.norm(self.current_state_grads[:,1:1+self.state_dim], dim=-1, keepdim=True) * self.max_joint_velocity
        proposed_x = safe_plan.detach().flatten().cpu()

        nlp_obj = SafePlan_NLP(
            start_position=start_position,
            proposed_x=proposed_x,
            num_links=self.state_dim,
            step_time=step_time,
            device='cpu',
            fk_chain=self.chain1 if arm1 else self.chain2,
            arm1=arm1,
        )

        nlp = cyipopt.Problem(
            n=self.state_dim,
            m=num_constraints,
            problem_obj=nlp_obj,
            lb=[-self.max_joint_velocity]*self.state_dim,
            ub=[self.max_joint_velocity]*self.state_dim,
            cl=[-1e20] + [-manifold_tolerance] * 3,
            cu=[self.max_joint_velocity ** 2] + [manifold_tolerance] * 3, 
        )
        nlp.add_option('sb', 'yes')
        nlp.add_option('print_level', 0)
        nlp.add_option('tol', 5e-3)
        nlp.add_option('max_iter', 15)
  
        # if optimization fails (when problem infeasible or problem exceeds time limit)
        # actiavte the action such the HJI value function is maximized
        k_opt, self.info = nlp.solve(proposed_x)
        if (self.info['status'] != 0):
            if self.verbose:
                print(f"Solving safe plan also failed, returning stopping plan...")
            return proposed_x * 0.
                
        return torch.from_numpy(k_opt)
    
    def predict_value_function(self, coords=None, compute_grad=False):
        model_outputs = self.model({'coords': coords})
        gradient = None
        values, inputs = model_outputs['model_out'], model_outputs['model_in']
        if compute_grad:
            gradient = grad(values, inputs, grad_outputs=torch.ones_like(values))[0]
            symmetry_mask = model_outputs['symmetry_mask']
            if symmetry_mask is not None:
                gradient[symmetry_mask] *= -1      
            # map to constraint manifold
            projection_matrix1 = model_outputs['dee1dq1_projection_matrix']
            projection_matrix2 = model_outputs['dee2dq2_projection_matrix']
            if projection_matrix1 is not None and projection_matrix2 is not None:
                projection_matrix1 = projection_matrix1.squeeze(0)
                gradient[..., 1:1+self.state_dim:] = (projection_matrix1 @ gradient[..., 1:7].unsqueeze(-1)).squeeze(-1)
                projection_matrix2 = projection_matrix2.squeeze(0)
                gradient[..., 1+self.state_dim:] = (projection_matrix2 @ gradient[..., 7:].unsqueeze(-1)).squeeze(-1)
        
        return values, gradient
    
    
class NNArm_NLP:
    def __init__(self, 
                 start_position=None,
                 goal_position=None,
                 num_links=6,
                 max_joint_velocity=1.0,
                 step_time=0.1,
                 safe_time=0.3,
                 val_func_model=None,
                 device='cpu',
                 adversary_positions=None,
                 fk_chain=None,
                 use_simple_planner=False,
                 use_very_simple_planner=False,
                 arm1=True,
                ):
        if isinstance(start_position, torch.Tensor):
            self.state = start_position.to(device=device, dtype=torch.float32)
        else:
            self.state = torch.tensor(start_position, device=device, dtype=torch.float32)
        if isinstance(goal_position, torch.Tensor):
            self.goal_state = goal_position.to(device=device, dtype=torch.float32)
        else:
            self.goal_state = torch.tensor(goal_position, device=device, dtype=torch.float32)
        if isinstance(adversary_positions, torch.Tensor):
            self.adversary_states = adversary_positions.to(device=device, dtype=torch.float32)
        elif adversary_positions is not None:
            self.adversary_states = torch.tensor(adversary_positions).to(device=device, dtype=torch.float32)
        
        self.num_links = num_links
        self.np_state = self.state.cpu().numpy()
        self.np_adversary_states = self.adversary_states.cpu().numpy() if adversary_positions is not None else None
        self.np_goal = self.goal_state.cpu().numpy()

        self.num_var = num_links
        self.max_joint_velocity = max_joint_velocity
        self.step_time = step_time
        self.safe_time = safe_time
        self.model = val_func_model
        self.device = device
        self.prev_x = np.zeros_like(self.np_state) * np.nan
        
        self.use_simple_planner = int(use_simple_planner)
        self.use_very_simple_planner = int(use_very_simple_planner)
        self.arm1 = arm1
        self.num_constraints = (1 - self.use_simple_planner) + 1 + 3 # 1 for collision avoidance, 1 for action constraint, 3 for manifold constraint
        self.chain = fk_chain.to(device=self.device)
        return 
    
    def objective(self, x):
        if np.all(self.np_goal == 0.):
            return 0.
        new_state = self.np_state + self.step_time * x
        return np.sum(np.square((new_state - self.np_goal)))

    def gradient(self, x):
        if np.all(self.np_goal == 0.):
            return np.zeros_like(x)
        new_state = self.np_state + self.step_time * x
        return 2 * self.step_time * (new_state - self.np_goal)

    def constraints(self, x):
        self.compute_constraints_jacobian(x)
        return self.cons

    def jacobian(self, x):
        self.compute_constraints_jacobian(x)
        return self.jac
    
    def compute_constraints_jacobian(self, x):
        if (self.prev_x != x).any():
            self.cons = torch.zeros(self.num_constraints, device=self.device)
            self.jac = torch.zeros((self.num_constraints, self.num_var), device=self.device)

            # constraint with other agents
            if (not self.use_simple_planner) and (not self.use_very_simple_planner):
                coords = torch.zeros(1, self.num_links * 2 + 1, device=self.device)
                coords[:,0] = self.safe_time
                coords[:,1:1+self.num_links] = self.state + torch.tensor(x, device=self.device) * self.step_time
                coords[:,1+self.num_links:] = self.adversary_states
                model_outputs = self.model({'coords': coords})
                values, inputs = model_outputs['model_out'], model_outputs['model_in']
                gradient = grad(values, inputs, grad_outputs=torch.ones_like(values))[0][:,1:1+self.num_links] * self.step_time
                
                symmetry_mask = model_outputs['symmetry_mask']
                if symmetry_mask is not None:
                    gradient[symmetry_mask] *= -1
                self.cons[0] = values.view(1)
                self.jac[0] = gradient
                
            # action constraint
            tensor_x = torch.tensor(x, device=self.device, requires_grad=True, dtype=torch.float32)
            self.cons[1-self.use_simple_planner] = torch.sum(torch.square(tensor_x))
            self.jac[1-self.use_simple_planner] = 2 * tensor_x
            
            # manifold constraint
            next_state = self.state + tensor_x * self.step_time
            fk = self.chain.forward_kinematics(next_state, end_only=True)
            # 1. orientation constraint
            R = fk.get_matrix()[0, :3, :3]
            R_d = torch.tensor([
                    [ 0., -1.,  0.],
                    [ 0.,  0.,  1.],
                    [-1.,  0.,  0.]
                ], device=self.device, dtype=torch.float32)
            M = (R_d.transpose(-1, -2) @ R - R.transpose(-1, -2) @ R_d) / 2
            e = torch.stack([M[2,1], M[0,2], M[1,0]], dim=-1).flatten()
                
            for i in range(3):    
                self.cons[i+2-self.use_simple_planner:i+2-self.use_simple_planner+1] = e[i]
                self.jac[i+2-self.use_simple_planner:i+2-self.use_simple_planner+1] = grad(e[i], tensor_x, retain_graph=True)[0]
                                    
            self.cons = self.cons.detach().cpu().numpy()
            self.jac = self.jac.detach().cpu().numpy()
            self.prev_x = x.copy()
        return
    

class SafePlan_NLP:
    def __init__(self, 
                 start_position=None,
                 proposed_x=None,
                 num_links=6,
                 step_time=0.1,
                 device='cpu',
                 fk_chain=None,
                 arm1=True,
                ):
        if isinstance(start_position, torch.Tensor):
            self.state = start_position.to(device=device, dtype=torch.float32)
        else:
            self.state = torch.tensor(start_position, device=device, dtype=torch.float32)
        self.num_var = self.num_links = num_links
        self.step_time = step_time
        self.device = device
        self.np_proposed_x = proposed_x.detach().cpu().numpy()
        self.prev_x = np.zeros_like(self.np_proposed_x) * np.nan
        
        self.num_constraints = 1 + 3
        self.chain = fk_chain.to(device=device)
        self.arm1 = arm1
        return 
    
    def objective(self, x):
        return -np.dot(x, self.np_proposed_x)

    def gradient(self, x):
        return -self.np_proposed_x

    def constraints(self, x):
        self.compute_constraints_jacobian(x)
        return self.cons

    def jacobian(self, x):
        self.compute_constraints_jacobian(x)
        return self.jac
    
    def compute_constraints_jacobian(self, x):
        if (self.prev_x != x).any():
            self.cons = torch.zeros(self.num_constraints, device=self.device)
            self.jac = torch.zeros((self.num_constraints, self.num_var), device=self.device)
            
            # action constraint
            tensor_x = torch.tensor(x, device=self.device, requires_grad=True, dtype=torch.float32)
            self.cons[0] = torch.sum(torch.square(tensor_x))
            self.jac[0] = 2 * tensor_x
            
            # manifold constraint
            next_state = self.state + tensor_x * self.step_time
            fk = self.chain.forward_kinematics(next_state, end_only=True)
            # 1. orientation constraint
            R = fk.get_matrix()[0, :3, :3]
            R_d = torch.tensor([
                    [ 0., -1.,  0.],
                    [ 0.,  0.,  1.],
                    [-1.,  0.,  0.]
                ], device=self.device, dtype=torch.float32)
            M = (R_d.transpose(-1, -2) @ R - R.transpose(-1, -2) @ R_d) / 2
            e = torch.stack([M[2,1], M[0,2], M[1,0]], dim=-1).flatten()
                
            for i in range(3):    
                self.cons[i+1:i+2] = e[i]
                self.jac[i+1:i+2] = grad(e[i], tensor_x, retain_graph=True)[0]

            self.cons = self.cons.detach().cpu().numpy()
            self.jac = self.jac.detach().cpu().numpy()
            self.prev_x = x.copy()
        return