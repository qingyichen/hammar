import torch
import diff_operators

def initialize_hji_constrainedUR5(dataset, minWith):
    # Initialize the loss function for the NN two realistic arms problem
    # The dynamics parameters
    max_joint_velocity = dataset.max_joint_velocity
    num_links = dataset.num_links

    def hji_ur5(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']  # (meta_batch_size, num_points, 12)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']
        
        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            du, status = diff_operators.jacobian(y, x)
            dudt = du[..., 0, 0]
            dudx = du[..., 0, 1:]
            dudx1 = dudx[..., :num_links // 2]
            dudx2 = dudx[..., num_links // 2:]

            # norm-based action constraints
            # Compute the constrained hamiltonian
            projection_matrix1 = model_output['dee1dq1_projection_matrix']
            projected_gradient1 = (projection_matrix1 @ dudx1.unsqueeze(-1)).squeeze(-1)
            projection_matrix2 = model_output['dee2dq2_projection_matrix']
            projected_gradient2 = (projection_matrix2 @ dudx2.unsqueeze(-1)).squeeze(-1)
            ham = (torch.linalg.norm(projected_gradient1, dim=-1) - torch.linalg.norm(projected_gradient2, dim=-1)) * max_joint_velocity

            # If we are computing BRT then take min with zero
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)
            
            diff_constraint_hom = dudt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.max(diff_constraint_hom[:, :, None], y - source_boundary_values)

        dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]

        # A factor of 15e2 to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet) * 40,
                'diff_constraint_hom': torch.abs(diff_constraint_hom)}
    
    return hji_ur5


def initialize_hjb_GoalReachingConstarinedParticle(dataset, consider_constraint_manifold=True):
    max_velocity_u = dataset.max_velocity_u
    constarint_circleR = dataset.constarint_circleR
    constarint_tolerance = dataset.constarint_tolerance
    
    def hjb_particle_loss(model_output, gt, validation=False):
        return hjb_particle(model_output, gt)

    def hjb_particle(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']  # (meta_batch_size, num_points, 3)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            du, status = diff_operators.jacobian(y, x)
            dudt = du[..., 0, 0]
            dudx = du[..., 0, 1:]            
            if consider_constraint_manifold:
                x_in = x[...,1:]
                dim = x_in.shape[-1]
                constraint_gradient = 2 * x_in
                c = constraint_gradient.unsqueeze(-1)
                c_T = torch.transpose(c, -1, -2)
                projection_matrix = torch.eye(dim, device=x_in.device).expand(x_in.shape[0], x_in.shape[1], dim, dim)
                projection_matrix = projection_matrix - c @ c_T / torch.sum(torch.square(constraint_gradient), dim=-1, keepdim=True).unsqueeze(-1)
                projected_gradient = (projection_matrix @ dudx.unsqueeze(-1)).squeeze(-1)
                ham = -torch.linalg.norm(projected_gradient, dim=-1) * max_velocity_u
            else:
                ham = -torch.linalg.norm(dudx, dim=-1) * max_velocity_u

            diff_constraint_hom = dudt - ham
            
        dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]

        losses = {'dirichlet': torch.abs(dirichlet) * 40,
                'diff_constraint_hom': torch.abs(diff_constraint_hom)}
        return losses

    return hjb_particle_loss