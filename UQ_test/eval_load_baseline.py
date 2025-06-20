import os
import torch
import sys
import numpy as np
import math
sys.path.append('..')
master_dir = '../UQ_baselines'
path_logpZO = os.path.join(master_dir, 'logpZO')
path_RND = os.path.join(master_dir, 'RND')
path_DER = os.path.join(master_dir, 'DER')
path_CFM = os.path.join(master_dir, 'CFM')
path_NatPN = os.path.join(master_dir, 'NatPN')
path_PCA_kmeans = os.path.join(master_dir, 'PCA_kmeans')
for p in [path_logpZO, path_RND, path_DER, path_CFM, path_NatPN, path_PCA_kmeans]:
    sys.path.append(p)

## Load models
def get_baseline_model(baseline_name, task_name, policy_type='flow'):
    if baseline_name == 'DER':
        from evidential_regression.networks_chen import MultivariateDerNet
        input_dim, full_in_dim, output_dim = 10, 280, 160
        if task_name == 'transport':            
            input_dim, full_in_dim, output_dim = 20, 560, 320
        elif task_name == 'tool_hang':
            input_dim, full_in_dim, output_dim = 20, 320, 160
        net = MultivariateDerNet(input_dim, full_in_dim, output_dim)
        path_now = path_DER
    if baseline_name == 'RND':
        from net import RNDPolicy
        input_dim, global_cond_dim = 10, 274                     
        if task_name == 'transport':
            input_dim, global_cond_dim = 20, 548
        net, path_now = RNDPolicy(input_dim, global_cond_dim), path_RND 
        net = net
    if baseline_name == 'CFM' or baseline_name == 'logpZO':
        import net_CFM as Net
        input_dim = 10
        if task_name == 'transport':
            input_dim = 20
        net = Net.get_unet(input_dim)
        if baseline_name == 'CFM':
            path_now = path_CFM
        else:
            path_now = path_logpZO
    if baseline_name == 'NatPN':
        from net_natpn import get_net
        net = get_net()
        input_size = 274 
        if task_name == 'transport':
            input_size = 548  
        input_size = torch.Size([input_size])
        # A bit annoying but I have to do this
        net = net._init_model(
            output_type = 'categorical',
            input_size = input_size,
            num_classes = 64
        )
        path_now = path_NatPN
    if baseline_name == 'PCA_kmeans':
        from net_PCA import PCAKMeansNet
        input_dim = 274; n_clusters = 64
        if policy_type == 'flow':
            emb_dim_dict = {'square': 55, 'transport': 120, 'tool_hang': 77, 'can': 56, 'lift': 50}
        else:
            emb_dim_dict = {'square': 46, 'transport': 65, 'tool_hang': 56, 'can': 39, 'lift': 47}
        emb_dim = emb_dim_dict[task_name]
        if task_name == 'transport':
            input_dim = 548
        net = PCAKMeansNet(input_dim=input_dim, emb_dim=emb_dim, n_clusters=n_clusters)
        path_now = path_PCA_kmeans
    ckpt = torch.load(os.path.join(path_logpZO, f'{task_name}_{policy_type}.ckpt'))
    net.load_state_dict(ckpt['model'])
    return net.eval()

## Get metrics
def adjust_xshape(x, in_dim):
    total_dim = x.shape[1]
    # Calculate the padding needed to make total_dim a multiple of in_dim
    remain_dim = total_dim % in_dim
    if remain_dim > 0:
        pad = in_dim - remain_dim
        total_dim += pad
        x = torch.cat([x, torch.zeros(x.shape[0], pad, device=x.device)], dim=1)
    # Calculate the padding needed to make (total_dim // in_dim) a multiple of 4
    reshaped_dim = total_dim // in_dim
    if reshaped_dim % 4 != 0:
        extra_pad = (4 - (reshaped_dim % 4)) * in_dim
        x = torch.cat([x, torch.zeros(x.shape[0], extra_pad, device=x.device)], dim=1)
    return x.reshape(x.shape[0], -1, in_dim)

# RND
def RND_UQ(baseline_model, action, observation):
    baseline_model.eval()
    with torch.no_grad():
        # Sum the differences over the temporal aspect
        return baseline_model(action, observation).sum(dim=1).cpu()
    
# DER
def DER_UQ(baseline_model, global_cond, task_name):
    baseline_model.eval()
    # Will just use the total variance from epistemic to do the epistemic UQ
    # Since we don't have the mu and actual variance
    with torch.no_grad():
        in_dim = 10
        if task_name == 'tool_hang':
            in_dim = 20
        if task_name == 'transport':
            in_dim = 20
        global_cond = adjust_xshape(global_cond, in_dim)
        _, _, epistemic, _, _ = baseline_model.get_prediction(global_cond)
    return torch.tensor([e.trace() for e in epistemic])

# CFM
def CFM_UQ(baseline_model, observation, task_name = 'square'):
    observation = observation
    in_dim = 10
    if task_name == 'transport':
        in_dim = 20
    observation = adjust_xshape(observation, in_dim)
    nstep = 5
    with torch.no_grad():
        timesteps = torch.linspace(1, 0, nstep+1, device=observation.device)[:-1]
        timesteps = (timesteps * 100).long() # Time-scale is 100 by default
        predicted_v = []
        for t in timesteps:
            pred_v = baseline_model(observation, t)
            predicted_v.append(pred_v.cpu().numpy())
            observation = observation - pred_v / nstep
        predicted_v = np.array(predicted_v).reshape(nstep, len(observation), -1)
    # Calculate the curvature
    curvature = np.std(predicted_v, axis=0).mean(axis=1)
    return torch.tensor(curvature)

def logpZO_UQ(baseline_model, observation, action_pred = None, task_name = 'square'):
    observation = observation
    in_dim = 10
    if task_name == 'transport':
        in_dim = 20
    observation = adjust_xshape(observation, in_dim)
    if action_pred is not None:
        action_pred = action_pred
        observation = torch.cat([observation, action_pred], dim=1)
    with torch.no_grad():
        timesteps = torch.zeros(observation.shape[0], device=observation.device)
        pred_v = baseline_model(observation, timesteps)
        observation = observation + pred_v
        logpZO = observation.reshape(len(observation), -1).pow(2).sum(dim=-1)
    return logpZO

def NatPN_UQ(baseline_model, observation):
    _, log_prob = baseline_model.forward(observation)
    return log_prob

def logpO_UQ(baseline_model, observation, task_name = 'square'):
    observation = observation
    in_dim = 10
    if task_name == 'transport':
        in_dim = 20
    observation = adjust_xshape(observation, in_dim)
    d = observation.reshape(observation.shape[0], -1).shape[1]
    sigma = 0.001/math.sqrt(d)
    if baseline_model.global_eps is None or baseline_model.global_eps.shape != observation.shape:
        baseline_model.global_eps = torch.randn(observation.shape, device=observation.device)
    with torch.no_grad():
        pred_v = baseline_model(observation, 0)
        pred_v_perturb = baseline_model(observation + sigma*baseline_model.global_eps, 0)
        e_dzdx = (pred_v_perturb - pred_v) / sigma
        e_dzdx_e = e_dzdx * baseline_model.global_eps
        approx_tr_dzdx = e_dzdx_e.reshape(observation.shape[0], -1).sum(dim=1)
        observation = observation + pred_v
        log_p_z = -0.5 * math.log(2 * math.pi) - 0.5 * observation**2
        log_p_z = log_p_z.reshape(observation.shape[0], -1).sum(dim=1)
    log_p_O = log_p_z + approx_tr_dzdx
    return log_p_O

def median_trick_bandwidth(x, y):
    # Combine x and y
    combined = torch.cat([x, y], dim=0)
    # Compute pairwise distances
    pairwise_dists = torch.cdist(combined, combined, p=2)
    # Get the upper triangle of the distance matrix, without the diagonal
    upper_tri_dists = pairwise_dists.triu(diagonal=1)
    # Flatten and remove zero distances (diagonal elements)
    upper_tri_dists = upper_tri_dists[upper_tri_dists > 0]
    # Compute the median of these distances
    bandwidth = upper_tri_dists.median()
    return 2*bandwidth.item()**2

def STAC_UQ(prev_actions, curr_actions):
    x = prev_actions; y = curr_actions
    if x is None:
        return torch.zeros(curr_actions.shape[0]).to(curr_actions.device)
    else:
        # Compare the MMD between the previous and current actions over batches (2nd dimension)
        N, B, a, b = x.shape
        # beta_1 = 1 / b # Following their Table 2 in appendix as 1/|A|
        # Flatten the last two dimensions for RBF kernel computation
        x_flat = x.reshape(N, B, a*b)
        y_flat = y.reshape(N, B, a*b) 
        # Compute pairwise RBF kernel
        def rbf_kernel(a, b, beta):
            pairwise_dists = torch.cdist(a, b, p=2) ** 2
            return torch.exp(-pairwise_dists / beta)       
        mmd_values = torch.zeros(N, device=x.device)
        for i in range(N):
            x_i = x_flat[i]
            y_i = y_flat[i]
            # use median heuristic for bandwidth
            beta_1 = median_trick_bandwidth(x_i, y_i)
            k_xx = rbf_kernel(x_i, x_i, beta_1).mean()
            k_yy = rbf_kernel(y_i, y_i, beta_1).mean()
            k_xy = rbf_kernel(x_i, y_i, beta_1).mean()
            mmd_values[i] = k_xx + k_yy - 2 * k_xy     
        return mmd_values

def PCA_kmeans_UQ(baseline_model, observation):
    return baseline_model(observation)