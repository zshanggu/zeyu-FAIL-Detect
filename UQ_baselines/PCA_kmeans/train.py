import torch
import sys
master_dir = '../../UQ_baselines'
sys.path.append(master_dir)
import data_loader
from argparse import ArgumentParser
from net_PCA import PCAKMeansNet
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    
parser = ArgumentParser()
parser.add_argument("--type", default='square', type=str)
parser.add_argument("--policy_type", default='flow', type=str)
args = parser.parse_args()

if __name__ == "__main__":
    type = args.type
    X, Y = data_loader.get_data(type=type, adjust_shape=False)
    # Initialize the network
    # (Chen) This would vary depending on data, but since we do not wish users to change this, we hard-code it here based on previous runs
    if args.policy_type == 'flow':
        emb_dim_dict = {'square': 55, 'transport': 120, 'tool_hang': 77, 'can': 56, 'lift': 50}
    else:
        emb_dim_dict = {'square': 46, 'transport': 65, 'tool_hang': 56, 'can': 39, 'lift': 47}
    emb_dim = emb_dim_dict[type]
    net = PCAKMeansNet(X, emb_dim=emb_dim).to(device)
    # Save the model state
    ckpt_file = f'{type}_{args.policy_type}.ckpt'
    ckpt = {
        'model': net.state_dict()
    }
    torch.save(ckpt, ckpt_file)
    