from natpn import suppress_pytorch_lightning_logs
suppress_pytorch_lightning_logs()
import pytorch_lightning as pl
pl.seed_everything(1103)
from argparse import ArgumentParser
import torch
import sys
sys.path.append('../../UQ_baselines')
import data_loader
from sklearn.cluster import KMeans
from natpn.datasets import FPDataModule
from net_natpn import get_net

parser = ArgumentParser()
parser.add_argument("--type", default='square', type=str)
parser.add_argument("--policy_type", default='flow', type=str)
args = parser.parse_args()
type = args.type

if __name__ == '__main__':
    X, _ = data_loader.get_data(type=type, adjust_shape=False, diffusion=args.policy_type == 'diffusion')
    # Do k-means on X and get Y as label
    ncluster = 64
    print(f'Total number of clusters: {ncluster}')
    kmeans = KMeans(n_clusters=ncluster, random_state=0).fit(X.cpu().numpy())
    Y = kmeans.labels_.astype(int)
    Y = torch.from_numpy(Y).to(X.device)
    dm = FPDataModule(X, Y)
    #################### Model
    estimator = get_net()
    # training
    estimator.fit(dm)
    # Evaluate
    # See "NatPN/natpn/model/lightning_module_ood.py"
    model = estimator.model_
    file = f'{type}_{args.policy_type}.ckpt'
    torch.save({'model': model.state_dict()}, file)
    # Just save this checkpoint
    posterior, log_prob = model.forward(X)
    print(X.shape, log_prob.shape)
    print(log_prob.mean(), log_prob.std())
