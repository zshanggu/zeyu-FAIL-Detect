import torch
import torch.nn as nn
import numpy as np

from .layers import DenseInverseWishart
import sys
sys.path.append('../../')
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

class concatenated(nn.Module):
	def __init__(self, input_dim, full_in_dim, output_dim):
		super(concatenated, self).__init__()
		self.unet = ConditionalUnet1D(
			input_dim=input_dim,
			local_cond_dim=None,
			global_cond_dim=None,
			diffusion_step_embed_dim=128,
			down_dims=[256, 512, 1024],
			kernel_size=5,
			n_groups=8,
			cond_predict_scale=True
		)
		self.DIW = DenseInverseWishart(in_features=full_in_dim, p=output_dim)

	def forward(self, x): 
		x = self.unet(x, 0)
		x = x.reshape(x.shape[0], -1)
		x = self.DIW(x)
		return x

class MultivariateDerNet(nn.Module):
	def __init__(self, input_dim, full_in_dim, output_dim):
		super(MultivariateDerNet, self).__init__()
		self.p = output_dim

		self.hidden = concatenated(input_dim, full_in_dim, output_dim)
		self.apply(self.init_weights)

	def forward(self, x):
		mu, nu, kappa, L = self.hidden(x)

		return mu, nu, kappa, L

	def init_weights(self, m):
		if isinstance(m, nn.Linear):
			torch.nn.init.xavier_uniform_(m.weight)

	def get_prediction(self, x):
		self.eval()

		mu, nu, kappa, L = self.hidden(x)

		mu = mu.detach().cpu().numpy().squeeze()
		nu = nu.detach().cpu().numpy().squeeze(axis=1)
		kappa = kappa.detach().cpu().numpy().squeeze()
		L = L.detach().cpu().numpy()

		sum_of_pairwise_deviation_products = np.einsum('bik, bkl -> bil', L, np.transpose(L, (0, -1, -2)))
		aleatoric = np.reciprocal(nu[:, None, None] - self.p - 1 + 1e-8) * sum_of_pairwise_deviation_products
		epistemic = np.reciprocal(nu[:, None, None] + 1e-8) * aleatoric
		meta_aleatoric = np.zeros_like(aleatoric)
		for i, j in zip(range(self.p), range(self.p)):
			meta_aleatoric[:, i, j] = (nu - self.p + 1) * aleatoric[:, i, j] + (nu - self.p - 1) * aleatoric[:, i, i] * aleatoric[:, j, j]
			meta_aleatoric[:, i, j] /= (nu - self.p) * (nu - self.p - 1)**2 * (nu - self.p - 3)

		return mu, aleatoric, epistemic, meta_aleatoric, {"nu": nu, "kappa": kappa, "L": L}

