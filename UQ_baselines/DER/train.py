import tqdm
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from evidential_regression.networks_chen import MultivariateDerNet
from evidential_regression.losses import MultivariateEvidentialRegressionLoss
import sys
master_dir = '../../UQ_baselines'
sys.path.append(master_dir)
import data_loader
from argparse import ArgumentParser
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")


parser = ArgumentParser()
parser.add_argument("--type", default='square', type=str)
parser.add_argument("--policy_type", default='flow', type=str)
args = parser.parse_args()
if __name__ == "__main__":
	type = args.type
	X, Y = data_loader.get_data(type=type, diffusion=args.policy_type == 'diffusion')
	input_dim = X.shape[-1]; full_in_dim = X.shape[1] * X.shape[-1]; output_dim = Y.shape[-1]
	train_data = torch.utils.data.TensorDataset(X, Y)
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
	ckpt_file = f'{type}_{args.policy_type}.ckpt'
	EPOCHS = 200
	optimizer_params = {
		"lr": 1e-04,
		"betas": (0.9, 0.999),
		"eps": 1e-8,
		"weight_decay": 1e-2,
		"amsgrad": False}

	# choice of model/method
	net = MultivariateDerNet(input_dim, full_in_dim, output_dim).to(device)
	criterion = MultivariateEvidentialRegressionLoss()

	optimizer = torch.optim.AdamW(net.parameters(), **optimizer_params)
	scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=optimizer_params["lr"], steps_per_epoch=len(train_loader), epochs=EPOCHS)

	if os.path.exists(ckpt_file):
		ckpt = torch.load(ckpt_file)
		net.load_state_dict(ckpt['model'])
		optimizer.load_state_dict(ckpt['optimizer'])
		scheduler.load_state_dict(ckpt['scheduler'])
		starting_epochs = ckpt['epoch']
		losses = ckpt['losses']
	else:
		starting_epochs = 0
		losses = []
	
	t = tqdm.trange(starting_epochs, EPOCHS)
	for i in t:
     	# Save checkpoint before to avoid nan loss
		ckpt = {
			'model': net.state_dict(),
			'optimizer': optimizer.state_dict(),
			'scheduler': scheduler.state_dict(),
			'epoch': i+1,
			'losses': losses
		}
		torch.save(ckpt, ckpt_file)
  
		net.train()
		loss_i = []
		for (x_batch, y_batch) in tqdm.tqdm(train_loader, desc='Training Batches'):
			inputs = x_batch.to(device)
			labels = y_batch.to(device)

			optimizer.zero_grad()
			outs = net(inputs)
			loss = criterion(labels, *outs)

			loss.backward()
			# Terminate if loss is NaN
			if torch.isnan(loss):
				print("Loss is NaN")
				raise ValueError(f"NaN loss at epoch {i}")
			loss_i += [loss.item()]
			optimizer.step()
			scheduler.step()
		""" Validation
		"""
		net.eval()
		mu, aleatoric, epistemic, meta_aleatoric, output_params = net.get_prediction(inputs)
		t.set_description(f"val. loss: {loss.detach().cpu().numpy():.2f}")
		t.refresh()
		losses += [np.mean(loss_i)]

		plt.title(r"NLL loss")
		plt.plot(losses)
		# plt.show()
		suffix = type
		os.makedirs('images', exist_ok=True)
		plt.savefig(f"images/training_loss_{suffix}.png")
		plt.clf()