import tqdm
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
master_dir = '../../UQ_baselines'
sys.path.append(master_dir)
import data_loader
from net import RNDPolicy
from argparse import ArgumentParser
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

parser = ArgumentParser()
parser.add_argument("--type", default='square', type=str)
parser.add_argument("--policy_type", default='flow', type=str)
args = parser.parse_args()
if __name__ == "__main__":
    type = args.type
    X, Y = data_loader.get_data(type=type, adjust_shape=False, diffusion=args.policy_type == 'diffusion')
    Y = Y.reshape(Y.shape[0], 16, -1)
    input_dim = Y.shape[-1]; global_cond_dim = X.shape[1]
    train_data = torch.utils.data.TensorDataset(X, Y)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    ckpt_file = f'{type}_{args.policy_type}.ckpt'
    EPOCHS = 200

    # choice of model/method
    net = RNDPolicy(input_dim, global_cond_dim).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    if os.path.exists(ckpt_file):
        ckpt = torch.load(ckpt_file)
        net.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
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
            'epoch': i+1,
            'losses': losses
        }
        torch.save(ckpt, ckpt_file)

        net.train()
        loss_i = []
        for (x_batch, y_batch) in tqdm.tqdm(train_loader, desc='Training Batches'):
            observation = x_batch.to(device)
            action = y_batch.to(device)

            optimizer.zero_grad()
            loss = net(action, observation).mean()
            loss.backward()
            # Terminate if loss is NaN
            if torch.isnan(loss):
                print("Loss is NaN")
                raise ValueError(f"NaN loss at epoch {i}")
            loss_i += [loss.item()]
            optimizer.step()
        """ Validation
        """
        t.set_description(f"val. loss: {loss.detach().cpu().numpy():.2f}")
        t.refresh()
        losses += [np.mean(loss_i)]

        plt.title(r"Training loss")
        plt.plot(losses)
        plt.show()
        suffix = type
        os.makedirs('images', exist_ok=True)
        plt.savefig(f"images/training_loss_{suffix}.png")
        plt.clf()