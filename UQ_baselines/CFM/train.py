import tqdm
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
master_dir = '../../UQ_baselines'
sys.path.append(master_dir)
import net as Net
import data_loader
from argparse import ArgumentParser
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    
parser = ArgumentParser()
parser.add_argument("--type", default='square', type=str)
parser.add_argument("--policy_type", default='flow', type=str)
args = parser.parse_args()

if __name__ == "__main__":
    type = args.type
    X, Y = data_loader.get_data(type=type, adjust_shape=False, diffusion=args.policy_type == 'diffusion')
    global_cond_dim = X.shape[1]; input_dim = Y.reshape(Y.shape[0], 16, -1).shape[-1]
    print(f'Current feature shape: {global_cond_dim}')
    train_data = torch.utils.data.TensorDataset(X)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    ckpt_file = f'{type}_{args.policy_type}.ckpt'
    
    # choice of model/method
    net = Net.get_unet(input_dim).to(device)
    EPOCHS = 200
    optimizer = torch.optim.Adam(net.parameters(), lr = 1e-4)
    if os.path.exists(ckpt_file):
        ckpt = torch.load(ckpt_file)
        net.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        starting_epochs = ckpt['epoch']
        losses = ckpt['losses']
        losses_FM = ckpt['losses_FM']
        losses_f = ckpt['losses_f']
        log_pO = ckpt['log_pO']
    else:
        starting_epochs = 0
        losses = []; losses_FM = []; losses_f = []
        log_pO = []

    t = tqdm.trange(starting_epochs, EPOCHS)
    for i in t:
        # Save checkpoint before to avoid nan loss
        ckpt = {
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': i+1,
            'losses': losses,
            'losses_FM': losses_FM,
            'losses_f': losses_f,
            'log_pO': log_pO
        }
        torch.save(ckpt, ckpt_file)

        net.train()
        loss_i = []; loss_FM_i = []; loss_f_i = []
        for (x_batch, ) in tqdm.tqdm(train_loader, desc='Training Batches'):
            observation = x_batch.to(device)
            optimizer.zero_grad()
            # x0, x1 = torch.randn_like(observation).to(device), observation
            x0, x1 = observation, torch.randn_like(observation).to(device)
            vtrue = x1 - x0
            cont_t = torch.rand(len(x1),).to(device)
            cont_t = cont_t.view(-1, *[1 for _ in range(len(observation.shape)-1)])
            xnow = x0 + cont_t * vtrue
            vhat = net(xnow, cont_t)
            loss_FM = (vhat - vtrue).pow(2).mean()
            ### Consistency loss
            delta = 5e-3
            cont_t_delta = torch.clamp(cont_t + delta, 0, 1)
            xnow_delta = x0 + cont_t_delta * vtrue
            with torch.no_grad():
                vhat_delta = net(xnow_delta, cont_t_delta)
            ft = xnow + (1 - cont_t) * vhat
            ft_delta = xnow_delta + (1 - cont_t_delta) * vhat_delta
            loss_f = (ft_delta - ft).pow(2).mean()
            beta = 1e-2 * loss_FM.item() / loss_f.item()
            loss = loss_FM + beta * loss_f
            ###
            loss.backward()
            # Terminate if loss is NaN
            if torch.isnan(loss):
                print("Loss is NaN")
                raise ValueError(f"NaN loss at epoch {i}")
            loss_i += [loss.item()]; loss_FM_i += [loss_FM.item()]; loss_f_i += [beta * loss_f.item()]
            optimizer.step()
        """ Validation
        """
        with torch.no_grad():
            X_sub = X[torch.randperm(len(X))[:1000]]
            logp_train = Net.get_logp(net, X_sub.to(device))
            log_pO += [logp_train.mean().item()]
            print(f"Epoch {i}, Train logp: {logp_train.mean().item():.2f}")
        losses += [np.mean(loss_i)]; losses_FM += [np.mean(loss_FM_i)]; losses_f += [np.mean(loss_f_i)]
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        ax[0, 0].set_title(r"FM loss")
        ax[0, 0].plot(losses_FM)
        ax[0, 1].set_title(r"Consistency loss")
        ax[0, 1].plot(losses_f)
        ax[1, 0].set_title(r"Summed Training loss")
        ax[1, 0].plot(losses)
        ax[1, 1].set_title(r"Train logp")
        ax[1, 1].plot(log_pO)
        suffix = type
        os.makedirs('images', exist_ok=True)
        plt.savefig(f"images/training_loss_{suffix}.png")
        plt.close('all')