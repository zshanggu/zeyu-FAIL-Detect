if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.pytorch_util import dict_apply

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionUnetHybridWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model: DiffusionUnetHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        device = torch.device(cfg.training.device)
        self.ema_model.set_normalizer(normalizer)
        self.ema_model.to(device)

        # configure env
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        # configure logging

        full_x = []; full_y = []
        with torch.no_grad():
            for i, batch in enumerate(train_dataloader):
                # device transfer
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                
                mod = self.ema_model # See "policy" folder
                nobs = mod.normalizer.normalize(batch['obs'])
                nactions = mod.normalizer['action'].normalize(batch['action'])
                batch_size = nactions.shape[0]

                # handle different ways of passing observation
                trajectory = nactions.reshape(batch_size, -1)
                # Get latent representation of observations
                this_nobs = dict_apply(nobs, 
                    lambda x: x[:,:mod.n_obs_steps,...].reshape(-1,*x.shape[2:]))
                nobs_features = mod.obs_encoder(this_nobs)
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)

                print(f'At batch {i}/{len(train_dataloader)}')
                print(f'X: {global_cond.shape}, Y: {trajectory.shape}')
                full_x.append(global_cond.cpu()); full_y.append(trajectory.cpu())
        full_x = torch.cat(full_x, dim=0)
        full_y = torch.cat(full_y, dim=0)
        print(f'Full X: {full_x.shape}, Full Y: {full_y.shape}')
        torch.save({'X': full_x, 'Y': full_y}, self.logging)

                

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
