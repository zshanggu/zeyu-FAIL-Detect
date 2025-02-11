# Save as "save_data.py"
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)
import torch
import dill
import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--ckpt_path', type=str, help='Path to the checkpoint')
parser.add_argument('--policy_type', type=str, help='Type of policy')
args = parser.parse_args()

OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    cfg['_target_'] = 'diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace_get_data.TrainDiffusionUnetHybridWorkspace'
    cfg['task']['env_runner']['_target_'] = 'diffusion_policy.env_runner.robomimic_image_runner_logp_together.RobomimicImageRunner'
    if args.policy_type == 'flow':
        cfg['policy']['_target_'] = 'diffusion_policy.policy.flow_unet_hybrid_image_policy.DiffusionUnetHybridImagePolicy'
    elif args.policy_type == 'diffusion':
        cfg['policy']['_target_'] = 'diffusion_policy.policy.diffusion_unet_hybrid_image_policy.DiffusionUnetHybridImagePolicy' 
    else:
        raise ValueError(f"Invalid policy type: {args.policy_type}")
    ## Modify ckpt and logging file
    log_name = cfg['logging']['name']
    cfg['checkpoint'] = f'data/outputs/{args.ckpt_path}'
    task_map = {
        'square': 'square_data.pt',
        'transport': 'transport_data.pt',
        'tool_hang': 'tool_hang_data.pt',
        'can': 'can_data.pt'
    }
    for task, file_name in task_map.items():
        if task in log_name:
            cfg['logging'] = f'data/outputs/{file_name}'
            break
    ## End of modification
    cfg['dataloader']['shuffle'] = False
    OmegaConf.resolve(cfg)
    if 'output_file' in cfg:
        cfg['output_file'] = cfg.output_file
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    payload = torch.load(open(cfg['checkpoint'], 'rb'), pickle_module=dill)
    workspace.logging = cfg['logging']
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    workspace.run()

if __name__ == "__main__":
    main()