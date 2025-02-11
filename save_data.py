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

OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    cfg['_target_'] = 'diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace_get_data.TrainDiffusionUnetHybridWorkspace'
    cfg['task']['env_runner']['_target_'] = 'diffusion_policy.env_runner.robomimic_image_runner_FAIL-Detect.RobomimicImageRunner'
    
    # Access the custom policy_type parameter
    policy_type = 'flow' if 'flow' in cfg['policy']['_target_'] else 'diffusion'
    if policy_type == 'flow':
        cfg['policy']['_target_'] = 'diffusion_policy.policy.flow_unet_hybrid_image_policy.DiffusionUnetHybridImagePolicy'
    elif policy_type == 'diffusion':
        cfg['policy']['_target_'] = 'diffusion_policy.policy.diffusion_unet_hybrid_image_policy.DiffusionUnetHybridImagePolicy' 
    else:
        raise ValueError(f"Invalid policy type: {policy_type}")
    ## Modify ckpt and logging file
    log_name = cfg['logging']['name']
    task_map = {
        'square': 'square_data',
        'transport': 'transport_data',
        'tool_hang': 'tool_hang_data',
        'can': 'can_data'
    }
    for task, file_name in task_map.items():
        if task in log_name:
            suffix = '_abs' if task == 'tool_hang' else ''
            cfg['logging'] = f'data/outputs/{file_name}_{policy_type}.pt'
            cfg['checkpoint'] = f'data/outputs/train_{policy_type}_unet_visual_{task}_image{suffix}/checkpoints/latest.ckpt'
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