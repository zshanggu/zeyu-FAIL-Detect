"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""
import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)
import os
import pathlib
import click
import hydra
import torch
import numpy
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace
torch.manual_seed(1103); numpy.random.seed(1103)

@click.command()
@click.option('-c', '--ckpt_path', required=True)
@click.option('-p', '--policy_type', required=True, default='flow')
@click.option('-t', '--task_name', required=True, default='square')
@click.option('-s', '--num_inference_step', default=10)
@click.option('-d', '--device', default=0)
@click.option('-m', '--modify', default=False, help='Modify the environment to make it OOD')
@click.option('-n', '--num', default=100)


def main(ckpt_path, policy_type, task_name, num_inference_step, device, modify, num):
    device = 'cuda:' + str(device) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    modify_suffix = '_modify' if modify else ''
    suffix = '_abs' if task_name == 'tool_hang' else ''
    output_dir = os.path.join(f'data/outputs/train_{policy_type}_unet_visual_{task_name}_image{suffix}', 'final_eval', f'steps_{num_inference_step}_OOD{modify_suffix}')
    json_filename = f'eval_log_steps_{num_inference_step}.json'
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint (for policy generator)
    def get_policy(checkpoint):
        payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        from omegaconf import DictConfig
        def get_curr_shape(checkpoint):
            env_target = 'diffusion_policy.env_runner.robomimic_image_runner_FAIL-Detect.RobomimicImageRunner'
            if 'flow' in checkpoint:
                policy_target = 'diffusion_policy.policy.flow_unet_hybrid_image_policy_together.DiffusionUnetHybridImagePolicy'
            else:
                policy_target = 'diffusion_policy.policy.diffusion_unet_hybrid_image_policy.DiffusionUnetHybridImagePolicy'
            shape = 84
            if 'tool_hang' in checkpoint:
                shape = 240
            return env_target, policy_target, shape, cond
        env_target, policy_target, curr_shape, cond = get_curr_shape(checkpoint)
        def update_nested_dict(cfg, target_value, new_value):
            if isinstance(cfg, DictConfig):
                for key in cfg.keys():
                    value = cfg[key]
                    if isinstance(value, DictConfig):
                        # If the value is a DictConfig, recursively call the function
                        update_nested_dict(value, target_value, new_value)
                    elif value == target_value:
                        # If the value matches the target value, update it
                        cfg[key] = new_value
                        print(f'Updated {key} to {new_value}')
            return cfg
        # Change shape_meta, task
        curr_target = [3, curr_shape, curr_shape]; target_shape = [3, 360, 360]
        cfg['shape_meta'] = update_nested_dict(cfg['shape_meta'], curr_target, target_shape)
        cfg['task'] = update_nested_dict(cfg['task'], curr_target, target_shape)
        ##
        cfg['task']['env_runner']['n_test'] = num
        cfg['task']['env_runner']['n_train'] = 0 # 3 rollouts for evaluation. Old was 6
        cfg['task']['env_runner']['n_envs'] = (cfg['task']['env_runner']['n_test'] + cfg['task']['env_runner']['n_train'])//2
        cfg['task']['env_runner']['n_envs'] = min(cfg['task']['env_runner']['n_envs'], 50)
        cfg['task']['env_runner']['n_test_vis'] = cfg['task']['env_runner']['n_test'] # Store all videos for debugging
        cfg['task']['env_runner']['_target_'] = env_target
        cfg['policy']['num_inference_steps'] = num_inference_step # change this would change the number of inference steps
        cfg['policy']['_target_'] = policy_target
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=output_dir)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
        # get policy from workspace
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        policy.to(device)
        policy.eval()
        return policy, cfg, curr_shape, cond

    policy, cfg, curr_shape, cond = get_policy(ckpt_path) # Policy generator

    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    env_runner.curr_shape = curr_shape
    ## Get baseline comparison
    import eval_load_baseline as elb
    # Get DER
    baseline_model = elb.get_baseline_model('DER', task_name, policy_type=policy_type).to(device)
    env_runner.baseline_model = baseline_model; env_runner.task_name = task_name
    # Get RND
    baseline_model_RND = elb.get_baseline_model('RND', task_name, policy_type=policy_type).to(device)
    env_runner.baseline_model_RND = baseline_model_RND
    ## Get CFM
    baseline_model_CFM = elb.get_baseline_model('CFM', task_name, policy_type=policy_type).to(device)
    env_runner.baseline_model_CFM = baseline_model_CFM
    ## Get logpZO
    baseline_model_logpZO = elb.get_baseline_model('logpZO', task_name, policy_type=policy_type).to(device)
    env_runner.baseline_model_logpZO = baseline_model_logpZO
    env_runner.baseline_model_Discrepancy.global_eps = None
    # Get NatPN on (O_t, K-means label)
    baseline_model_natpn = elb.get_baseline_model('NatPN', task_name, policy_type=policy_type).to(device)
    env_runner.baseline_model_natpn = baseline_model_natpn
    # PCA + K-means as SOTA
    baseline_model_PCA_kmeans = elb.get_baseline_model('PCA_kmeans', task_name, policy_type=policy_type).to(device)
    env_runner.baseline_model_PCA_kmeans = baseline_model_PCA_kmeans
    #####
    env_runner.modify_t = 50 if 'can' not in ckpt_path else 15
    with torch.no_grad():
        env_runner.cond = cond
        runner_log = env_runner.run(policy, modify=modify)
            
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, json_filename)
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)
        
if __name__ == '__main__':
    main()