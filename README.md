## FAIL-Detect: Failure Analysis in Imitation Learning – Detecting failures without failure data

- Please direct implementation questions to Chen Xu (chen.xu@tri.global).

## Prerequisite

Set up the environment by running

```
mamba env create -f conda_environment.yaml
```
- [TODO] Run everything (DP/FP x 4 tasks) from scratch (with small # of training steps) to verify everything is runnable
    - For the STAC repetition, change it to be 1 or 2 for speed
    - Any additional library needed? (see `requirements.txt` for DER and NatPN)
    
- [TODO] Change back 
    - `/home/chenxu/Downloads/TRI_chen/sim_eval/data` to be `data`. 
    - `  num_epochs: 5` to be 800 for `can` and `square` and 300 for `transport` and `tool_hang`.

## Usage

### 1. Policy training

**Tasks**: we support all except the `lift` task in robomimic, which is too simple to fail in most cases.

**Policy backbone**: either diffusion policy or flow-matching policy.

**Usage**: see `diffusion_policy/configs_robomimic` for the set of configs.

```
# This trains a flow policy (e.g, on the square task)
python train.py --config-dir=diffusion_policy/configs_robomimic --config-name=image_square_ph_visual_flow_policy_cnn.yaml training.seed=1103 training.device=cuda:0 hydra.run.dir='data/outputs/${name}_${task_name}'

# This trains a diffusion policy (e.g, on the square task)
python train.py --config-dir=diffusion_policy/configs_robomimic --config-name=image_square_ph_visual_diffusion_policy_cnn.yaml training.seed=1103 training.device=cuda:0 hydra.run.dir='data/outputs/${name}_${task_name}'

```

### 2. Obtain $\{(A_t, O_t)\}$ given a trained policy

Here, 
- $O_t$ = [Embedded visual features, non-visual information (e.g., robot states)]. 
- $A_t$ = corresponding action *in training data*.

**Note**: the required argument `ckpt_path` is the checkpoint for each task, which can be found in `data/outputs/folder_name/checkpoints`

```
# For flow policy (e.g, on the square task)
python save_data.py --config-dir=diffusion_policy/configs_robomimic \
--config-name=image_square_ph_visual_flow_policy_cnn.yaml \
training.seed=1103 training.device=cuda:0 hydra.run.dir='data/outputs/${name}_${task_name}' \
--policy_type=flow --ckpt_path=TO_BE_SPECIFIED

# For diffusion policy (e.g, on the square task)
python save_data.py --config-dir=diffusion_policy/configs_robomimic \
--config-name=image_square_ph_visual_diffusion_policy_cnn.yaml \
training.seed=1103 training.device=cuda:0 hydra.run.dir='data/outputs/${name}_${task_name}' \
--policy_type=diffusion --ckpt_path=TO_BE_SPECIFIED
```

### 3. Train scalar scores given $\{(A_t, O_t)\}$

We give the examples of using **logpZO** and **RND**, which are the best performings ones. The other baselines are similar by swicthing to the corresponding folders

```
# Can change to other tasks among ['square', 'transport', 'tool_hang', 'can'], or use diffusion policy by setting --policy_type='diffusion'

# For logpZO
cd UQ_baselines/logpZO/
python train.py --policy_type='flow' --type 'square'

# For RND
cd UQ_baselines/RND/
python train.py --policy_type='flow' --type 'square'
```

### 4. Run evaluation

**ckpt_path** is the same as step 2.
```
cd UQ_test
python eval_together.py --ckpt_path TO_BE_SPECIFIED \
--policy_type='flow' --task_name='square' --num_inference_step=1 \
--device=0 --modify=false --num=200
```

### 5. CP band + visualization

```
python plot_with_CP_band.py # Generate CP band and make decision
python barplot.py # Generate barplots

```
