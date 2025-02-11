## FAIL-Detect: Failure Analysis in Imitation Learning – Detecting failures without failure data

- Please direct implementation questions to Chen Xu (chen.xu@tri.global).

## Prerequisite

Set up the environment by running

```
mamba env create -f conda_environment.yaml
```

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

```
# For flow policy (e.g, on the square task)
python save_data.py --config-dir=diffusion_policy/configs_robomimic \
--config-name=image_square_ph_visual_flow_policy_cnn.yaml \
training.seed=1103 training.device=cuda:0 hydra.run.dir='data/outputs/${name}_${task_name}' 

# For diffusion policy (e.g, on the square task)
python save_data.py --config-dir=diffusion_policy/configs_robomimic \
--config-name=image_square_ph_visual_diffusion_policy_cnn.yaml \
training.seed=1103 training.device=cuda:0 hydra.run.dir='data/outputs/${name}_${task_name}'
```

### 3. Train scalar scores given $\{(A_t, O_t)\}$

We give the examples of using **logpZO** and **RND**, which are the best performings ones. The other baselines are similar by swicthing to the corresponding folders

```
# Can change to other tasks among ['square', 'transport', 'tool_hang', 'can']

cd UQ_baselines/logpZO/ # Or change to /RND/, /CFM/, /NatPN/, /DER/ ...
# flow policy
python train.py --policy_type='flow' --type 'square'
# diffusion policy
python train.py --policy_type='diffusion' --type 'square'
```

### 4. Run evaluation

```
cd UQ_test
# modify = False is ID
python eval_together.py --policy_type='flow' --task_name='square' --device=0 --modify=false --num=100
python eval_together.py --policy_type='diffusion' --task_name='square' --device=0 --modify=false --num=100

# modify = True is OOD
python eval_together.py --policy_type='flow' --task_name='square' --device=0 --modify=true --num=100
python eval_together.py --policy_type='diffusion' --task_name='square' --device=0 --modify=true --num=100
```

### 5. CP band + visualization

```
cd UQ_test
python plot_with_CP_band.py --num_train 20 --num_cal 30 --num_te 50 # Generate CP band and make decision
python barplot.py --num_train 20 --num_cal 30 --num_te 50 # Generate barplots
```



---
---
- [TODO] Change back after testing
    - Data path: `/home/chenxu/Downloads/TRI_chen/sim_eval/data` to be `data`. 
    - Training epoch: `  num_epochs: 1` to be 800 for `can` and `square` and 300 for `transport` and `tool_hang`.
    - `trainer_params=dict(max_epochs=10)` be `trainer_params=dict(max_epochs=1000)`
    - 'policy.num_rep = 2' to be 'policy.num_rep = 256'
    - 'python eval_together.py --policy_type='flow' --task_name='square' --device=0 --modify=false --num=100' change to be 'python eval_together.py --policy_type='flow' --task_name='square' --device=0 --modify=false --num=2000'
    - Remove `--num_train 20 --num_cal 30 --num_te 50` below
    - Change 'EPOCHS = 5' to be 'EPOCHS = 200'