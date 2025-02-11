# # Step 2
# python save_data.py --config-dir=diffusion_policy/configs_robomimic \
#     --config-name=image_tool_hang_ph_visual_flow_policy_cnn.yaml \
#     training.seed=1103 training.device=cuda:2 hydra.run.dir="data/outputs/${name}_${task_name}" 
# python save_data.py --config-dir=diffusion_policy/configs_robomimic \
#     --config-name=image_tool_hang_ph_visual_diffusion_policy_cnn.yaml \
#     training.seed=1103 training.device=cuda:2 hydra.run.dir="data/outputs/${name}_${task_name}"
# # Step 3
# export CUDA_VISIBLE_DEVICES=1
# for folder in 'CFM' 'DER' 'logpZO' 'NatPN' 'PCA_kmeans' 'RND'
#     do  
#         if [ -d "UQ_baselines/$folder" ]; then
#             cd UQ_baselines/$folder
#             python train.py --policy_type='flow' --type tool_hang
#             python train.py --policy_type='diffusion' --type tool_hang
#             cd ../..
#         else
#             echo "Directory UQ_baselines/$folder does not exist."
#         fi
#     done
# Step 4
cd UQ_test
python eval_together.py --policy_type='flow' --task_name='tool_hang' --device=2 --modify=false --num=100
python eval_together.py --policy_type='diffusion' --task_name='tool_hang' --device=2 --modify=false --num=100

python eval_together.py --policy_type='flow' --task_name='tool_hang' --device=2 --modify=true --num=100
python eval_together.py --policy_type='diffusion' --task_name='tool_hang' --device=2 --modify=true --num=100

# Step 5
python plot_with_CP_band.py --num_train 20 --num_cal 30 --num_te 50 # Generate CP band and make decision
python barplot.py --num_train 20 --num_cal 30 --num_te 50 # Generate barplots