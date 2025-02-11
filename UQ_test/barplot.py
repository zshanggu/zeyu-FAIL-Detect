import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from argparse import ArgumentParser

def str2bool(value):
    """Convert string to boolean for argparse."""
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = ArgumentParser()
parser.add_argument('--num_train', type=int, default=300, help='Number of training samples')
parser.add_argument('--num_cal', type=int, default=700, help='Number of testing samples')
parser.add_argument('--num_te', type=int, default=1000, help='Number of testing samples')
parser.add_argument('--diffusion_policy', action='store_true', help='Whether to use diffusion policy')
args = parser.parse_args()

if __name__ == '__main__':
    num_train, num_cal = args.num_train, args.num_cal
    num_te = args.num_te
    master_dir = 'full_results'
    sub_dir = f'barplots'
    curr_dir = os.path.join(master_dir, sub_dir)
    os.makedirs(curr_dir, exist_ok=True)
    mean_std_dict = {'square_ID': [151, 20], 
                     'square_OOD': [195, 60],
                     'transport_ID': [469, 54],
                     'transport_OOD': [484, 63],
                     'toolhang_ID': [480, 88],
                     'toolhang_OOD': [505, 90],
                     'can_ID': [116, 14],
                     'can_OOD': [116, 14]}
    for dataset in mean_std_dict.keys():
        if args.num_train < 20 and 'ID' in dataset:
            continue
        methods = ['STAC', 'PCA-kmeans', 'logpO', 'logpZO-Ot', 'DER', 'NatPN', 'CFM', 'RND-Ot+At']
        methods_name = ['STAC', 'PCA-kmeans', 'logpO', 'logpZO', 'DER', 'NatPN', 'CFM', 'RND']
        file = os.path.join(master_dir, 'data', f'metrics_failure_detection_tr{num_train}_cal{num_cal}_te{num_te}_DP{args.diffusion_policy}.pkl')
        with open(file, 'rb') as f:
            result = pickle.load(f)
        result = result.fillna(0)
        to_plot = ['Accuracy', 'Accuracy weighted', 'Detect Time']
        ncol = len(to_plot)
        time_idx = len(to_plot) - 1
        fig, ax = plt.subplots(1, ncol, figsize=(ncol * 7, 5))
        
        for i, metric in enumerate(to_plot):
            curr_result = result.loc[(dataset, metric)].loc[methods].round(2)
            curr_ax = ax[i]
            
            # Determine the color for each bar
            bar_colors = ['gray'] * len(methods)
            if metric == 'Detect Time':
                # Ignore zeros for the bottom indices
                non_zero_indices = np.where(curr_result != 0)[0]
                sorted_non_zero_indices = non_zero_indices[np.argsort(curr_result[non_zero_indices])]
                bottom_indices = sorted_non_zero_indices[:3]
                colors = ['red', 'skyblue', 'green']
                for idx, color in zip(bottom_indices, colors):
                    bar_colors[idx] = color
            else:
                # Get unique values sorted descendingly
                unique_values = np.unique(curr_result)
                sorted_unique_values = np.sort(unique_values)[::-1]  # Sort descending
                # Extract top 3 unique values (even if fewer than 3 exist)
                top_3_unique = sorted_unique_values[:3]
                # Assign colors to ranks: red (1st), skyblue (2nd), green (3rd)
                colors = ['red', 'skyblue', 'green']
                bar_colors = ['gray'] * len(curr_result)
                for idx, value in enumerate(curr_result):
                    if value in top_3_unique:
                        rank = np.where(sorted_unique_values == value)[0][0]  # Get rank (0 = highest)
                        bar_colors[idx] = colors[rank]  # Assign color based on rank
            
            curr_ax.bar(range(len(methods)), curr_result, color=bar_colors)
            
            # Compute and plot standard error for non-'Detect Time' metrics
            if i != time_idx:
                num_samples = args.num_te
                standard_error = curr_result * (1 - curr_result) / np.sqrt(num_samples)
                curr_ax.errorbar(range(len(methods)), curr_result, yerr=standard_error, fmt='none', ecolor='black', capsize=3)
                upper_lim = 1 if max(curr_result) < 0.94 else max(curr_result) * 1.3
                upper_lim = 1.15
                curr_ax.set_ylim([0, upper_lim])      
            else:
                curr_ax.set_ylim([0, max(curr_result) * 1.3])
            
            if i == time_idx:
                # Draw horizontal line with mean and std
                mean, std = mean_std_dict[dataset]
                curr_ax.fill_between([-0.5, len(methods) - 0.5], mean - std, mean + std, color='gray', alpha=0.5)
                curr_ax.axhline(mean, color='black', linestyle='--')
                curr_result_se = result.loc[(dataset, 'Detect Time SE')][:len(methods)]
                curr_ax.errorbar(range(len(methods)), curr_result, yerr=curr_result_se, fmt='none', ecolor='black', capsize=3)
            
            fontsize = 18
            metric_title_dict = {'Accuracy': 'Accuracy', 
                                 'Accuracy weighted': 'Weighted Accuracy', 
                                 'Detect Time': 'Detection Time'}
            metric_title = metric_title_dict[metric]
            curr_ax.set_title(metric_title, fontsize=fontsize+6)
            curr_ax.set_xticks(range(len(methods)))  # Ensure ticks are set correctly
            curr_ax.set_xticklabels(methods_name, rotation=25, fontsize=fontsize+2, ha='right')  # Added ha='right' for better alignment
            curr_ax.tick_params(axis='y', labelsize=fontsize)
            
            for j, v in enumerate(curr_result):
                label = str(np.round(v, 2))
                if i == time_idx:
                    if v == 0:
                        label = 'NaN'
                    else:
                        label = str(int(round(v, 0)))
                se = curr_result_se[j] if metric == 'Detect Time' else (v * (1 - v) / np.sqrt(num_samples))
                curr_ax.text(j, (v + se)*1.01, label, color='black', ha='center', va='bottom', fontsize=fontsize)
        
        fig.tight_layout()
        dir_now = os.path.join(curr_dir, 'Abbreviated')
        os.makedirs(dir_now, exist_ok=True)
        fig.savefig(os.path.join(dir_now, f'{dataset}_tr{num_train}_cal{num_cal}_te{num_te}_DP{args.diffusion_policy}_abbrev.png'), bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)