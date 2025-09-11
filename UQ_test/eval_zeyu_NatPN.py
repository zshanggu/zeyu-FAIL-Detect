#!/usr/bin/env python3
"""
Multi-Trajectory FAIL-Detect Inference Script with Evaluation Metrics
Processes multiple trajectory files from a folder to compute failure scores and evaluation metrics

Usage:
    python eval_zeyu.py --task_name='can' --policy_type='diffusion' --data_folder='path/to/trajectories'

Folder Structure:
    data_folder/
    ├── trajectory_001.pt  # Contains {'X': [T1, 274], 'Y': [T1, 160], 'risk': risk_timestep}
    ├── trajectory_002.pt  # Contains {'X': [T2, 274], 'Y': [T2, 160], 'risk': risk_timestep}
    └── ...

Output Structure:
    results/
    ├── plots/
    │   ├── trajectory_001.png      # Visualization plot
    │   ├── trajectory_002.png
    │   └── ...
    ├── evaluation_metrics.txt      # Precision, Recall, Execution Rate
    ├── detailed_results.csv        # Per-trajectory detailed results
    └── summary_results.pkl         # Complete results
"""

import torch
import numpy as np
import pickle
import argparse
import os
import sys
import matplotlib.pyplot as plt
import glob
import pandas as pd
from scipy import stats
from pathlib import Path

# Add UQ_baselines to path for loading models
sys.path.append('UQ_baselines/')
sys.path.append('../UQ_baselines/')  # Alternative path
try:
    import eval_load_baseline as elb
except ImportError:
    print("Error: Could not import eval_load_baseline. Make sure UQ_baselines directory is accessible.")
    sys.exit(1)

class MultiTrajectoryFailureDetector:
    def __init__(self, task_name, policy_type, device='cuda'):
        """
        Initialize the failure detector for processing multiple trajectories
        
        Args:
            task_name: Task name ('can', 'square', 'transport', 'toolhang')
            policy_type: Policy type ('diffusion' or 'flow')
            device: Device to run on ('cuda' or 'cpu')
        """
        self.task_name = task_name
        self.policy_type = policy_type
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load the trained NatPN model using the exact same logic as eval_load_baseline.py
        print(f"Loading NatPN model for {task_name} with {policy_type} policy...")
        self.NatPN_model = elb.get_baseline_model('NatPN', task_name, policy_type=policy_type).to(self.device)
        self.NatPN_model.eval()
        print("Model loaded successfully!")
        
        # Show reshaping info for NatPN
        in_dim = 10 if task_name != 'transport' else 20
        print(f"\nNatPN model expects:")
        print(f"  Input chunks of size: {in_dim}")
        print(f"  Your 274D observations will be reshaped accordingly")
        
        # Set global_eps attribute for logpO compatibility (even though we're using NatPN)
        if not hasattr(self.NatPN_model, 'global_eps'):
            self.NatPN_model.global_eps = None
        
    def find_trajectory_files(self, data_folder):
        """
        Find all .pt trajectory files in the specified folder
        
        Args:
            data_folder: Path to folder containing trajectory files
            
        Returns:
            List of trajectory file paths sorted by filename
        """
        print(f"Searching for trajectory files in: {data_folder}")
        
        # Look for .pt files
        pt_files = glob.glob(os.path.join(data_folder, "*.pt"))
        pt_files.sort()  # Sort for consistent ordering
        
        if len(pt_files) == 0:
            raise ValueError(f"No .pt files found in {data_folder}")
            
        print(f"Found {len(pt_files)} trajectory files:")
        for i, file_path in enumerate(pt_files[:5]):  # Show first 5
            filename = os.path.basename(file_path)
            print(f"  {i+1:3d}. {filename}")
        if len(pt_files) > 5:
            print(f"  ... and {len(pt_files) - 5} more files")
            
        return pt_files
        
    def load_single_trajectory(self, file_path):
        """
        Load a single trajectory file
        
        Args:
            file_path: Path to .pt trajectory file
            
        Returns:
            observations: (T, obs_dim) - T timesteps, obs_dim features
            actions: (T, act_dim) - T timesteps, act_dim actions
            t_stop: int or None - Ground truth failure timestep (risk_timestep)
            filename: Base filename without extension
        """
        filename = os.path.splitext(os.path.basename(file_path))[0]
        
        try:
            data = torch.load(file_path, map_location='cpu')
            
            if 'X' in data and 'Y' in data:
                observations = data['X'].numpy()  # Convert to numpy
                actions = data['Y'].numpy()
            else:
                raise ValueError(f"Expected keys 'X' and 'Y' in {file_path}")
            
            # Load risk timestep (t_stop) if available
            t_stop = None
            if 'risk' in data:
                t_stop = data['risk']
                # Convert to int if it's a tensor/numpy array
                if hasattr(t_stop, 'item'):
                    t_stop = int(t_stop.item())
                elif hasattr(t_stop, '__len__') and len(t_stop) == 1:
                    t_stop = int(t_stop[0])
                else:
                    t_stop = int(t_stop)
                
                # Handle -1 case (no failure)
                if t_stop == -1:
                    t_stop = None
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None, None, None
            
        # Ensure 2D shape (T, dim)
        if len(observations.shape) != 2 or len(actions.shape) != 2:
            print(f"Warning: Unexpected shape in {filename}")
            print(f"  Observations: {observations.shape}, Actions: {actions.shape}")
            
        return observations, actions, t_stop, filename
        
    def compute_failure_scores_single(self, observations):
        """
        Compute NatPN failure scores for a single trajectory
        
        Args:
            observations: (T, obs_dim) - Single trajectory observations
            
        Returns:
            scores: (T,) - Failure score for each timestep
        """
        T, obs_dim = observations.shape
        
        # Convert to tensor and add batch dimension
        obs_tensor = torch.from_numpy(observations).float().to(self.device)  # (T, obs_dim)
        
        with torch.no_grad():
            # Use the exact same NatPN_UQ function from eval_load_baseline.py
            # This function:
            # 1. Reshapes observations into chunks (e.g., 274D -> chunks of 10D)
            # 2. Calls the flow model at t=0 to get velocity: v = model(obs, t=0)  
            # 3. Moves to noise space: z = obs + v
            # 4. Computes squared norm: score = ||z||²
            # Higher scores indicate observations that are "far" from the training distribution
            scores = elb.NatPN_UQ(
                baseline_model=self.NatPN_model,
                observation=obs_tensor
                # task_name=self.task_name  # Now properly passed
            )
            
        return scores.cpu().numpy()
        
    def compute_thresholds_single(self, scores, method='percentile', percentile=95.0):
        """
        Compute thresholds for a single trajectory based on its own score distribution
        
        Args:
            scores: (T,) - Failure scores for this trajectory
            method: 'percentile', 'mean_std', or 'constant'
            percentile: Percentile for threshold (if method='percentile')
            
        Returns:
            thresholds: (T,) - Time-varying threshold for each timestep
        """
        T = len(scores)
        
        if method == 'percentile':
            # Use rolling percentile with window
            window_size = min(50, T // 4)  # Adaptive window size
            if window_size < 5:
                # For very short trajectories, use global percentile
                threshold_val = np.percentile(scores, percentile)
                thresholds = np.full(T, threshold_val)
            else:
                # Rolling percentile
                thresholds = np.zeros(T)
                for t in range(T):
                    start_idx = max(0, t - window_size // 2)
                    end_idx = min(T, t + window_size // 2 + 1)
                    window_scores = scores[start_idx:end_idx]
                    thresholds[t] = np.percentile(window_scores, percentile)
            
        elif method == 'mean_std':
            # Use rolling mean + 2*std
            window_size = min(50, T // 4)
            if window_size < 5:
                # For very short trajectories, use global stats
                threshold_val = np.mean(scores) + 2 * np.std(scores)
                thresholds = np.full(T, threshold_val)
            else:
                # Rolling mean + std
                thresholds = np.zeros(T)
                for t in range(T):
                    start_idx = max(0, t - window_size // 2)
                    end_idx = min(T, t + window_size // 2 + 1)
                    window_scores = scores[start_idx:end_idx]
                    thresholds[t] = np.mean(window_scores) + 2 * np.std(window_scores)
                    
        elif method == 'constant':
            # Use global percentile as constant threshold
            threshold_val = np.percentile(scores, percentile)
            thresholds = np.full(T, threshold_val)
            
        else:
            raise ValueError("Method must be 'percentile', 'mean_std', or 'constant'")
            
        return thresholds
        
    def detect_failures_single(self, scores, thresholds):
        """
        Detect potential failures for a single trajectory
        
        Args:
            scores: (T,) - Failure scores
            thresholds: (T,) - Thresholds
            
        Returns:
            is_failure_predicted: bool - Whether any failure was predicted
            t_pred_stop: int - Timestep when failure was first predicted (-1 if none)
            exceed_scores: (T,) - Amount by which scores exceed thresholds
        """
        T = len(scores)
        exceed_scores = scores - thresholds
        failure_mask = scores > thresholds
        
        if np.any(failure_mask):
            t_pred_stop = np.argmax(failure_mask)  # First occurrence
            is_failure_predicted = True
        else:
            t_pred_stop = -1
            is_failure_predicted = False
            
        return is_failure_predicted, t_pred_stop, exceed_scores
        
    def classify_episode(self, is_failure_predicted, t_pred_stop, t_stop):
        """
        Classify episode according to the confusion matrix
        
        Args:
            is_failure_predicted: bool - Whether failure was predicted
            t_pred_stop: int - Predicted failure timestep (-1 if none)
            t_stop: int or None - Ground truth failure timestep (None if no failure)
            
        Returns:
            prediction_type: str - 'TP', 'FP1', 'FP2', 'FN', or 'TN'
            tau_gt: str - 'τ_gt_fail' or 'τ_gt_no−fail'
            tau_pred: str - 'τ_pred_fail' or 'τ_pred_no−fail'
        """
        # Determine ground truth label
        if t_stop is not None:
            tau_gt = 'τ_gt_fail'
        else:
            tau_gt = 'τ_gt_no−fail'
            
        # Determine predicted label
        if is_failure_predicted:
            tau_pred = 'τ_pred_fail'
        else:
            tau_pred = 'τ_pred_no−fail'
            
        # Classify according to confusion matrix
        if tau_gt == 'τ_gt_fail' and tau_pred == 'τ_pred_fail':
            # Check if prediction is on time
            if t_pred_stop <= t_stop:
                prediction_type = 'TP'  # True Positive
            else:
                prediction_type = 'FP2'  # False Positive (too late - Late Prediction)
        elif tau_gt == 'τ_gt_no−fail' and tau_pred == 'τ_pred_fail':
            prediction_type = 'FP1'  # False Positive (False Alarm)
        elif tau_gt == 'τ_gt_fail' and tau_pred == 'τ_pred_no−fail':
            prediction_type = 'FN'  # False Negative
        elif tau_gt == 'τ_gt_no−fail' and tau_pred == 'τ_pred_no−fail':
            prediction_type = 'TN'  # True Negative
        else:
            prediction_type = 'UNKNOWN'
            
        return prediction_type, tau_gt, tau_pred
        
    def process_single_trajectory(self, file_path, threshold_method='percentile', percentile=95.0):
        """
        Process a single trajectory file and return results
        
        Args:
            file_path: Path to trajectory .pt file
            threshold_method: Method to compute thresholds
            percentile: Percentile for threshold computation
            
        Returns:
            Dictionary with results or None if failed
        """
        # Load trajectory
        observations, actions, t_stop, filename = self.load_single_trajectory(file_path)
        if observations is None:
            return None
            
        T, obs_dim = observations.shape
        T_act, act_dim = actions.shape
        
        print(f"\nProcessing {filename}:")
        print(f"  Timesteps: {T}")
        print(f"  Observation dim: {obs_dim}")
        print(f"  Action dim: {act_dim}")
        if t_stop is not None:
            print(f"  Ground truth failure timestep (t_stop): {t_stop}")
        else:
            print(f"  Ground truth: No failure (τ_gt_no−fail)")
        
        # Compute failure scores
        scores = self.compute_failure_scores_single(observations)
        
        # Compute thresholds based on this trajectory's score distribution
        thresholds = self.compute_thresholds_single(scores, threshold_method, percentile)
        
        # Detect potential failures
        is_failure_predicted, t_pred_stop, exceed_scores = self.detect_failures_single(scores, thresholds)
        
        # Classify episode according to confusion matrix
        prediction_type, tau_gt, tau_pred = self.classify_episode(is_failure_predicted, t_pred_stop, t_stop)
        
        # Calculate execution rate for TP cases
        execution_rate = None
        if prediction_type == 'TP' and t_stop is not None and t_stop > 0:
            execution_rate = t_pred_stop / t_stop
        
        # Create results dictionary
        results = {
            'filename': filename,
            'scores': scores,                    # (T,) - failure score per timestep
            'thresholds': thresholds,           # (T,) - threshold per timestep  
            'is_failure_predicted': is_failure_predicted,  # bool - any failure predicted?
            't_pred_stop': t_pred_stop,         # int - when first predicted (-1 if none)
            't_stop': t_stop,                   # int or None - ground truth failure timestep
            'prediction_type': prediction_type, # str - 'TP', 'FP1', 'FP2', 'FN', 'TN'
            'tau_gt': tau_gt,                   # str - ground truth label
            'tau_pred': tau_pred,               # str - predicted label
            'execution_rate': execution_rate,   # float or None - t_pred_stop/t_stop for TP
            'exceed_scores': exceed_scores,     # (T,) - how much scores exceed thresholds
            'observations': observations,       # (T, obs_dim) - original observations
            'actions': actions,                 # (T, act_dim) - original actions
            'trajectory_length': T,
            'score_stats': {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores), 
                'max': np.max(scores)
            },
            'threshold_stats': {
                'mean': np.mean(thresholds),
                'std': np.std(thresholds),
                'min': np.min(thresholds),
                'max': np.max(thresholds)
            }
        }
        
        print(f"  Score range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
        print(f"  Prediction: {tau_pred}")
        print(f"  Classification: {prediction_type}")
        if prediction_type == 'TP':
            print(f"  Execution rate: {execution_rate:.3f}")
        if is_failure_predicted and t_stop is not None:
            time_diff = t_pred_stop - t_stop
            print(f"  Prediction vs Ground Truth: {time_diff:+d} timesteps")
            
        return results
        
    def create_trajectory_plot(self, results, output_dir):
        """
        Create visualization plot for a single trajectory and save directly to plots folder
        
        Args:
            results: Results dictionary
            output_dir: Directory to save plots
        """
        filename = results['filename']
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        scores = results['scores']
        thresholds = results['thresholds']
        t_stop = results['t_stop']
        t_pred_stop = results['t_pred_stop']
        T = len(scores)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot scores and thresholds
        ax.plot(scores, 'blue', label='Failure Score', linewidth=1.5)
        ax.plot(thresholds, 'red', label='Threshold', linewidth=2)
        ax.fill_between(range(T), thresholds, np.max(scores), alpha=0.2, color='red', label='Potential Failure Zone')
        
        # Mark predicted failure point if any
        if results['is_failure_predicted']:
            ax.axvline(t_pred_stop, color='orange', linestyle='--', linewidth=2, 
                      label=f'Predicted Failure (t_pred_stop={t_pred_stop})')
            ax.scatter(t_pred_stop, scores[t_pred_stop], color='orange', s=100, zorder=5)
        
        # Mark ground truth failure timestep if available
        if t_stop is not None and 0 <= t_stop < T:
            ax.axvline(t_stop, color='green', linestyle=':', linewidth=3, 
                      label=f'Ground Truth Failure (t_stop={t_stop})')
            ax.scatter(t_stop, scores[t_stop], color='green', s=100, marker='s', zorder=5)
            
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Score')
        ax.set_title(f'{filename} - Classification: {results["prediction_type"]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add text annotation with key information
        info_text = f"Length: {T} steps\n"
        info_text += f"Ground Truth: {results['tau_gt']}\n"
        info_text += f"Prediction: {results['tau_pred']}\n"
        info_text += f"Classification: {results['prediction_type']}"
        
        if results['execution_rate'] is not None:
            info_text += f"\nExecution Rate: {results['execution_rate']:.3f}"
        if results['is_failure_predicted'] and t_stop is not None:
            time_diff = t_pred_stop - t_stop
            info_text += f"\nTime Difference: {time_diff:+d} steps"
                
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{filename}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def calculate_evaluation_metrics(self, all_results):
        """
        Calculate precision, recall, and execution rate across all trajectories
        
        Args:
            all_results: List of result dictionaries
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Count prediction types
        tp_count = sum(1 for r in all_results if r['prediction_type'] == 'TP')
        fp1_count = sum(1 for r in all_results if r['prediction_type'] == 'FP1')
        fp2_count = sum(1 for r in all_results if r['prediction_type'] == 'FP2')
        fn_count = sum(1 for r in all_results if r['prediction_type'] == 'FN')
        tn_count = sum(1 for r in all_results if r['prediction_type'] == 'TN')
        
        total_episodes = len(all_results)
        
        # Calculate precision and recall
        precision = tp_count / (tp_count + fp1_count + fp2_count) if (tp_count + fp1_count + fp2_count) > 0 else 0.0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate accuracy
        accuracy = (tp_count + tn_count) / total_episodes if total_episodes > 0 else 0.0
        
        # Calculate execution rate (only for TP cases)
        tp_execution_rates = [r['execution_rate'] for r in all_results 
                             if r['prediction_type'] == 'TP' and r['execution_rate'] is not None]
        
        mean_execution_rate = np.mean(tp_execution_rates) if tp_execution_rates else None
        std_execution_rate = np.std(tp_execution_rates) if tp_execution_rates else None
        median_execution_rate = np.median(tp_execution_rates) if tp_execution_rates else None
        
        # For mode with continuous data, use the most frequent value (may not be meaningful for small datasets)
        mode_execution_rate = None
        if tp_execution_rates:
            try:
                mode_result = stats.mode(tp_execution_rates, keepdims=True)
                mode_execution_rate = mode_result.mode[0] if len(mode_result.mode) > 0 else None
            except:
                mode_execution_rate = None

        # Ground truth and prediction distributions
        gt_fail_count = sum(1 for r in all_results if r['tau_gt'] == 'τ_gt_fail')
        gt_no_fail_count = sum(1 for r in all_results if r['tau_gt'] == 'τ_gt_no−fail')
        pred_fail_count = sum(1 for r in all_results if r['tau_pred'] == 'τ_pred_fail')
        pred_no_fail_count = sum(1 for r in all_results if r['tau_pred'] == 'τ_pred_no−fail')
        
        metrics = {
            'total_episodes': total_episodes,
            'confusion_matrix': {
                'TP': tp_count,
                'FP1': fp1_count,
                'FP2': fp2_count,
                'FN': fn_count,
                'TN': tn_count
            },
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'execution_rate': {
                'mean': mean_execution_rate,
                'std': std_execution_rate,
                'median': median_execution_rate,
                'mode': mode_execution_rate,
                'count': len(tp_execution_rates),
                'values': tp_execution_rates
            },
            'ground_truth_distribution': {
                'τ_gt_fail': gt_fail_count,
                'τ_gt_no−fail': gt_no_fail_count
            },
            'prediction_distribution': {
                'τ_pred_fail': pred_fail_count,
                'τ_pred_no−fail': pred_no_fail_count
            }
        }
        
        return metrics

def main():
    parser = argparse.ArgumentParser(description='Multi-trajectory FAIL-Detect inference with evaluation metrics')
    parser.add_argument('--task_name', required=True, choices=['can', 'square', 'transport', 'toolhang'],
                       help='Task name')
    parser.add_argument('--policy_type', required=True, choices=['diffusion', 'flow'],
                       help='Policy type used to collect data')
    parser.add_argument('--data_folder', required=True, help='Path to folder containing trajectory .pt files')
    parser.add_argument('--output_dir', default='evaluation_results', 
                       help='Directory to save results')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--threshold_method', default='percentile', 
                       choices=['percentile', 'mean_std', 'constant'],
                       help='Method to compute thresholds')
    parser.add_argument('--percentile', type=float, default=95.0,
                       help='Percentile for threshold computation')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save visualization plots for each trajectory')
    parser.add_argument('--max_trajectories', type=int, default=None,
                       help='Maximum number of trajectories to process (for testing)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("MULTI-TRAJECTORY FAIL-DETECT EVALUATION")
    print("="*70)
    print(f"Task: {args.task_name}")
    print(f"Policy: {args.policy_type}")
    print(f"Data folder: {args.data_folder}")
    print(f"Output directory: {args.output_dir}")
    print(f"Threshold method: {args.threshold_method}")
    print(f"Device: {args.device}")
    print("="*70)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize detector
    detector = MultiTrajectoryFailureDetector(
        task_name=args.task_name,
        policy_type=args.policy_type,
        device=args.device
    )
    
    # Find trajectory files
    trajectory_files = detector.find_trajectory_files(args.data_folder)
    
    if args.max_trajectories:
        trajectory_files = trajectory_files[:args.max_trajectories]
        print(f"Processing only first {args.max_trajectories} trajectories")
    
    # Process each trajectory
    all_results = []
    failed_files = []
    
    print(f"\nProcessing {len(trajectory_files)} trajectories...")
    print("-" * 50)
    
    for i, file_path in enumerate(trajectory_files):
        print(f"\n[{i+1}/{len(trajectory_files)}] {os.path.basename(file_path)}")
        
        try:
            # Process trajectory
            results = detector.process_single_trajectory(
                file_path, 
                threshold_method=args.threshold_method,
                percentile=args.percentile
            )
            
            if results is not None:
                # Create plot if requested
                if args.save_plots:
                    detector.create_trajectory_plot(results, args.output_dir)
                
                all_results.append(results)
            else:
                failed_files.append(file_path)
                
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            failed_files.append(file_path)
    
    # Calculate evaluation metrics
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    if len(all_results) > 0:
        metrics = detector.calculate_evaluation_metrics(all_results)
        
        # Print metrics
        print(f"Total Episodes: {metrics['total_episodes']}")
        print(f"Successfully processed: {len(all_results)}/{len(trajectory_files)} trajectories")
        if failed_files:
            print(f"Failed files: {len(failed_files)}")
        
        print(f"\nConfusion Matrix:")
        print(f"  True Positives (TP):  {metrics['confusion_matrix']['TP']}")
        print(f"  False Positives FP1 (False Alarm): {metrics['confusion_matrix']['FP1']}")
        print(f"  False Positives FP2 (Late Prediction): {metrics['confusion_matrix']['FP2']}")
        print(f"  False Negatives (FN): {metrics['confusion_matrix']['FN']}")
        print(f"  True Negatives (TN):  {metrics['confusion_matrix']['TN']}")
        
        print(f"\nEvaluation Metrics:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        
        if metrics['execution_rate']['mean'] is not None:
            print(f"\nExecution Rate (TP cases only):")
            print(f"  Mean: {metrics['execution_rate']['mean']:.4f}")
            print(f"  Median: {metrics['execution_rate']['median']:.4f}")
            if metrics['execution_rate']['mode'] is not None:
                print(f"  Mode:   {metrics['execution_rate']['mode']:.4f}")
            print(f"  Std:  {metrics['execution_rate']['std']:.4f}")
            print(f"  Count: {metrics['execution_rate']['count']} TP cases")
        else:
            print(f"\nExecution Rate: No TP cases found")
        
        print(f"\nGround Truth Distribution:")
        print(f"  τ_gt_fail:     {metrics['ground_truth_distribution']['τ_gt_fail']}")
        print(f"  τ_gt_no−fail:  {metrics['ground_truth_distribution']['τ_gt_no−fail']}")
        
        print(f"\nPrediction Distribution:")
        print(f"  τ_pred_fail:     {metrics['prediction_distribution']['τ_pred_fail']}")
        print(f"  τ_pred_no−fail:  {metrics['prediction_distribution']['τ_pred_no−fail']}")
        
        # Save detailed results as CSV
        detailed_data = []
        for r in all_results:
            detailed_data.append({
                'filename': r['filename'],
                'trajectory_length': r['trajectory_length'],
                't_stop': r['t_stop'] if r['t_stop'] is not None else -1,
                't_pred_stop': r['t_pred_stop'],
                'tau_gt': r['tau_gt'],
                'tau_pred': r['tau_pred'],
                'prediction_type': r['prediction_type'],
                'execution_rate': r['execution_rate'] if r['execution_rate'] is not None else -1,
                'score_mean': r['score_stats']['mean'],
                'score_std': r['score_stats']['std'],
                'threshold_mean': r['threshold_stats']['mean']
            })
        
        df = pd.DataFrame(detailed_data)
        csv_path = os.path.join(args.output_dir, 'detailed_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nDetailed results saved to: {csv_path}")
        
        # Save evaluation metrics as text
        metrics_path = os.path.join(args.output_dir, 'evaluation_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write("FAIL-Detect Evaluation Metrics\n")
            f.write("="*40 + "\n\n")
            f.write(f"Task: {args.task_name}\n")
            f.write(f"Policy: {args.policy_type}\n")
            f.write(f"Total Episodes: {metrics['total_episodes']}\n\n")
            
            f.write("Confusion Matrix:\n")
            f.write(f"  TP: {metrics['confusion_matrix']['TP']}\n")
            f.write(f"  FP1: {metrics['confusion_matrix']['FP1']}\n")
            f.write(f"  FP2: {metrics['confusion_matrix']['FP2']}\n")
            f.write(f"  FN: {metrics['confusion_matrix']['FN']}\n")
            f.write(f"  TN: {metrics['confusion_matrix']['TN']}\n\n")
            
            f.write("Evaluation Metrics:\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall: {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score: {metrics['f1_score']:.4f}\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n\n")
            
            if metrics['execution_rate']['mean'] is not None:
                f.write("Execution Rate (TP cases):\n")
                f.write(f"  Mean: {metrics['execution_rate']['mean']:.4f}\n")
                f.write(f"  Median: {metrics['execution_rate']['median']:.4f}\n")
                if metrics['execution_rate']['mode'] is not None:
                    f.write(f"  Mode: {metrics['execution_rate']['mode']:.4f}\n")
                f.write(f"  Std: {metrics['execution_rate']['std']:.4f}\n")
                f.write(f"  Count: {metrics['execution_rate']['count']}\n")
            else:
                f.write("Execution Rate: No TP cases found\n")
        
        print(f"Evaluation metrics saved to: {metrics_path}")
        
        # Save complete results as pickle
        complete_results = {
            'task_name': args.task_name,
            'policy_type': args.policy_type,
            'threshold_method': args.threshold_method,
            'percentile': args.percentile,
            'metrics': metrics,
            'all_results': all_results,
            'failed_files': failed_files
        }
        
        summary_path = os.path.join(args.output_dir, 'summary_results.pkl')
        with open(summary_path, 'wb') as f:
            pickle.dump(complete_results, f)
        print(f"Complete results saved to: {summary_path}")
        
    print(f"\nAll results saved to: {args.output_dir}/")
    if args.save_plots:
        print(f"Trajectory plots saved to: {args.output_dir}/plots/")
    print("="*50)

if __name__ == '__main__':
    main()