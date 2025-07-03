#!/usr/bin/env python3
"""
Multi-Trajectory FAIL-Detect Inference Script
Processes multiple trajectory files from a folder to compute failure scores and thresholds

This script processes each .pt trajectory file separately and outputs failure scores 
and thresholds for each timestep of each trajectory.

Usage:
    python multi_trajectory_inference.py --task_name='can' --policy_type='diffusion' --data_folder='path/to/trajectories'

Folder Structure:
    data_folder/
    ├── trajectory_001.pt  # Contains {'X': [T1, 274], 'Y': [T1, 160], 'risk': risk_timestep}
    ├── trajectory_002.pt  # Contains {'X': [T2, 274], 'Y': [T2, 160], 'risk': risk_timestep}
    ├── trajectory_003.pt  # Contains {'X': [T3, 274], 'Y': [T3, 160], 'risk': risk_timestep}
    └── ...

Output Structure:
    results/
    ├── trajectory_001/
    │   ├── failure_scores.npy      # Shape: (T1,) - failure score per timestep
    │   ├── thresholds.npy          # Shape: (T1,) - threshold per timestep
    │   └── trajectory_results.pkl  # Complete results with metadata
    ├── trajectory_002/
    │   ├── failure_scores.npy      # Shape: (T2,) 
    │   ├── thresholds.npy          # Shape: (T2,)
    │   └── trajectory_results.pkl
    └── summary_results.pkl         # Combined results for all trajectories
"""

import torch
import numpy as np
import pickle
import argparse
import os
import sys
import matplotlib.pyplot as plt
import glob
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
        
        # Load the trained logpZO model using the exact same logic as eval_load_baseline.py
        print(f"Loading logpZO model for {task_name} with {policy_type} policy...")
        self.logpZO_model = elb.get_baseline_model('logpZO', task_name, policy_type=policy_type).to(self.device)
        self.logpZO_model.eval()
        print("Model loaded successfully!")
        
        # Show reshaping info for logpZO
        in_dim = 10 if task_name != 'transport' else 20
        print(f"\nlogpZO model expects:")
        print(f"  Input chunks of size: {in_dim}")
        print(f"  Your 274D observations will be reshaped accordingly")
        
        # Set global_eps attribute for logpO compatibility (even though we're using logpZO)
        if not hasattr(self.logpZO_model, 'global_eps'):
            self.logpZO_model.global_eps = None
        
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
            risk_timestep: int or None - Ground truth risk/failure timestep
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
            
            # Load risk timestep if available
            risk_timestep = None
            if 'risk' in data:
                risk_timestep = data['risk']
                # Convert to int if it's a tensor/numpy array
                if hasattr(risk_timestep, 'item'):
                    risk_timestep = int(risk_timestep.item())
                elif hasattr(risk_timestep, '__len__') and len(risk_timestep) == 1:
                    risk_timestep = int(risk_timestep[0])
                else:
                    risk_timestep = int(risk_timestep)
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None, None, None
            
        # Ensure 2D shape (T, dim)
        if len(observations.shape) != 2 or len(actions.shape) != 2:
            print(f"Warning: Unexpected shape in {filename}")
            print(f"  Observations: {observations.shape}, Actions: {actions.shape}")
            
        return observations, actions, risk_timestep, filename
        
    def compute_failure_scores_single(self, observations):
        """
        Compute logpZO failure scores for a single trajectory
        
        Args:
            observations: (T, obs_dim) - Single trajectory observations
            
        Returns:
            scores: (T,) - Failure score for each timestep
        """
        T, obs_dim = observations.shape
        
        # Convert to tensor and add batch dimension
        obs_tensor = torch.from_numpy(observations).float().to(self.device)  # (T, obs_dim)
        
        with torch.no_grad():
            # Use the exact same logpZO_UQ function from eval_load_baseline.py
            # This function:
            # 1. Reshapes observations into chunks (e.g., 274D -> chunks of 10D)
            # 2. Calls the flow model at t=0 to get velocity: v = model(obs, t=0)  
            # 3. Moves to noise space: z = obs + v
            # 4. Computes squared norm: score = ||z||²
            # Higher scores indicate observations that are "far" from the training distribution
            scores = elb.logpZO_UQ(
                baseline_model=self.logpZO_model,
                observation=obs_tensor,
                action_pred=None,  # Not using actions for logpZO
                task_name=self.task_name
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
            is_failure: bool - Whether any failure was detected
            detection_time: int - Timestep when failure was first detected (-1 if none)
            exceed_scores: (T,) - Amount by which scores exceed thresholds
        """
        T = len(scores)
        exceed_scores = scores - thresholds
        failure_mask = scores > thresholds
        
        if np.any(failure_mask):
            detection_time = np.argmax(failure_mask)  # First occurrence
            is_failure = True
        else:
            detection_time = -1
            is_failure = False
            
        return is_failure, detection_time, exceed_scores
        
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
        observations, actions, risk_timestep, filename = self.load_single_trajectory(file_path)
        if observations is None:
            return None
            
        T, obs_dim = observations.shape
        T_act, act_dim = actions.shape
        
        print(f"\nProcessing {filename}:")
        print(f"  Timesteps: {T}")
        print(f"  Observation dim: {obs_dim}")
        print(f"  Action dim: {act_dim}")
        if risk_timestep is not None:
            print(f"  Ground truth risk timestep: {risk_timestep}")
        else:
            print(f"  Ground truth risk timestep: Not available")
        print(f"  Score range: computing...")
        
        # Compute failure scores
        scores = self.compute_failure_scores_single(observations)
        
        # Compute thresholds based on this trajectory's score distribution
        thresholds = self.compute_thresholds_single(scores, threshold_method, percentile)
        
        # Detect potential failures
        is_failure, detection_time, exceed_scores = self.detect_failures_single(scores, thresholds)
        
        # Create results dictionary
        results = {
            'filename': filename,
            'scores': scores,                    # (T,) - failure score per timestep
            'thresholds': thresholds,           # (T,) - threshold per timestep  
            'is_failure': is_failure,           # bool - any failure detected?
            'detection_time': detection_time,   # int - when first detected (-1 if none)
            'risk_timestep': risk_timestep,     # int or None - ground truth risk timestep
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
        
        print(f"  Score stats: mean={np.mean(scores):.4f}, std={np.std(scores):.4f}")
        print(f"  Score range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
        print(f"  Threshold range: [{np.min(thresholds):.4f}, {np.max(thresholds):.4f}]")
        print(f"  Failure detected: {is_failure}")
        if is_failure:
            print(f"  Detection time: {detection_time}")
        if risk_timestep is not None and is_failure:
            time_diff = detection_time - risk_timestep
            print(f"  Detection vs Ground Truth: {time_diff:+d} timesteps")
            
        return results
        
    def save_trajectory_results(self, results, output_dir):
        """
        Save results for a single trajectory
        
        Args:
            results: Results dictionary from process_single_trajectory
            output_dir: Directory to save results
        """
        filename = results['filename']
        traj_dir = os.path.join(output_dir, filename)
        os.makedirs(traj_dir, exist_ok=True)
        
        # Save scores and thresholds as numpy arrays
        np.save(os.path.join(traj_dir, 'failure_scores.npy'), results['scores'])
        np.save(os.path.join(traj_dir, 'thresholds.npy'), results['thresholds'])
        
        # Save complete results as pickle
        with open(os.path.join(traj_dir, 'trajectory_results.pkl'), 'wb') as f:
            pickle.dump(results, f)
            
        # Save summary as text
        with open(os.path.join(traj_dir, 'summary.txt'), 'w') as f:
            f.write(f"Trajectory: {filename}\n")
            f.write("="*40 + "\n\n")
            f.write(f"Length: {results['trajectory_length']} timesteps\n")
            f.write(f"Failure detected: {results['is_failure']}\n")
            if results['is_failure']:
                f.write(f"Detection time: {results['detection_time']}\n")
            if results['risk_timestep'] is not None:
                f.write(f"Ground truth risk time: {results['risk_timestep']}\n")
                if results['is_failure']:
                    time_diff = results['detection_time'] - results['risk_timestep']
                    f.write(f"Detection vs Ground Truth: {time_diff:+d} timesteps\n")
            f.write(f"\nScore Statistics:\n")
            for key, val in results['score_stats'].items():
                f.write(f"  {key}: {val:.6f}\n")
            f.write(f"\nThreshold Statistics:\n")
            for key, val in results['threshold_stats'].items():
                f.write(f"  {key}: {val:.6f}\n")
                
        print(f"  Saved to: {traj_dir}/")
        
    def create_trajectory_plot(self, results, output_dir):
        """
        Create visualization plot for a single trajectory
        
        Args:
            results: Results dictionary
            output_dir: Directory to save plot
        """
        filename = results['filename']
        traj_dir = os.path.join(output_dir, filename)
        
        scores = results['scores']
        thresholds = results['thresholds']
        risk_timestep = results['risk_timestep']
        T = len(scores)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot scores and thresholds
        ax.plot(scores, 'blue', label='Failure Score', linewidth=1.5)
        ax.plot(thresholds, 'red', label='Threshold', linewidth=2)
        ax.fill_between(range(T), thresholds, np.max(scores), alpha=0.2, color='red', label='Potential Failure Zone')
        
        # Mark detection point if any
        if results['is_failure']:
            det_time = results['detection_time']
            ax.axvline(det_time, color='orange', linestyle='--', linewidth=2, label=f'Detection at t={det_time}')
            ax.scatter(det_time, scores[det_time], color='orange', s=100, zorder=5)
        
        # Mark ground truth risk timestep if available
        if risk_timestep is not None and 0 <= risk_timestep < T:
            ax.axvline(risk_timestep, color='green', linestyle=':', linewidth=3, label=f'Ground Truth Risk at t={risk_timestep}')
            ax.scatter(risk_timestep, scores[risk_timestep], color='green', s=100, marker='s', zorder=5)
            
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Score')
        ax.set_title(f'Failure Detection - {filename}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add text annotation with key information
        info_text = f"Length: {T} steps"
        if results['is_failure']:
            info_text += f"\nDetected: t={results['detection_time']}"
        if risk_timestep is not None:
            info_text += f"\nGround Truth: t={risk_timestep}"
            if results['is_failure']:
                time_diff = results['detection_time'] - risk_timestep
                info_text += f"\nDifference: {time_diff:+d} steps"
                
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(traj_dir, 'trajectory_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Multi-trajectory FAIL-Detect inference')
    parser.add_argument('--task_name', required=True, choices=['can', 'square', 'transport', 'toolhang'],
                       help='Task name')
    parser.add_argument('--policy_type', required=True, choices=['diffusion', 'flow'],
                       help='Policy type used to collect data')
    parser.add_argument('--data_folder', required=True, help='Path to folder containing trajectory .pt files')
    parser.add_argument('--output_dir', default='multi_trajectory_results', 
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
    print("MULTI-TRAJECTORY FAIL-DETECT INFERENCE")
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
                # Save trajectory results
                detector.save_trajectory_results(results, args.output_dir)
                
                # Create plot if requested
                if args.save_plots:
                    detector.create_trajectory_plot(results, args.output_dir)
                
                all_results.append(results)
            else:
                failed_files.append(file_path)
                
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            failed_files.append(file_path)
    
    # Create summary
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Successfully processed: {len(all_results)}/{len(trajectory_files)} trajectories")
    if failed_files:
        print(f"Failed files: {len(failed_files)}")
        for f in failed_files:
            print(f"  - {os.path.basename(f)}")
    
    if len(all_results) > 0:
        # Summary statistics
        total_timesteps = sum(r['trajectory_length'] for r in all_results)
        failures_detected = sum(1 for r in all_results if r['is_failure'])
        has_ground_truth = sum(1 for r in all_results if r['risk_timestep'] is not None)
        
        print(f"\nOverall Statistics:")
        print(f"  Total timesteps processed: {total_timesteps}")
        print(f"  Trajectories with failures detected: {failures_detected}/{len(all_results)}")
        print(f"  Trajectories with ground truth risk: {has_ground_truth}/{len(all_results)}")
        print(f"  Average trajectory length: {total_timesteps/len(all_results):.1f}")
        
        # Detection accuracy analysis if ground truth is available
        if has_ground_truth > 0:
            print(f"\nDetection Accuracy Analysis:")
            detection_errors = []
            early_detections = 0
            late_detections = 0
            correct_detections = 0
            missed_detections = 0
            
            for r in all_results:
                if r['risk_timestep'] is not None:
                    if r['is_failure']:
                        error = r['detection_time'] - r['risk_timestep']
                        detection_errors.append(error)
                        if error < 0:
                            early_detections += 1
                        elif error > 0:
                            late_detections += 1
                        else:
                            correct_detections += 1
                    else:
                        missed_detections += 1
            
            print(f"  Early detections: {early_detections}")
            print(f"  Exact detections: {correct_detections}")
            print(f"  Late detections: {late_detections}")
            print(f"  Missed detections: {missed_detections}")
            
            if detection_errors:
                print(f"  Mean detection error: {np.mean(detection_errors):.2f} timesteps")
                print(f"  Std detection error: {np.std(detection_errors):.2f} timesteps")
                print(f"  Detection error range: [{np.min(detection_errors)}, {np.max(detection_errors)}]")
        
        # Score statistics across all trajectories
        all_scores = np.concatenate([r['scores'] for r in all_results])
        all_thresholds = np.concatenate([r['thresholds'] for r in all_results])
        
        print(f"\nScore Statistics (across all trajectories):")
        print(f"  Mean: {np.mean(all_scores):.4f}")
        print(f"  Std: {np.std(all_scores):.4f}")
        print(f"  Range: [{np.min(all_scores):.4f}, {np.max(all_scores):.4f}]")
        
        print(f"\nThreshold Statistics (across all trajectories):")
        print(f"  Mean: {np.mean(all_thresholds):.4f}")
        print(f"  Std: {np.std(all_thresholds):.4f}")
        print(f"  Range: [{np.min(all_thresholds):.4f}, {np.max(all_thresholds):.4f}]")
        
        # Save combined summary
        summary = {
            'task_name': args.task_name,
            'policy_type': args.policy_type, 
            'threshold_method': args.threshold_method,
            'percentile': args.percentile,
            'total_trajectories': len(all_results),
            'failures_detected': failures_detected,
            'total_timesteps': total_timesteps,
            'all_results': all_results,
            'failed_files': failed_files,
            'global_stats': {
                'scores': {
                    'mean': np.mean(all_scores),
                    'std': np.std(all_scores),
                    'min': np.min(all_scores),
                    'max': np.max(all_scores)
                },
                'thresholds': {
                    'mean': np.mean(all_thresholds),
                    'std': np.std(all_thresholds),
                    'min': np.min(all_thresholds),
                    'max': np.max(all_thresholds)
                }
            }
        }
        
        summary_path = os.path.join(args.output_dir, 'summary_results.pkl')
        with open(summary_path, 'wb') as f:
            pickle.dump(summary, f)
        print(f"\nCombined summary saved to: {summary_path}")
        
        # Save trajectory list with key info
        summary_txt_path = os.path.join(args.output_dir, 'trajectory_summary.txt')
        with open(summary_txt_path, 'w') as f:
            f.write("Multi-Trajectory FAIL-Detect Results\n")
            f.write("="*50 + "\n\n")
            f.write(f"Task: {args.task_name}\n")
            f.write(f"Policy: {args.policy_type}\n")
            f.write(f"Total trajectories: {len(all_results)}\n")
            f.write(f"Failures detected: {failures_detected}\n\n")
            
            f.write("Per-Trajectory Results:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Trajectory':<20} | {'Steps':<5} | {'Status':<10} | {'Detection':<12} | {'Ground Truth':<12} | {'Error':<8}\n")
            f.write("-" * 60 + "\n")
            for i, r in enumerate(all_results):
                status = "FAILURE" if r['is_failure'] else "NORMAL"
                det_str = f"t={r['detection_time']}" if r['is_failure'] else "None"
                gt_str = f"t={r['risk_timestep']}" if r['risk_timestep'] is not None else "None"
                error_str = ""
                if r['is_failure'] and r['risk_timestep'] is not None:
                    error = r['detection_time'] - r['risk_timestep']
                    error_str = f"{error:+d}"
                f.write(f"{r['filename']:<20} | {r['trajectory_length']:<5} | {status:<10} | {det_str:<12} | {gt_str:<12} | {error_str:<8}\n")
                
        print(f"Trajectory summary saved to: {summary_txt_path}")
        
    print(f"\nAll results saved to: {args.output_dir}/")
    print("="*50)

if __name__ == '__main__':
    main()