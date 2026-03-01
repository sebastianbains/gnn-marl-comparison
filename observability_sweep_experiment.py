#!/usr/bin/env python3
"""
Observability Sweep Experiment
Tests GNN vs IQL performance across different communication/observation ranges
"""

import os
import sys
import json
import yaml
import numpy as np
from datetime import datetime
from pathlib import Path
import time
import copy

# add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from experiments.run_experiment import ExperimentRunner


def run_observability_sweep(config_path: str):
    """
    Run observability sweep experiment across multiple communication ranges.
    
    Args:
        config_path: Path to observability sweep configuration file
    """
    # load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    print("OBSERVABILITY SWEEP EXPERIMENT")
    print("=" * 80)
    print(f"Experiment: {config['experiment_name']}")
    print(f"Algorithms: {config['algorithms']}")
    print(f"Communication Ranges: {config['communication_ranges']}")
    print(f"Runs per configuration: {config['num_runs']}")
    print(f"Total experiments: {len(config['algorithms']) * len(config['communication_ranges']) * config['num_runs']}")
    print("=" * 80)
    
    # create results directory
    results_dir = Path(config['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # store all results
    all_results = {
        'experiment_name': config['experiment_name'],
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'config': config,
        'results': []
    }
    
    total_configs = len(config['algorithms']) * len(config['communication_ranges'])
    current_config = 0
    
    # sweep over communication ranges
    for comm_range in config['communication_ranges']:
        print(f"\n{'='*80}")
        print(f"COMMUNICATION RANGE: {comm_range}")
        print(f"{'='*80}")
        
        # run experiments for each algorithm
        for algorithm in config['algorithms']:
            current_config += 1
            print(f"\n[{current_config}/{total_configs}] Algorithm: {algorithm}, Range: {comm_range}")
            print("-" * 80)
            
            # run multiple runs for this configuration
            range_results = []
            for run_idx in range(config['num_runs']):
                print(f"  Run {run_idx + 1}/{config['num_runs']}...", end=" ", flush=True)
                
                # create a deep copy of config with unique seed for each run
                run_config = copy.deepcopy(config)
                run_config['communication_range'] = comm_range
                run_config['agent_config']['comm_range'] = comm_range
                
                # set unique seed for this specific run (varies by range and run index)
                run_seed = config['seed'] + run_idx + int(comm_range * 1000)
                run_config['seed'] = run_seed
                
                # create experiment runner with isolated config and unique seed
                runner = ExperimentRunner(run_config)
                
                try:
                    start_time = time.time()
                    
                    # convert task name to CoordinationTask enum
                    from environments.coordination_env import CoordinationTask
                    task_map = {
                        'TARGET_COVERAGE': CoordinationTask.TARGET_COVERAGE,
                        'RESOURCE_COLLECTION': CoordinationTask.RESOURCE_COLLECTION,
                        'FORMATION_CONTROL': CoordinationTask.FORMATION_CONTROL,
                        'COOPERATIVE_NAVIGATION': CoordinationTask.COOPERATIVE_NAVIGATION
                    }
                    task_type = task_map[config['tasks'][0]]
                    
                    # run single experiment with correct signature
                    experiment_result = runner.run_single_experiment(
                        algorithm_name=algorithm,
                        num_agents=config['agent_counts'][0],
                        task_type=task_type
                    )
                    
                    elapsed = time.time() - start_time
                    
                    # convert ExperimentResult to dict and add metadata
                    result = {
                        'algorithm_name': experiment_result.algorithm_name,
                        'task_name': experiment_result.task_name,
                        'num_agents': experiment_result.num_agents,
                        'episode_rewards': experiment_result.episode_rewards,
                        'episode_lengths': experiment_result.episode_lengths,
                        'success_rates': experiment_result.success_rates,
                        'final_performance': experiment_result.final_performance,
                        'communication_range': comm_range,
                        'run_index': run_idx,
                        'training_time': elapsed
                    }
                    
                    range_results.append(result)
                    
                    print(f"✓ (Final: {result['final_performance']:.1f}, Time: {elapsed:.1f}s)")
                    
                except Exception as e:
                    print(f"✗ Error: {e}")
                    continue
            
            # store results for this algorithm-range combination
            all_results['results'].extend(range_results)
            
            # compute summary statistics for this configuration
            if range_results:
                final_perfs = [r['final_performance'] for r in range_results]
                print(f"\n  Summary for {algorithm} @ range={comm_range}:")
                print(f"    Mean: {np.mean(final_perfs):.2f}")
                print(f"    Std:  {np.std(final_perfs, ddof=1):.2f}")
                print(f"    Min:  {np.min(final_perfs):.2f}")
                print(f"    Max:  {np.max(final_perfs):.2f}")
    
    # save raw results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = results_dir / f'observability_sweep_results_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")
    
    return output_file, all_results


def main():
    """Main entry point."""
    config_path = 'configs/observability_sweep.yaml'
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    output_file, results = run_observability_sweep(config_path)
    print(f"\nNext steps:")
    print(f"1. Run statistical analysis: python analyze_observability_sweep.py {output_file}")
    print(f"2. Generate visualizations")


if __name__ == '__main__':
    main()
