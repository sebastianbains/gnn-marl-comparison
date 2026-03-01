# consolidated analysis script for gnn-marl experiments
# combines statistical analysis and observability sweep analysis
import json
import numpy as np
from pathlib import Path

def analyze_baseline_experiment(results_path):
    # analyze baseline comparison between gnn and iql
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    iql_performances = []
    gnn_performances = []
    
    for result in data['raw_results']:
        if result['algorithm_name'] == 'IQL':
            iql_performances.append(result['final_performance'])
        elif result['algorithm_name'] == 'GNN':
            gnn_performances.append(result['final_performance'])
    
    print("=" * 60)
    print("BASELINE EXPERIMENT ANALYSIS")
    print("=" * 60)
    print(f"\nIQL (n={len(iql_performances)}):")
    print(f"  Mean: {np.mean(iql_performances):.2f}")
    print(f"  Std Dev: {np.std(iql_performances, ddof=1):.2f}")
    print(f"  Median: {np.median(iql_performances):.2f}")
    print(f"  Range: [{np.min(iql_performances):.2f}, {np.max(iql_performances):.2f}]")
    
    print(f"\nGNN (n={len(gnn_performances)}):")
    print(f"  Mean: {np.mean(gnn_performances):.2f}")
    print(f"  Std Dev: {np.std(gnn_performances, ddof=1):.2f}")
    print(f"  Median: {np.median(gnn_performances):.2f}")
    print(f"  Range: [{np.min(gnn_performances):.2f}, {np.max(gnn_performances):.2f}]")
    
    diff = np.mean(gnn_performances) - np.mean(iql_performances)
    pct = (diff / np.mean(iql_performances)) * 100
    pooled_std = np.sqrt((np.std(iql_performances, ddof=1)**2 + np.std(gnn_performances, ddof=1)**2) / 2)
    cohens_d = diff / pooled_std
    
    print(f"\nComparison:")
    print(f"  Mean Difference: +{diff:.2f}")
    print(f"  Percentage Improvement: {pct:.1f}%")
    print(f"  Cohen's d: {cohens_d:.3f}")
    
    return {
        'iql': iql_performances,
        'gnn': gnn_performances,
        'diff': diff,
        'pct': pct,
        'cohens_d': cohens_d
    }

def analyze_observability_sweep(results_path):
    # analyze observability sweep experiment
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    print("\n" + "=" * 60)
    print("OBSERVABILITY SWEEP ANALYSIS")
    print("=" * 60)
    
    ranges = sorted(set(r['comm_range'] for r in data['raw_results']))
    
    for comm_range in ranges:
        iql_perf = [r['final_performance'] for r in data['raw_results'] 
                    if r['algorithm_name'] == 'IQL' and r['comm_range'] == comm_range]
        gnn_perf = [r['final_performance'] for r in data['raw_results'] 
                    if r['algorithm_name'] == 'GNN' and r['comm_range'] == comm_range]
        
        if iql_perf and gnn_perf:
            diff = np.mean(gnn_perf) - np.mean(iql_perf)
            pct = (diff / np.mean(iql_perf)) * 100
            
            print(f"\nRange {comm_range}:")
            print(f"  IQL: {np.mean(iql_perf):.2f} ± {np.std(iql_perf, ddof=1):.2f}")
            print(f"  GNN: {np.mean(gnn_perf):.2f} ± {np.std(gnn_perf, ddof=1):.2f}")
            print(f"  Difference: {diff:+.2f} ({pct:+.1f}%)")

if __name__ == "__main__":
    # analyze baseline experiment
    baseline_path = Path("results/experiment_results_20260215_145731.json")
    if baseline_path.exists():
        analyze_baseline_experiment(baseline_path)
    else:
        print(f"Baseline results not found at {baseline_path}")
    
    # analyze observability sweep
    sweep_path = Path("results/observability_sweep/observability_sweep_results_20260216_115852.json")
    if sweep_path.exists():
        analyze_observability_sweep(sweep_path)
    else:
        print(f"\nObservability sweep results not found at {sweep_path}")
