#!/usr/bin/env python3
# statistical analysis for observability sweep experiment
# analyzes gnn vs iql performance across different communication ranges

import json
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(results_file: str) -> Dict:
    # load experiment results from json file
    with open(results_file, 'r') as f:
        return json.load(f)


def organize_data(results: Dict) -> pd.DataFrame:
    # organize results into a pandas dataframe for analysis
    data = []
    for result in results['results']:
        data.append({
            'algorithm': result['algorithm_name'],
            'comm_range': result['communication_range'],
            'run_index': result['run_index'],
            'final_performance': result['final_performance'],
            'training_time': result.get('training_time', 0),
            'mean_reward': np.mean(result['episode_rewards']),
            'success_rate': np.mean(result['success_rates'])
        })
    
    return pd.DataFrame(data)


def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    # compute summary statistics for each algorithm-range combination
    summary = df.groupby(['algorithm', 'comm_range']).agg({
        'final_performance': ['mean', 'std', 'median', 'min', 'max', 'count'],
        'training_time': 'mean'
    }).reset_index()
    
    # flatten column names
    summary.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                      for col in summary.columns.values]
    
    # compute coefficient of variation
    summary['cv'] = (summary['final_performance_std'] / 
                     summary['final_performance_mean'] * 100)
    
    return summary


def pairwise_comparison_at_range(df: pd.DataFrame, comm_range: float) -> Dict:
    # perform statistical comparison between gnn and iql at a specific range
    # filter data for this range
    range_data = df[df['comm_range'] == comm_range]
    
    iql_data = range_data[range_data['algorithm'] == 'IQL']['final_performance'].values
    gnn_data = range_data[range_data['algorithm'] == 'GNN']['final_performance'].values
    
    if len(iql_data) == 0 or len(gnn_data) == 0:
        return None
    
    # mann-whitney u test
    u_stat, p_value = stats.mannwhitneyu(gnn_data, iql_data, alternative='two-sided')
    
    # cohen's d
    n1, n2 = len(iql_data), len(gnn_data)
    s1, s2 = np.std(iql_data, ddof=1), np.std(gnn_data, ddof=1)
    pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    cohens_d = (np.mean(gnn_data) - np.mean(iql_data)) / pooled_std if pooled_std > 0 else 0
    
    # win rate
    wins = int(np.sum(gnn_data > iql_data))
    win_rate = wins / len(gnn_data)
    binom_test = stats.binomtest(wins, len(gnn_data), 0.5)
    
    # confidence interval for mean difference
    mean_diff = np.mean(gnn_data) - np.mean(iql_data)
    se_diff = np.sqrt(s1**2/n1 + s2**2/n2)
    df_welch = ((s1**2/n1 + s2**2/n2)**2) / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
    t_crit = stats.t.ppf(0.975, df_welch)
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff
    
    return {
        'comm_range': comm_range,
        'iql_mean': np.mean(iql_data),
        'iql_std': s1,
        'gnn_mean': np.mean(gnn_data),
        'gnn_std': s2,
        'mean_difference': mean_diff,
        'percent_improvement': (mean_diff / np.mean(iql_data) * 100) if np.mean(iql_data) > 0 else 0,
        'mann_whitney_u': u_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'cohens_d': cohens_d,
        'effect_size_category': 'large' if abs(cohens_d) >= 0.8 else ('medium' if abs(cohens_d) >= 0.5 else 'small'),
        'win_rate': win_rate,
        'wins': wins,
        'total': len(gnn_data),
        'binomial_p': binom_test.pvalue,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }


def analyze_observability_effect(df: pd.DataFrame) -> Dict:
    # analyze how observability affects the performance gap between gnn and iql
    ranges = sorted(df['comm_range'].unique())
    comparisons = []
    
    for comm_range in ranges:
        result = pairwise_comparison_at_range(df, comm_range)
        if result:
            comparisons.append(result)
    
    comp_df = pd.DataFrame(comparisons)
    
    # correlation between range and performance gap
    if len(comp_df) > 2:
        corr_coef, corr_p = stats.spearmanr(comp_df['comm_range'], 
                                            comp_df['mean_difference'])
    else:
        corr_coef, corr_p = np.nan, np.nan
    
    return {
        'comparisons': comp_df,
        'correlation_coefficient': corr_coef,
        'correlation_p_value': corr_p,
        'ranges_tested': len(ranges),
        'significant_ranges': int(comp_df['significant'].sum()),
        'max_advantage_range': comp_df.loc[comp_df['mean_difference'].idxmax(), 'comm_range'],
        'max_advantage_value': comp_df['mean_difference'].max()
    }


def print_analysis_report(df: pd.DataFrame, analysis: Dict):
    # print comprehensive analysis report
    print("\n" + "="*80)
    print("OBSERVABILITY SWEEP ANALYSIS REPORT")
    print("="*80)
    
    comp_df = analysis['comparisons']
    
    print(f"\nTotal Configurations Tested: {analysis['ranges_tested']}")
    print(f"Communication Ranges: {sorted(df['comm_range'].unique())}")
    print(f"Runs per configuration: {df.groupby(['algorithm', 'comm_range']).size().iloc[0]}")
    
    print("\n" + "-"*80)
    print("PERFORMANCE ACROSS COMMUNICATION RANGES")
    print("-"*80)
    
    for _, row in comp_df.iterrows():
        print(f"\nRange: {row['comm_range']:.1f}")
        print(f"  IQL:  {row['iql_mean']:.2f} ± {row['iql_std']:.2f}")
        print(f"  GNN:  {row['gnn_mean']:.2f} ± {row['gnn_std']:.2f}")
        print(f"  Diff: {row['mean_difference']:+.2f} ({row['percent_improvement']:+.1f}%)")
        print(f"  Stats: U={row['mann_whitney_u']:.0f}, p={row['p_value']:.4f} {'***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else 'ns'}")
        print(f"  Effect: Cohen's d={row['cohens_d']:.3f} ({row['effect_size_category']})")
        print(f"  Win Rate: {row['wins']}/{row['total']} ({row['win_rate']*100:.1f}%)")
    
    print("\n" + "-"*80)
    print("OBSERVABILITY EFFECT ANALYSIS")
    print("-"*80)
    
    print(f"\nCorrelation (Range vs GNN Advantage):")
    print(f"  Spearman ρ = {analysis['correlation_coefficient']:.3f}")
    print(f"  p-value = {analysis['correlation_p_value']:.4f}")
    
    if analysis['correlation_p_value'] < 0.05:
        if analysis['correlation_coefficient'] > 0:
            print(f"  → GNN advantage INCREASES with communication range")
        else:
            print(f"  → GNN advantage DECREASES with communication range")
    else:
        print(f"  → No significant correlation detected")
    
    print(f"\nMaximum GNN Advantage:")
    print(f"  Range: {analysis['max_advantage_range']:.1f}")
    print(f"  Advantage: {analysis['max_advantage_value']:.2f} points")
    
    print(f"\nStatistical Significance:")
    print(f"  Significant ranges: {analysis['significant_ranges']}/{analysis['ranges_tested']}")
    print(f"  Proportion: {analysis['significant_ranges']/analysis['ranges_tested']*100:.1f}%")
    
    # identify optimal range for each algorithm
    summary = df.groupby(['algorithm', 'comm_range'])['final_performance'].mean().reset_index()
    for alg in ['IQL', 'GNN']:
        alg_data = summary[summary['algorithm'] == alg]
        best_range = alg_data.loc[alg_data['final_performance'].idxmax(), 'comm_range']
        best_perf = alg_data['final_performance'].max()
        print(f"\n{alg} Optimal Range: {best_range:.1f} (Performance: {best_perf:.2f})")


def create_visualizations(df: pd.DataFrame, analysis: Dict, output_dir: Path):
    # create comprehensive visualizations
    output_dir.mkdir(parents=True, exist_ok=True)
    comp_df = analysis['comparisons']
    
    # set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    
    # 1. performance vs communication range
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for alg in ['IQL', 'GNN']:
        alg_data = df[df['algorithm'] == alg].groupby('comm_range')['final_performance'].agg(['mean', 'std']).reset_index()
        ax.plot(alg_data['comm_range'], alg_data['mean'], 'o-', 
               label=alg, linewidth=2, markersize=8)
        ax.fill_between(alg_data['comm_range'], 
                       alg_data['mean'] - alg_data['std'],
                       alg_data['mean'] + alg_data['std'],
                       alpha=0.2)
    
    ax.set_xlabel('Communication Range', fontsize=12, fontweight='bold')
    ax.set_ylabel('Final Performance', fontsize=12, fontweight='bold')
    ax.set_title('Performance vs Communication Range: GNN vs IQL', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_vs_range.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'performance_vs_range.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. performance difference (gnn - iql)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['green' if sig else 'gray' for sig in comp_df['significant']]
    bars = ax.bar(comp_df['comm_range'], comp_df['mean_difference'], 
                  color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # add error bars (confidence intervals)
    errors = [(row['mean_difference'] - row['ci_lower'], 
              row['ci_upper'] - row['mean_difference']) 
             for _, row in comp_df.iterrows()]
    ax.errorbar(comp_df['comm_range'], comp_df['mean_difference'],
               yerr=np.array(errors).T, fmt='none', color='black', 
               capsize=5, linewidth=2)
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='No difference')
    ax.set_xlabel('Communication Range', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance Difference (GNN - IQL)', fontsize=12, fontweight='bold')
    ax.set_title('GNN Advantage Across Communication Ranges', 
                fontsize=14, fontweight='bold')
    
    # add significance markers
    for i, (_, row) in enumerate(comp_df.iterrows()):
        if row['p_value'] < 0.001:
            marker = '***'
        elif row['p_value'] < 0.01:
            marker = '**'
        elif row['p_value'] < 0.05:
            marker = '*'
        else:
            marker = 'ns'
        
        y_pos = row['mean_difference'] + (row['ci_upper'] - row['mean_difference']) + 50
        ax.text(row['comm_range'], y_pos, marker, ha='center', fontsize=10, fontweight='bold')
    
    # legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Significant (p < 0.05)'),
        Patch(facecolor='gray', alpha=0.7, label='Not significant')
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_difference.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'performance_difference.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. effect size (cohen's d) across ranges
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(comp_df['comm_range'], comp_df['cohens_d'], 'o-', 
           linewidth=2, markersize=10, color='purple')
    ax.axhline(y=0.8, color='red', linestyle='--', linewidth=1.5, label='Large effect (d=0.8)')
    ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1.5, label='Medium effect (d=0.5)')
    ax.axhline(y=0.2, color='yellow', linestyle='--', linewidth=1.5, label='Small effect (d=0.2)')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    ax.set_xlabel('Communication Range', fontsize=12, fontweight='bold')
    ax.set_ylabel("Cohen's d (Effect Size)", fontsize=12, fontweight='bold')
    ax.set_title("Effect Size of GNN Advantage Across Communication Ranges", 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'effect_size_vs_range.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'effect_size_vs_range.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. box plots for each range
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
    axes = axes.flatten()
    
    ranges = sorted(df['comm_range'].unique())
    for idx, comm_range in enumerate(ranges):
        ax = axes[idx]
        range_data = df[df['comm_range'] == comm_range]
        
        data_to_plot = [
            range_data[range_data['algorithm'] == 'IQL']['final_performance'].values,
            range_data[range_data['algorithm'] == 'GNN']['final_performance'].values
        ]
        
        bp = ax.boxplot(data_to_plot, labels=['IQL', 'GNN'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightgreen')
        
        ax.set_title(f'Range: {comm_range:.1f}', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        if idx % 5 == 0:
            ax.set_ylabel('Final Performance', fontweight='bold')
    
    plt.suptitle('Performance Distribution Across Communication Ranges', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'boxplots_by_range.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'boxplots_by_range.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualizations saved to: {output_dir}")


def save_analysis_report(df: pd.DataFrame, analysis: Dict, output_file: Path):
    # save detailed analysis report to json
    # convert summary statistics to serializable format
    summary_stats = df.groupby(['algorithm', 'comm_range']).agg({
        'final_performance': ['mean', 'std', 'median', 'min', 'max']
    }).reset_index()
    
    # flatten multi-index columns
    summary_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                             for col in summary_stats.columns.values]
    
    report = {
        'summary_statistics': summary_stats.to_dict('records'),
        'pairwise_comparisons': analysis['comparisons'].to_dict('records'),
        'observability_effect': {
            'correlation_coefficient': float(analysis['correlation_coefficient']) if not np.isnan(analysis['correlation_coefficient']) else None,
            'correlation_p_value': float(analysis['correlation_p_value']) if not np.isnan(analysis['correlation_p_value']) else None,
            'max_advantage_range': float(analysis['max_advantage_range']),
            'max_advantage_value': float(analysis['max_advantage_value']),
            'significant_ranges': int(analysis['significant_ranges']),
            'total_ranges': int(analysis['ranges_tested'])
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nAnalysis report saved to: {output_file}")


def main():
    # main entry point
    if len(sys.argv) < 2:
        print("Usage: python analyze_observability_sweep.py <results_file.json>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    if not Path(results_file).exists():
        print(f"Error: Results file not found: {results_file}")
        sys.exit(1)
    
    print("Loading results...")
    results = load_results(results_file)
    
    print("Organizing data...")
    df = organize_data(results)
    
    print("Computing statistics...")
    summary = compute_summary_statistics(df)
    
    print("Analyzing observability effect...")
    analysis = analyze_observability_effect(df)
    
    # print report
    print_analysis_report(df, analysis)
    
    # create visualizations
    output_dir = Path(results_file).parent / 'visualizations'
    print("\nGenerating visualizations...")
    create_visualizations(df, analysis, output_dir)
    
    # save analysis report
    report_file = Path(results_file).parent / 'observability_analysis_report.json'
    save_analysis_report(df, analysis, report_file)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
