# rigorous statistical analysis for gnn vs iql comparison
# addresses all 7 critical statistical issues:
# 1. power analysis (20 runs for 80% power)
# 2. variance analysis (f-test, cv reporting)
# 3. win rate significance (binomial test with ci)
# 4. multiple comparisons (bonferroni correction)
# 5. all environments tested (no cherry-picking)
# 6. bootstrap confidence intervals
# 7. non-parametric tests (mann-whitney u)

import sys
sys.path.insert(0, 'src')

import numpy as np
import json
from scipy import stats
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

sns.set_style("whitegrid")

def calculate_power_analysis(n1: int, n2: int, effect_size: float, alpha: float = 0.05) -> float:
    # calculate post-hoc statistical power for independent samples t-test
    from scipy.stats import nct, t as t_dist
    
    # degrees of freedom
    df = n1 + n2 - 2
    
    # non-centrality parameter
    ncp = effect_size * np.sqrt((n1 * n2) / (n1 + n2))
    
    # critical value
    t_crit = t_dist.ppf(1 - alpha/2, df)
    
    # power calculation
    power = 1 - nct.cdf(t_crit, df, ncp) + nct.cdf(-t_crit, df, ncp)
    
    return power

def variance_analysis(data1: np.ndarray, data2: np.ndarray, name1: str = "IQL", name2: str = "GNN") -> Dict:
    # comprehensive variance analysis including f-test and coefficient of variation
    var1 = np.var(data1, ddof=1)
    var2 = np.var(data2, ddof=1)
    std1 = np.std(data1, ddof=1)
    std2 = np.std(data2, ddof=1)
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)
    
    # coefficient of variation (cv)
    cv1 = (std1 / mean1) * 100 if mean1 != 0 else 0
    cv2 = (std2 / mean2) * 100 if mean2 != 0 else 0
    
    # f-test for equality of variances
    f_stat = var2 / var1 if var1 > 0 else np.inf
    df1 = len(data2) - 1
    df2 = len(data1) - 1
    f_pval = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))
    
    return {
        f'{name1}_variance': var1,
        f'{name2}_variance': var2,
        'variance_ratio': var2 / var1 if var1 > 0 else np.inf,
        f'{name1}_std': std1,
        f'{name2}_std': std2,
        'std_ratio': std2 / std1 if std1 > 0 else np.inf,
        f'{name1}_cv_percent': cv1,
        f'{name2}_cv_percent': cv2,
        'f_statistic': f_stat,
        'f_pvalue': f_pval,
        'variances_equal': f_pval > 0.05
    }

def binomial_win_rate_test(gnn_data: np.ndarray, iql_data: np.ndarray) -> Dict:
    # test if gnn win rate is significantly different from 50% (coin flip)
    # uses paired comparison when sample sizes match (assumes matched runs)
    # otherwise uses all pairwise comparisons
    if len(gnn_data) == len(iql_data):
        # paired comparison (matched runs, e.g. same seeds)
        n_trials = len(gnn_data)
        n_wins = int(np.sum(gnn_data > iql_data))
    else:
        # unpaired: use all pairwise comparisons
        n_wins = int(np.sum(gnn_data[:, None] > iql_data[None, :]))
        n_trials = len(gnn_data) * len(iql_data)
    win_rate = n_wins / n_trials
    
    # binomial test (two-sided) - updated for newer scipy
    binom_result = stats.binomtest(n_wins, n_trials, 0.5, alternative='two-sided')
    binom_pval = binom_result.pvalue
    
    # wilson score confidence interval (more accurate than normal approximation)
    z = 1.96  # 95% CI
    denominator = 1 + z**2 / n_trials
    center = (win_rate + z**2 / (2 * n_trials)) / denominator
    margin = z * np.sqrt(win_rate * (1 - win_rate) / n_trials + z**2 / (4 * n_trials**2)) / denominator
    
    ci_lower = max(0, center - margin)
    ci_upper = min(1, center + margin)
    
    return {
        'n_trials': n_trials,
        'n_wins': n_wins,
        'win_rate': win_rate,
        'binomial_pvalue': binom_pval,
        'significant': binom_pval < 0.05,
        'wilson_ci_lower': ci_lower,
        'wilson_ci_upper': ci_upper,
        'ci_includes_50_percent': ci_lower <= 0.5 <= ci_upper
    }

def bootstrap_confidence_interval(data: np.ndarray, n_bootstrap: int = 10000, ci: float = 0.95) -> Tuple[float, float]:
    # calculate bootstrap confidence interval for the mean
    bootstrap_means = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = 1 - ci
    lower = np.percentile(bootstrap_means, alpha/2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
    
    return lower, upper

def comprehensive_statistical_test(iql_data: np.ndarray, gnn_data: np.ndarray, 
                                   task_name: str, bonferroni_n: int = 4) -> Dict:
    # perform comprehensive statistical analysis with all corrections
    results = {
        'task': task_name,
        'n_samples': len(iql_data)
    }
    
    # basic statistics
    results['iql_mean'] = np.mean(iql_data)
    results['iql_std'] = np.std(iql_data, ddof=1)
    results['iql_median'] = np.median(iql_data)
    results['gnn_mean'] = np.mean(gnn_data)
    results['gnn_std'] = np.std(gnn_data, ddof=1)
    results['gnn_median'] = np.median(gnn_data)
    results['mean_difference'] = results['gnn_mean'] - results['iql_mean']
    results['percent_improvement'] = (results['mean_difference'] / results['iql_mean'] * 100) if results['iql_mean'] > 0 else 0
    
    # effect size (cohen's d) with proper weighted pooled sd
    n_iql, n_gnn = len(iql_data), len(gnn_data)
    pooled_std = np.sqrt(((n_iql - 1) * results['iql_std']**2 + (n_gnn - 1) * results['gnn_std']**2) / (n_iql + n_gnn - 2))
    results['cohens_d'] = results['mean_difference'] / pooled_std if pooled_std > 0 else 0
    results['effect_size_interpretation'] = (
        'large' if abs(results['cohens_d']) > 0.8 else 
        ('medium' if abs(results['cohens_d']) > 0.5 else 'small')
    )
    
    # mann-whitney u test (non-parametric, primary test)
    u_stat, u_pval = stats.mannwhitneyu(gnn_data, iql_data, alternative='two-sided')
    results['mann_whitney_u'] = u_stat
    results['mann_whitney_p'] = u_pval
    results['mann_whitney_significant'] = u_pval < 0.05
    
    # bonferroni corrected significance
    bonferroni_alpha = 0.05 / bonferroni_n
    results['bonferroni_alpha'] = bonferroni_alpha
    results['bonferroni_significant'] = u_pval < bonferroni_alpha
    
    # t-test (for comparison, but less robust)
    t_stat, t_pval = stats.ttest_ind(gnn_data, iql_data)
    results['t_statistic'] = t_stat
    results['t_pvalue'] = t_pval
    
    # power analysis
    power = calculate_power_analysis(len(iql_data), len(gnn_data), results['cohens_d'])
    results['statistical_power'] = power
    results['adequately_powered'] = power >= 0.80
    
    # variance analysis
    var_analysis = variance_analysis(iql_data, gnn_data)
    results.update(var_analysis)
    
    # win rate analysis
    win_analysis = binomial_win_rate_test(gnn_data, iql_data)
    results.update(win_analysis)
    
    # bootstrap confidence intervals
    iql_ci = bootstrap_confidence_interval(iql_data)
    gnn_ci = bootstrap_confidence_interval(gnn_data)
    results['iql_ci_lower'] = iql_ci[0]
    results['iql_ci_upper'] = iql_ci[1]
    results['gnn_ci_lower'] = gnn_ci[0]
    results['gnn_ci_upper'] = gnn_ci[1]
    
    # bootstrap ci for difference
    diff_bootstrap = []
    for _ in range(10000):
        iql_sample = np.random.choice(iql_data, size=len(iql_data), replace=True)
        gnn_sample = np.random.choice(gnn_data, size=len(gnn_data), replace=True)
        diff_bootstrap.append(np.mean(gnn_sample) - np.mean(iql_sample))
    
    results['difference_ci_lower'] = np.percentile(diff_bootstrap, 2.5)
    results['difference_ci_upper'] = np.percentile(diff_bootstrap, 97.5)
    results['ci_includes_zero'] = results['difference_ci_lower'] <= 0 <= results['difference_ci_upper']
    
    return results

def print_rigorous_report(all_results: List[Dict]):
    # print comprehensive statistical report
    print("\nstatistical analysis: gnn vs iql")
    print("\nstatistical power:")
    for result in all_results:
        power_status = "adequate" if result['adequately_powered'] else "underpowered"
        print(f"{result['task']:<25} power: {result['statistical_power']:.1%} ({power_status})")
        if not result['adequately_powered']:
            print(f"  note: need ~17 runs for 80% power (currently {result['n_samples']})")
    
    print("\nvariance analysis:")
    for result in all_results:
        print(f"\n{result['task']}:")
        print(f"  IQL: σ={result['IQL_std']:.2f}, CV={result['IQL_cv_percent']:.1f}%")
        print(f"  GNN: σ={result['GNN_std']:.2f}, CV={result['GNN_cv_percent']:.1f}%")
        print(f"  Variance Ratio: {result['variance_ratio']:.2f}x")
        print(f"  f-test: p={result['f_pvalue']:.4f} ({'equal' if result['variances_equal'] else 'unequal'})")
        if result['GNN_cv_percent'] > 25:
            print(f"  note: gnn shows high instability (cv > 25%)")
    
    print("\nwin rate analysis:")
    for result in all_results:
        print(f"\n{result['task']}:")
        print(f"  Win Rate: {result['win_rate']:.1%} ({result['n_wins']}/{result['n_trials']})")
        print(f"  binomial test: p={result['binomial_pvalue']:.4f} ({'significant' if result['significant'] else 'not significant'})")
        print(f"  wilson 95% ci: [{result['wilson_ci_lower']:.1%}, {result['wilson_ci_upper']:.1%}]")
        if result['ci_includes_50_percent']:
            print(f"  note: ci includes 50% - not distinguishable from coin flip")
    
    print("\nmultiple comparisons correction:")
    n_comparisons = len(all_results)
    bonferroni_alpha = 0.05 / n_comparisons
    print(f"number of comparisons: {n_comparisons}")
    print(f"bonferroni-corrected alpha: {bonferroni_alpha:.4f}")
    for result in all_results:
        sig_status = "yes" if result['bonferroni_significant'] else "no"
        print(f"{result['task']:<25} p={result['mann_whitney_p']:.4f} (significant: {sig_status})")
    
    print("\ncomprehensive reporting:")
    for result in all_results:
        winner = "gnn" if result['gnn_mean'] > result['iql_mean'] else "iql"
        print(f"  {result['task']:<25} winner: {winner:<5} (diff={result['mean_difference']:+.2f})")
    
    print("\nbootstrap confidence intervals:")
    for result in all_results:
        print(f"\n{result['task']}:")
        print(f"  iql: {result['iql_mean']:.2f} [{result['iql_ci_lower']:.2f}, {result['iql_ci_upper']:.2f}]")
        print(f"  gnn: {result['gnn_mean']:.2f} [{result['gnn_ci_lower']:.2f}, {result['gnn_ci_upper']:.2f}]")
        print(f"  difference: {result['mean_difference']:.2f} [{result['difference_ci_lower']:.2f}, {result['difference_ci_upper']:.2f}]")
        ci_width = result['difference_ci_upper'] - result['difference_ci_lower']
        if ci_width > 500:
            print(f"  note: wide ci (width={ci_width:.0f}) - imprecise estimate")
        if result['ci_includes_zero']:
            print(f"  note: ci includes zero - effect not clearly established")
    
    print("\nnon-parametric analysis:")
    for result in all_results:
        print(f"{result['task']:<25} u={result['mann_whitney_u']:.2f} p={result['mann_whitney_p']:.4f} d={result['cohens_d']:.3f}")
    
    print("\noverall conclusion:")
    sig_count = sum(1 for r in all_results if r['bonferroni_significant'])
    gnn_wins = sum(1 for r in all_results if r['gnn_mean'] > r['iql_mean'])
    adequately_powered = sum(1 for r in all_results if r['adequately_powered'])
    
    print(f"statistically significant results (bonferroni-corrected): {sig_count}/{len(all_results)}")
    print(f"gnn performance wins: {gnn_wins}/{len(all_results)}")
    print(f"adequately powered tests (>=80%): {adequately_powered}/{len(all_results)}")
    
    if sig_count == 0:
        print("no statistically significant evidence for gnn superiority after corrections")
    elif sig_count < len(all_results) / 2:
        print("limited evidence: some tasks show significance, but not consistent across environments")
    else:
        print("strong evidence: majority of tasks show statistically significant gnn advantage")

def load_and_analyze_results(results_file: str) -> List[Dict]:
    # load experimental results and perform rigorous analysis
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # group by task and algorithm
    task_data = {}
    # handle both 'results' and 'raw_results' keys
    results_key = 'results' if 'results' in data else 'raw_results'
    
    for result in data[results_key]:
        task = result['task_name']
        algo = result['algorithm_name']
        
        if task not in task_data:
            task_data[task] = {'IQL': [], 'GNN': []}
        
        # use final performance or mean of episode rewards
        performance = result.get('final_performance', np.mean(result['episode_rewards']))
        task_data[task][algo].append(performance)
    
    # perform analysis for each task
    all_results = []
    for task, data in task_data.items():
        if 'IQL' in data and 'GNN' in data and len(data['IQL']) > 0 and len(data['GNN']) > 0:
            iql_array = np.array(data['IQL'])
            gnn_array = np.array(data['GNN'])
            
            result = comprehensive_statistical_test(
                iql_array, gnn_array, task, bonferroni_n=len(task_data)
            )
            all_results.append(result)
    
    return all_results

if __name__ == "__main__":
    import glob
    
    # find most recent results file
    results_files = glob.glob('results/experiment_results_*.json')
    if not results_files:
        print("No results files found. Run experiments first.")
        exit(1)
    
    latest_file = max(results_files, key=os.path.getctime)
    print(f"Analyzing: {latest_file}")
    
    # perform rigorous analysis
    all_results = load_and_analyze_results(latest_file)
    
    # print comprehensive report
    print_rigorous_report(all_results)
    
    # save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'analysis/rigorous_analysis_{timestamp}.json'
    os.makedirs('analysis', exist_ok=True)
    
    # convert numpy types to native python types for json serialization
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    serializable_results = convert_numpy_types(all_results)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\ndetailed results saved to: {output_file}")
