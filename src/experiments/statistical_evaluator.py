import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import json
import os
from tqdm import tqdm


@dataclass
class ExperimentResult:
    """Container for experiment results"""
    algorithm_name: str
    task_name: str
    num_agents: int
    episode_rewards: List[float]
    episode_lengths: List[int]
    success_rates: List[float]
    convergence_episodes: Optional[int] = None
    final_performance: Optional[float] = None
    training_time: Optional[float] = None


class StatisticalEvaluator:
    """Statistical evaluation framework for comparing multi-agent learning algorithms"""
    
    def __init__(self, significance_level: float = 0.05, 
                 effect_size_threshold: float = 0.5):
        self.significance_level = significance_level
        self.effect_size_threshold = effect_size_threshold
        self.results: List[ExperimentResult] = []
        
    def add_result(self, result: ExperimentResult):
        """Add experiment result"""
        self.results.append(result)
    
    def compute_performance_metrics(self, result: ExperimentResult) -> Dict[str, float]:
        """Compute comprehensive performance metrics"""
        rewards = np.array(result.episode_rewards)
        lengths = np.array(result.episode_lengths)
        success_rates = np.array(result.success_rates)
        
        metrics = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'median_reward': np.median(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'mean_episode_length': np.mean(lengths),
            'std_episode_length': np.std(lengths),
            'mean_success_rate': np.mean(success_rates),
            'std_success_rate': np.std(success_rates),
            'final_performance': result.final_performance or np.mean(rewards[-10:]),
            'convergence_episode': result.convergence_episodes or len(rewards),
            'stability': 1.0 / (1.0 + np.std(rewards[-20:]) / np.abs(np.mean(rewards[-20:]) + 1e-6))
        }
        
        return metrics
    
    def pairwise_comparison(self, result1: ExperimentResult, 
                          result2: ExperimentResult) -> Dict[str, Any]:
        """Perform statistical comparison between two algorithms"""
        rewards1 = np.array(result1.episode_rewards)
        rewards2 = np.array(result2.episode_rewards)
        
        # primary test: Mann-Whitney U (non-parametric, robust to non-normality)
        test_type = "Mann-Whitney U"
        statistic, p_value = stats.mannwhitneyu(rewards1, rewards2, alternative='two-sided')
        
        # rank-biserial correlation as effect size for Mann-Whitney
        n1, n2 = len(rewards1), len(rewards2)
        rank_biserial = 1 - (2 * statistic) / (n1 * n2)
        
        # also compute Cohen's d for interpretability
        pooled_std = np.sqrt(((n1 - 1) * np.var(rewards1, ddof=1) + 
                             (n2 - 1) * np.var(rewards2, ddof=1)) / 
                            (n1 + n2 - 2))
        effect_size = (np.mean(rewards1) - np.mean(rewards2)) / pooled_std if pooled_std > 0 else 0
        
        # confidence interval for mean difference (Welch–Satterthwaite df)
        mean_diff = np.mean(rewards1) - np.mean(rewards2)
        s1_sq_n = np.var(rewards1, ddof=1) / len(rewards1)
        s2_sq_n = np.var(rewards2, ddof=1) / len(rewards2)
        se_diff = np.sqrt(s1_sq_n + s2_sq_n)
        # welch–Satterthwaite degrees of freedom
        df_welch = (s1_sq_n + s2_sq_n)**2 / (
            s1_sq_n**2 / (len(rewards1) - 1) + s2_sq_n**2 / (len(rewards2) - 1)
        ) if (s1_sq_n + s2_sq_n) > 0 else max(len(rewards1), len(rewards2)) - 1
        t_crit = stats.t.ppf(0.975, df_welch)
        ci_lower = mean_diff - t_crit * se_diff
        ci_upper = mean_diff + t_crit * se_diff
        
        return {
            'algorithm1': result1.algorithm_name,
            'algorithm2': result2.algorithm_name,
            'test_type': test_type,
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': effect_size,
            'rank_biserial': rank_biserial,
            'significant': p_value < self.significance_level,
            'large_effect': abs(effect_size) > self.effect_size_threshold,
            'mean_difference': mean_diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'algorithm1_better': mean_diff > 0
        }
    
    def multiple_comparison(self) -> pd.DataFrame:
        """Perform multiple comparisons between all algorithms"""
        comparisons = []
        algorithms = list(set([r.algorithm_name for r in self.results]))
        
        for i, alg1 in enumerate(algorithms):
            for alg2 in algorithms[i+1:]:
                # get results for each algorithm
                results1 = [r for r in self.results if r.algorithm_name == alg1]
                results2 = [r for r in self.results if r.algorithm_name == alg2]
                
                if results1 and results2:
                    # use run-level summaries to avoid pseudoreplication
                    # each run contributes one data point (final performance or mean reward)
                    run_summaries1 = [
                        r1.final_performance if r1.final_performance is not None 
                        else np.mean(r1.episode_rewards) for r1 in results1
                    ]
                    run_summaries2 = [
                        r2.final_performance if r2.final_performance is not None 
                        else np.mean(r2.episode_rewards) for r2 in results2
                    ]
                    
                    # create temporary results with run-level summaries
                    temp_result1 = ExperimentResult(
                        alg1, results1[0].task_name, results1[0].num_agents,
                        run_summaries1, [1] * len(run_summaries1), [0.0] * len(run_summaries1)
                    )
                    temp_result2 = ExperimentResult(
                        alg2, results2[0].task_name, results2[0].num_agents,
                        run_summaries2, [1] * len(run_summaries2), [0.0] * len(run_summaries2)
                    )
                    
                    comparison = self.pairwise_comparison(temp_result1, temp_result2)
                    comparisons.append(comparison)
        
        return pd.DataFrame(comparisons)
    
    def scalability_analysis(self) -> Dict[str, pd.DataFrame]:
        """Analyze scalability across different numbers of agents"""
        scalability_data = {}
        
        # group by algorithm and number of agents
        for result in self.results:
            key = f"{result.algorithm_name}_{result.num_agents}_agents"
            if key not in scalability_data:
                scalability_data[key] = []
            scalability_data[key].append(result)
        
        # create DataFrames for analysis
        analysis_results = {}
        
        for algorithm in set([r.algorithm_name for r in self.results]):
            algorithm_data = []
            
            for result in self.results:
                if result.algorithm_name == algorithm:
                    metrics = self.compute_performance_metrics(result)
                    metrics['num_agents'] = result.num_agents
                    metrics['task_name'] = result.task_name
                    algorithm_data.append(metrics)
            
            if algorithm_data:
                df = pd.DataFrame(algorithm_data)
                analysis_results[algorithm] = df
        
        return analysis_results
    
    def convergence_analysis(self) -> Dict[str, Any]:
        """Analyze convergence properties"""
        convergence_data = {}
        
        for result in self.results:
            if result.algorithm_name not in convergence_data:
                convergence_data[result.algorithm_name] = []
            
            # find convergence point (when performance stabilizes)
            rewards = np.array(result.episode_rewards)
            convergence_episode = self._find_convergence_point(rewards)
            
            convergence_data[result.algorithm_name].append({
                'num_agents': result.num_agents,
                'task_name': result.task_name,
                'convergence_episode': convergence_episode,
                'final_performance': np.mean(rewards[-10:]),
                'peak_performance': np.max(rewards),
                'performance_at_convergence': np.mean(rewards[convergence_episode:convergence_episode+10]) if convergence_episode < len(rewards) else np.mean(rewards[-10:])
            })
        
        # convert to DataFrames
        convergence_analysis = {}
        for algorithm, data in convergence_data.items():
            convergence_analysis[algorithm] = pd.DataFrame(data)
        
        return convergence_analysis
    
    def _find_convergence_point(self, rewards: np.ndarray, window_size: int = 20, 
                               threshold: float = 0.05) -> int:
        """Find the episode where performance converges"""
        if len(rewards) < window_size * 2:
            return len(rewards) - 1
        
        # calculate moving average and variance
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        moving_var = []
        
        for i in range(len(moving_avg)):
            start_idx = i
            end_idx = i + window_size
            if end_idx <= len(rewards):
                moving_var.append(np.var(rewards[start_idx:end_idx]))
        
        moving_var = np.array(moving_var)
        
        # find point where variance drops below threshold
        for i in range(len(moving_var) - window_size):
            recent_var = np.mean(moving_var[i:i+window_size])
            if recent_var < threshold * (np.max(rewards) - np.min(rewards) + 1e-6):
                return i + window_size
        
        return len(rewards) - 1
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        report = {
            'experiment_summary': {
                'total_results': len(self.results),
                'algorithms': list(set([r.algorithm_name for r in self.results])),
                'tasks': list(set([r.task_name for r in self.results])),
                'agent_counts': sorted(list(set([r.num_agents for r in self.results])))
            },
            'performance_metrics': {},
            'statistical_comparisons': {},
            'scalability_analysis': {},
            'convergence_analysis': {}
        }
        
        # performance metrics for each algorithm
        for algorithm in set([r.algorithm_name for r in self.results]):
            algorithm_results = [r for r in self.results if r.algorithm_name == algorithm]
            all_metrics = []
            
            for result in algorithm_results:
                metrics = self.compute_performance_metrics(result)
                all_metrics.append(metrics)
            
            if all_metrics:
                df = pd.DataFrame(all_metrics)
                report['performance_metrics'][algorithm] = {
                    'mean_reward_mean': df['mean_reward'].mean(),
                    'mean_reward_std': df['mean_reward'].std(),
                    'success_rate_mean': df['mean_success_rate'].mean(),
                    'convergence_mean': df['convergence_episode'].mean(),
                    'stability_mean': df['stability'].mean()
                }
        
        # statistical comparisons
        comparison_df = self.multiple_comparison()
        if not comparison_df.empty:
            report['statistical_comparisons'] = {
                'significant_comparisons': comparison_df[comparison_df['significant']].to_dict('records'),
                'large_effect_comparisons': comparison_df[comparison_df['large_effect']].to_dict('records'),
                'summary': {
                    'total_comparisons': len(comparison_df),
                    'significant_count': len(comparison_df[comparison_df['significant']]),
                    'large_effect_count': len(comparison_df[comparison_df['large_effect']])
                }
            }
        
        # scalability analysis
        scalability_data = self.scalability_analysis()
        report['scalability_analysis'] = {}
        for algorithm, df in scalability_data.items():
            report['scalability_analysis'][algorithm] = {
                'performance_vs_agents': df.groupby('num_agents')['mean_reward'].mean().to_dict(),
                'stability_vs_agents': df.groupby('num_agents')['stability'].mean().to_dict()
            }
        
        # convergence analysis
        convergence_data = self.convergence_analysis()
        report['convergence_analysis'] = {}
        for algorithm, df in convergence_data.items():
            report['convergence_analysis'][algorithm] = {
                'mean_convergence_episode': df['convergence_episode'].mean(),
                'mean_final_performance': df['final_performance'].mean(),
                'mean_peak_performance': df['peak_performance'].mean()
            }
        
        return report
    
    def save_results(self, filepath: str):
        """Save all results and analysis to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # generate comprehensive report
        report = self.generate_summary_report()
        
        # save raw results
        raw_results = []
        for result in self.results:
            raw_results.append({
                'algorithm_name': result.algorithm_name,
                'task_name': result.task_name,
                'num_agents': result.num_agents,
                'episode_rewards': result.episode_rewards,
                'episode_lengths': result.episode_lengths,
                'success_rates': result.success_rates,
                'convergence_episodes': result.convergence_episodes,
                'final_performance': result.final_performance,
                'training_time': result.training_time
            })
        
        # save to JSON
        save_data = {
            'metadata': {
                'significance_level': self.significance_level,
                'effect_size_threshold': self.effect_size_threshold,
                'timestamp': pd.Timestamp.now().isoformat()
            },
            'raw_results': raw_results,
            'analysis_report': report
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str):
        """Load results from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # load metadata
        self.significance_level = data['metadata']['significance_level']
        self.effect_size_threshold = data['metadata']['effect_size_threshold']
        
        # load raw results
        self.results = []
        for result_data in data['raw_results']:
            result = ExperimentResult(
                algorithm_name=result_data['algorithm_name'],
                task_name=result_data['task_name'],
                num_agents=result_data['num_agents'],
                episode_rewards=result_data['episode_rewards'],
                episode_lengths=result_data['episode_lengths'],
                success_rates=result_data['success_rates'],
                convergence_episodes=result_data.get('convergence_episodes'),
                final_performance=result_data.get('final_performance'),
                training_time=result_data.get('training_time')
            )
            self.results.append(result)
        
        print(f"Loaded {len(self.results)} results from {filepath}")


class Visualizer:
    """Visualization tools for experiment results"""
    
    @staticmethod
    def plot_performance_comparison(results: List[ExperimentResult], 
                                  save_path: Optional[str] = None):
        """Plot performance comparison between algorithms"""
        plt.figure(figsize=(12, 8))
        
        # group by algorithm
        algorithms = list(set([r.algorithm_name for r in results]))
        
        for algorithm in algorithms:
            algorithm_results = [r for r in results if r.algorithm_name == algorithm]
            
            # aggregate rewards across all runs
            all_rewards = []
            for result in algorithm_results:
                all_rewards.extend(result.episode_rewards)
            
            # plot performance curve
            if all_rewards:
                # smooth the curve using moving average
                window_size = min(50, len(all_rewards) // 10)
                if window_size > 1:
                    smoothed = np.convolve(all_rewards, np.ones(window_size)/window_size, mode='valid')
                    episodes = range(window_size - 1, len(all_rewards))
                    plt.plot(episodes, smoothed, label=algorithm, linewidth=2)
                else:
                    plt.plot(all_rewards, label=algorithm, linewidth=2)
        
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Performance Comparison Across Algorithms')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_scalability_analysis(scalability_data: Dict[str, pd.DataFrame], 
                                save_path: Optional[str] = None):
        """Plot scalability analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # performance vs number of agents
        for algorithm, df in scalability_data.items():
            agent_counts = sorted(df['num_agents'].unique())
            mean_rewards = [df[df['num_agents'] == n]['mean_reward'].mean() for n in agent_counts]
            ax1.plot(agent_counts, mean_rewards, 'o-', label=algorithm, linewidth=2, markersize=6)
        
        ax1.set_xlabel('Number of Agents')
        ax1.set_ylabel('Mean Reward')
        ax1.set_title('Scalability: Performance vs Agent Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # stability vs number of agents
        for algorithm, df in scalability_data.items():
            agent_counts = sorted(df['num_agents'].unique())
            mean_stability = [df[df['num_agents'] == n]['stability'].mean() for n in agent_counts]
            ax2.plot(agent_counts, mean_stability, 's-', label=algorithm, linewidth=2, markersize=6)
        
        ax2.set_xlabel('Number of Agents')
        ax2.set_ylabel('Stability')
        ax2.set_title('Scalability: Stability vs Agent Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_statistical_significance(comparison_df: pd.DataFrame, 
                                    save_path: Optional[str] = None):
        """Plot statistical significance matrix"""
        if comparison_df.empty:
            print("No comparison data available")
            return
        
        # create significance matrix
        algorithms = sorted(set(comparison_df['algorithm1'].tolist() + comparison_df['algorithm2'].tolist()))
        n_algorithms = len(algorithms)
        
        significance_matrix = np.zeros((n_algorithms, n_algorithms))
        effect_size_matrix = np.zeros((n_algorithms, n_algorithms))
        
        for _, row in comparison_df.iterrows():
            i = algorithms.index(row['algorithm1'])
            j = algorithms.index(row['algorithm2'])
            
            if row['significant']:
                significance_matrix[i, j] = 1
                significance_matrix[j, i] = 1
            
            effect_size_matrix[i, j] = row['effect_size']
            effect_size_matrix[j, i] = -row['effect_size']  # negative for opposite direction
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # significance heatmap
        sns.heatmap(significance_matrix, xticklabels=algorithms, yticklabels=algorithms,
                   annot=True, cmap='RdYlBu_r', ax=ax1, cbar_kws={'label': 'Significant'})
        ax1.set_title('Statistical Significance Matrix')
        
        # effect size heatmap
        sns.heatmap(effect_size_matrix, xticklabels=algorithms, yticklabels=algorithms,
                   annot=True, cmap='RdBu_r', center=0, ax=ax2, cbar_kws={'label': 'Effect Size'})
        ax2.set_title('Effect Size Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
