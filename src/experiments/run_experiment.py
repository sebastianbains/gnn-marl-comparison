#!/usr/bin/env python3
# main experiment runner for comparing graph-based vs traditional multi-agent learning

import argparse
import time
import random
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm

# optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

# import our modules
from environments.coordination_env import MultiAgentCoordinationEnv, CoordinationTask
from agents.traditional.iql import MultiAgentIQL
from agents.graph_based.gnn_agent import MultiAgentGNN
from agents.graph_based.attention_gnn import AdvancedMultiAgentGNN
from experiments.statistical_evaluator import StatisticalEvaluator, ExperimentResult, Visualizer

class ExperimentRunner:
    # main experiment runner
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # set random seeds
        self.set_seeds(config.get('seed', 42))
        
        # initialize evaluator
        self.evaluator = StatisticalEvaluator(
            significance_level=config.get('significance_level', 0.05),
            effect_size_threshold=config.get('effect_size_threshold', 0.5)
        )
        
        # initialize wandb if enabled
        if config.get('use_wandb', False) and WANDB_AVAILABLE:
            wandb.init(
                project=config.get('wandb_project', 'multi-agent-gnn-comparison'),
                config=config,
                name=config.get('experiment_name', 'experiment')
            )
    
    def set_seeds(self, seed: int):
        # set random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def create_environment(self, num_agents: int, task_type: CoordinationTask) -> MultiAgentCoordinationEnv:
        # create coordination environment
        return MultiAgentCoordinationEnv(
            num_agents=num_agents,
            grid_size=self.config.get('grid_size', 10),
            task_type=task_type,
            max_steps=self.config.get('max_steps', 100),
            reward_type=self.config.get('reward_type', 'global'),
            observation_type=self.config.get('observation_type', 'local'),
            communication_range=self.config.get('communication_range', 3.0)
        )
    
    def create_agent(self, algorithm_name: str, num_agents: int, obs_dim: int, 
                    action_dim: int, state_dim: int = None) -> Any:
        # create learning agent based on algorithm name
        agent_config = self.config.get('agent_config', {})
        
        if algorithm_name == 'IQL':
            return MultiAgentIQL(
                num_agents=num_agents,
                obs_dim=obs_dim,
                action_dim=action_dim,
                lr=agent_config.get('lr', 1e-3),
                gamma=agent_config.get('gamma', 0.99),
                epsilon=agent_config.get('epsilon', 0.1)
            )
        
        elif algorithm_name == 'GNN':
            return MultiAgentGNN(
                num_agents=num_agents,
                obs_dim=obs_dim,
                action_dim=action_dim,
                lr=agent_config.get('lr', 1e-3),
                gamma=agent_config.get('gamma', 0.99),
                epsilon=agent_config.get('epsilon', 0.1),
                gnn_type=agent_config.get('gnn_type', 'gat'),
                hidden_dim=agent_config.get('hidden_dim', 64),
                communication_graph=agent_config.get('communication_graph', 'learnable')
            )
        
        elif algorithm_name == 'AttentionGNN':
            return AdvancedMultiAgentGNN(
                num_agents=num_agents,
                obs_dim=obs_dim,
                action_dim=action_dim,
                lr=agent_config.get('lr', 1e-3),
                gamma=agent_config.get('gamma', 0.99),
                epsilon=agent_config.get('epsilon', 0.1),
                hidden_dim=agent_config.get('hidden_dim', 64),
                num_heads=agent_config.get('num_heads', 8),
                adaptive_topology=agent_config.get('adaptive_topology', True)
            )
        
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    def train_agent(self, agent: Any, env: MultiAgentCoordinationEnv, algorithm_name: str,
                   num_episodes: int) -> Dict[str, List[float]]:
        """Train agent and return training metrics"""
        episode_rewards = []
        episode_lengths = []
        success_rates = []
        
        # experience replay buffer
        replay_buffer = []
        buffer_size = self.config.get('buffer_size', 10000)
        batch_size = self.config.get('batch_size', 32)
        
        # training loop
        for episode in tqdm(range(num_episodes), desc=f"Training {algorithm_name}"):
            obs, _ = env.reset()
            total_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                # select actions
                if hasattr(agent, 'select_actions'):
                    if algorithm_name == 'AttentionGNN':
                        actions, _ = agent.select_actions(obs, training=True)
                    else:
                        actions = agent.select_actions(obs, training=True)
                else:
                    # handle single agent methods
                    actions = []
                    for i in range(env.num_agents):
                        action = agent.agents[i].select_action(obs[i], training=True)
                        actions.append(action)
                
                # step environment
                next_obs, rewards, terminated, truncated, info = env.step(np.array(actions))
                done = terminated or truncated
                
                # store experience
                if algorithm_name == 'IQL':
                    # iQL: Store per-agent experiences
                    for i in range(env.num_agents):
                        replay_buffer.append((obs[i], actions[i], rewards[i], 
                                           next_obs[i], done))
                elif algorithm_name in ['GNN', 'AttentionGNN']:
                    # gNN: Store full multi-agent experience (all agents' data together)
                    replay_buffer.append((obs.copy(), actions.copy(), rewards.copy(), 
                                        next_obs.copy(), done))
                else:
                    # other algorithms: store basic experience
                    replay_buffer.append((obs.copy(), actions.copy(), rewards.copy(), 
                                        next_obs.copy(), done))
                
                # update agent
                if len(replay_buffer) >= batch_size:
                    batch = random.sample(replay_buffer, min(batch_size, len(replay_buffer)))
                    
                    if algorithm_name == 'IQL':
                        # update each agent separately
                        agent_batches = [[] for _ in range(env.num_agents)]
                        for experience in batch:
                            agent_idx = random.randint(0, env.num_agents - 1)
                            agent_batches[agent_idx].append(experience)
                        
                        losses = agent.update_agents(agent_batches)
                    elif algorithm_name in ['GNN', 'AttentionGNN']:
                        loss = agent.update(batch)
                
                obs = next_obs
                total_reward += sum(rewards)
                episode_length += 1
            
            # record metrics
            episode_rewards.append(total_reward)
            episode_lengths.append(episode_length)
            
            # calculate success rate
            if 'success_rate' in info:
                success_rates.append(info['success_rate'])
            else:
                # simple success metric: positive reward
                success_rates.append(1.0 if total_reward > 0 else 0.0)
            
            # update target networks
            if hasattr(agent, 'update_target_network'):
                if episode % 100 == 0:
                    agent.update_target_network()
            
            # decay epsilon
            if hasattr(agent, 'set_epsilon'):
                current_epsilon = agent.get_epsilon() if hasattr(agent, 'get_epsilon') else agent.epsilon
                epsilon = max(0.01, current_epsilon * 0.995)
                agent.set_epsilon(epsilon)
            
            # log to wandb
            if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
                wandb.log({
                    f'{algorithm_name}/episode_reward': total_reward,
                    f'{algorithm_name}/episode_length': episode_length,
                    f'{algorithm_name}/success_rate': success_rates[-1],
                    'episode': episode
                })
            
            # limit buffer size
            if len(replay_buffer) > buffer_size:
                replay_buffer = replay_buffer[-buffer_size:]
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'success_rates': success_rates
        }
    
    def evaluate_agent(self, agent: Any, env: MultiAgentCoordinationEnv, 
                      algorithm_name: str, num_episodes: int) -> Dict[str, List[float]]:
        """Evaluate trained agent"""
        episode_rewards = []
        episode_lengths = []
        success_rates = []
        
        # set evaluation mode
        if hasattr(agent, 'set_epsilon'):
            agent.set_epsilon(0.0)  # no exploration during evaluation
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            total_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                # select actions (no exploration)
                if hasattr(agent, 'select_actions'):
                    if algorithm_name == 'AttentionGNN':
                        actions, _ = agent.select_actions(obs, training=False)
                    else:
                        actions = agent.select_actions(obs, training=False)
                else:
                    actions = []
                    for i in range(env.num_agents):
                        action = agent.agents[i].select_action(obs[i], training=False)
                        actions.append(action)
                
                # step environment
                next_obs, rewards, terminated, truncated, info = env.step(np.array(actions))
                done = terminated or truncated
                
                obs = next_obs
                total_reward += sum(rewards)
                episode_length += 1
            
            episode_rewards.append(total_reward)
            episode_lengths.append(episode_length)
            
            if 'success_rate' in info:
                success_rates.append(info['success_rate'])
            else:
                success_rates.append(1.0 if total_reward > 0 else 0.0)
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'success_rates': success_rates
        }
    
    def run_single_experiment(self, algorithm_name: str, num_agents: int, 
                            task_type: CoordinationTask) -> ExperimentResult:
        """Run single experiment configuration"""
        print(f"\n{'='*50}")
        print(f"Running {algorithm_name} with {num_agents} agents on {task_type}")
        print(f"{'='*50}")
        
        # create environment
        env = self.create_environment(num_agents, task_type)
        
        # get dimensions
        obs_sample, _ = env.reset()
        obs_dim = obs_sample.shape[1]
        action_dim = 5  # 4 directions + stay
        state_dim = obs_sample.flatten().shape[0]
        
        # create agent
        agent = self.create_agent(algorithm_name, num_agents, obs_dim, action_dim, state_dim)
        
        # train agent
        start_time = time.time()
        training_metrics = self.train_agent(
            agent, env, algorithm_name, 
            self.config.get('training_episodes', 1000)
        )
        training_time = time.time() - start_time
        
        # evaluate agent
        eval_metrics = self.evaluate_agent(
            agent, env, algorithm_name,
            self.config.get('evaluation_episodes', 100)
        )
        
        # create result
        result = ExperimentResult(
            algorithm_name=algorithm_name,
            task_name=task_type.value,
            num_agents=num_agents,
            episode_rewards=eval_metrics['episode_rewards'],
            episode_lengths=eval_metrics['episode_lengths'],
            success_rates=eval_metrics['success_rates'],
            final_performance=np.mean(eval_metrics['episode_rewards'][-10:]),
            training_time=training_time
        )
        
        print(f"Final performance: {result.final_performance:.2f}")
        print(f"Training time: {training_time:.2f}s")
        
        return result
    
    def run_all_experiments(self):
        """Run all experiment configurations"""
        algorithms = self.config.get('algorithms', ['IQL', 'GNN', 'AttentionGNN'])
        agent_counts = self.config.get('agent_counts', [3, 5, 7, 10])
        tasks = [CoordinationTask[t] for t in self.config.get('tasks', ['RESOURCE_COLLECTION'])]
        num_runs = self.config.get('num_runs', 5)
        
        total_experiments = len(algorithms) * len(agent_counts) * len(tasks) * num_runs
        current_experiment = 0
        
        for algorithm in algorithms:
            for num_agents in agent_counts:
                for task in tasks:
                    for run in range(num_runs):
                        current_experiment += 1
                        print(f"\nProgress: {current_experiment}/{total_experiments}")
                        
                        result = self.run_single_experiment(algorithm, num_agents, task)
                        self.evaluator.add_result(result)
                        
                        # save intermediate results
                        if current_experiment % 10 == 0:
                            self.save_results()
        
        # final save
        self.save_results()
        
        # generate analysis
        self.generate_analysis()
    
    def save_results(self):
        """Save experiment results"""
        results_dir = Path(self.config.get('results_dir', 'results'))
        results_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_path = results_dir / f"experiment_results_{timestamp}.json"
        
        self.evaluator.save_results(str(results_path))
        
        if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
            wandb.save(str(results_path))
    
    def generate_analysis(self):
        """Generate and save analysis"""
        results_dir = Path(self.config.get('results_dir', 'results'))
        analysis_dir = Path('analysis')
        analysis_dir.mkdir(exist_ok=True)
        
        # generate summary report
        report = self.evaluator.generate_summary_report()
        
        # save report
        report_path = analysis_dir / "summary_report.json"
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # generate visualizations
        Visualizer.plot_performance_comparison(
            self.evaluator.results,
            save_path=str(analysis_dir / "performance_comparison.png")
        )
        
        scalability_data = self.evaluator.scalability_analysis()
        if scalability_data:
            Visualizer.plot_scalability_analysis(
                scalability_data,
                save_path=str(analysis_dir / "scalability_analysis.png")
            )
        
        comparison_df = self.evaluator.multiple_comparison()
        if not comparison_df.empty:
            Visualizer.plot_statistical_significance(
                comparison_df,
                save_path=str(analysis_dir / "statistical_significance.png")
            )
        
        print(f"\nAnalysis saved to {analysis_dir}")
        
        # print key findings
        self.print_key_findings(report)
    
    def print_key_findings(self, report: Dict[str, Any]):
        """Print key findings from the analysis"""
        print(f"\n{'='*60}")
        print("KEY FINDINGS")
        print(f"{'='*60}")
        
        # performance comparison
        print("\n1. PERFORMANCE COMPARISON:")
        performance = report['performance_metrics']
        best_algorithm = max(performance.keys(), 
                            key=lambda x: performance[x]['mean_reward_mean'])
        
        for alg, metrics in performance.items():
            status = "(best)" if alg == best_algorithm else ""
            print(f"   {alg}: {metrics['mean_reward_mean']:.2f} ± {metrics['mean_reward_std']:.2f} {status}")
        
        # statistical significance
        print("\n2. STATISTICAL SIGNIFICANCE:")
        if 'statistical_comparisons' in report:
            sig_comparisons = report['statistical_comparisons'].get('significant_comparisons', [])
            print(f"   {len(sig_comparisons)} statistically significant comparisons found")
            
            # check if GNN methods are significantly better
            gnn_better_count = 0
            for comp in sig_comparisons:
                if ('GNN' in comp['algorithm1'] and comp['algorithm1_better']) or \
                   ('GNN' in comp['algorithm2'] and not comp['algorithm1_better']):
                    gnn_better_count += 1
            
            if gnn_better_count > 0:
                print(f"   gnn-based methods show statistically significant improvement in {gnn_better_count} comparisons")
            else:
                print(f"   no statistically significant advantage for gnn methods found")
        
        # scalability
        print("\n3. SCALABILITY ANALYSIS:")
        scalability = report.get('scalability_analysis', {})
        for alg, data in scalability.items():
            agent_counts = list(data['performance_vs_agents'].keys())
            if len(agent_counts) > 1:
                performance_trend = data['performance_vs_agents']
                print(f"   {alg}: Performance trend across agent counts")
                for agents, perf in performance_trend.items():
                    print(f"     {agents} agents: {perf:.2f}")
        
        # convergence
        print("\n4. CONVERGENCE ANALYSIS:")
        convergence = report.get('convergence_analysis', {})
        for alg, data in convergence.items():
            print(f"   {alg}: {data['mean_convergence_episode']:.1f} episodes to converge")
        
        print("\nconclusion:")
        if best_algorithm and 'GNN' in best_algorithm:
            print("graph-based methods demonstrate superior performance in multi-agent coordination tasks")
        else:
            print("results vary by task and configuration - see detailed analysis for insights")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Run multi-agent learning comparison experiments')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--algorithms', nargs='+', 
                       choices=['IQL', 'GNN', 'AttentionGNN'],
                       help='Algorithms to compare')
    parser.add_argument('--agents', nargs='+', type=int,
                       help='Number of agents to test')
    parser.add_argument('--tasks', nargs='+',
                       choices=['RESOURCE_COLLECTION', 'TARGET_COVERAGE', 'FORMATION_CONTROL', 'COOPERATIVE_NAVIGATION'],
                       help='Tasks to test')
    parser.add_argument('--episodes', type=int,
                       help='Number of training episodes')
    parser.add_argument('--runs', type=int,
                       help='Number of experimental runs')
    
    args = parser.parse_args()
    
    # load configuration
    config = load_config(args.config)
    
    # override config with command line arguments
    if args.algorithms:
        config['algorithms'] = args.algorithms
    if args.agents:
        config['agent_counts'] = args.agents
    if args.tasks:
        config['tasks'] = args.tasks
    if args.episodes:
        config['training_episodes'] = args.episodes
    if args.runs:
        config['num_runs'] = args.runs
    
    # run experiments
    runner = ExperimentRunner(config)
    runner.run_all_experiments()


if __name__ == "__main__":
    main()
