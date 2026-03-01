import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Tuple, Dict, Any, Optional
import networkx as nx
from enum import Enum


class CoordinationTask(Enum):
    """Types of coordination tasks"""
    RESOURCE_COLLECTION = "resource_collection"
    TARGET_COVERAGE = "target_coverage"
    FORMATION_CONTROL = "formation_control"
    COOPERATIVE_NAVIGATION = "cooperative_navigation"


class MultiAgentCoordinationEnv(gym.Env):
    """Multi-agent coordination environment for testing learning algorithms"""
    
    def __init__(self, 
                 num_agents: int = 5,
                 grid_size: int = 10,
                 task_type: CoordinationTask = CoordinationTask.RESOURCE_COLLECTION,
                 max_steps: int = 100,
                 reward_type: str = "global",
                 observation_type: str = "local",
                 communication_range: float = 3.0,
                 seed: Optional[int] = None):
        
        super().__init__()
        
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.task_type = task_type
        self.max_steps = max_steps
        self.reward_type = reward_type  # "global", "local", "mixed"
        self.observation_type = observation_type  # "local", "global", "partial"
        self.communication_range = communication_range
        
        # action space: 4 directions + stay
        self.action_space = spaces.MultiDiscrete([5] * num_agents)
        
        # observation space depends on observation type
        if observation_type == "local":
            # local observation: agent position + nearby agents + nearby resources
            obs_dim = 2 + 4 * 4 + 4 * 4  # position + nearby agents + nearby resources
        elif observation_type == "global":
            # global observation: all positions and resources
            obs_dim = 2 * num_agents + 2 * 20  # agent positions + resource positions
        else:  # partial
            obs_dim = 2 + 8 * 4  # position + partial view
            
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(num_agents, obs_dim), dtype=np.float32
        )
        
        # initialize environment state
        self.reset(seed=seed)
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment"""
        if seed is not None:
            np.random.seed(seed)
            
        self.current_step = 0
        
        # initialize agent positions randomly
        self.agent_positions = np.random.uniform(
            0, self.grid_size, (self.num_agents, 2)
        )
        
        # initialize task-specific elements
        if self.task_type == CoordinationTask.RESOURCE_COLLECTION:
            self._init_resource_collection()
        elif self.task_type == CoordinationTask.TARGET_COVERAGE:
            self._init_target_coverage()
        elif self.task_type == CoordinationTask.FORMATION_CONTROL:
            self._init_formation_control()
        elif self.task_type == CoordinationTask.COOPERATIVE_NAVIGATION:
            self._init_cooperative_navigation()
        
        return self._get_observations(), self._get_info()
    
    def _init_resource_collection(self):
        """Initialize resource collection task"""
        # place resources randomly - make it harder with more resources
        self.num_resources = min(30, self.grid_size * self.grid_size // 3)  # more resources
        self.resource_positions = np.random.uniform(
            0, self.grid_size, (self.num_resources, 2)
        )
        # make resources have varying values - some more valuable than others
        self.resource_values = np.random.uniform(0.5, 3.0, self.num_resources)
        self.collected_resources = np.zeros(self.num_resources, dtype=bool)
        
        # add resource respawn probability
        self.respawn_probability = 0.1  # 10% chance of resource respawning
        
    def _init_target_coverage(self):
        """Initialize target coverage task"""
        # define target positions that need to be covered
        self.num_targets = min(10, self.num_agents * 2)
        self.target_positions = np.random.uniform(
            1, self.grid_size - 1, (self.num_targets, 2)
        )
        self.covered_targets = np.zeros(self.num_targets, dtype=bool)
        self.coverage_radius = 2.0
        
    def _init_formation_control(self):
        """Initialize formation control task"""
        # define target formation (e.g., circle, line, grid)
        formation_type = np.random.choice(['circle', 'line', 'grid'])
        
        if formation_type == 'circle':
            angles = np.linspace(0, 2 * np.pi, self.num_agents, endpoint=False)
            radius = min(self.grid_size / 4, 3)
            self.target_formation = np.column_stack([
                self.grid_size / 2 + radius * np.cos(angles),
                self.grid_size / 2 + radius * np.sin(angles)
            ])
        elif formation_type == 'line':
            x_positions = np.linspace(self.grid_size * 0.2, self.grid_size * 0.8, self.num_agents)
            self.target_formation = np.column_stack([
                x_positions,
                np.full(self.num_agents, self.grid_size / 2)
            ])
        else:  # grid
            grid_size = int(np.ceil(np.sqrt(self.num_agents)))
            positions = []
            for i in range(grid_size):
                for j in range(grid_size):
                    if len(positions) < self.num_agents:
                        x = self.grid_size * 0.2 + i * self.grid_size * 0.6 / (grid_size - 1)
                        y = self.grid_size * 0.2 + j * self.grid_size * 0.6 / (grid_size - 1)
                        positions.append([x, y])
            self.target_formation = np.array(positions[:self.num_agents])
    
    def _init_cooperative_navigation(self):
        """Initialize cooperative navigation task"""
        # define goal and obstacles
        self.goal_position = np.array([self.grid_size * 0.8, self.grid_size * 0.8])
        
        # place obstacles
        self.num_obstacles = min(15, self.grid_size * self.grid_size // 10)
        self.obstacle_positions = []
        
        while len(self.obstacle_positions) < self.num_obstacles:
            pos = np.random.uniform(0, self.grid_size, 2)
            # don't place obstacles too close to start or goal
            if (np.linalg.norm(pos - np.array([1, 1])) > 2 and 
                np.linalg.norm(pos - self.goal_position) > 2):
                self.obstacle_positions.append(pos)
        
        self.obstacle_positions = np.array(self.obstacle_positions)
        self.obstacle_radius = 0.5
        
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, List[float], bool, bool, Dict]:
        """Execute one step in the environment"""
        self.current_step += 1
        
        # execute actions
        self._execute_actions(actions)
        
        # calculate rewards
        rewards = self._calculate_rewards(actions)
        
        # check termination
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps
        
        # get observations
        observations = self._get_observations()
        
        # get info
        info = self._get_info()
        
        return observations, rewards, terminated, truncated, info
    
    def _execute_actions(self, actions: np.ndarray):
        """Execute agent actions"""
        for i, action in enumerate(actions):
            if action == 0:  # up
                self.agent_positions[i][1] = min(self.agent_positions[i][1] + 1, self.grid_size - 1)
            elif action == 1:  # down
                self.agent_positions[i][1] = max(self.agent_positions[i][1] - 1, 0)
            elif action == 2:  # left
                self.agent_positions[i][0] = max(self.agent_positions[i][0] - 1, 0)
            elif action == 3:  # right
                self.agent_positions[i][0] = min(self.agent_positions[i][0] + 1, self.grid_size - 1)
            # action == 4: stay
    
    def _calculate_rewards(self, actions: np.ndarray) -> List[float]:
        """Calculate rewards based on task type and reward type"""
        if self.task_type == CoordinationTask.RESOURCE_COLLECTION:
            return self._resource_collection_rewards()
        elif self.task_type == CoordinationTask.TARGET_COVERAGE:
            return self._target_coverage_rewards()
        elif self.task_type == CoordinationTask.FORMATION_CONTROL:
            return self._formation_control_rewards()
        elif self.task_type == CoordinationTask.COOPERATIVE_NAVIGATION:
            return self._cooperative_navigation_rewards()
        
        return [0.0] * self.num_agents
    
    def _resource_collection_rewards(self) -> List[float]:
        """Calculate rewards for resource collection task"""
        rewards = [0.0] * self.num_agents
        
        # check for resource collection
        for i, agent_pos in enumerate(self.agent_positions):
            for j, resource_pos in enumerate(self.resource_positions):
                if not self.collected_resources[j]:
                    distance = np.linalg.norm(agent_pos - resource_pos)
                    if distance < 1.0:  # collection range
                        self.collected_resources[j] = True
                        reward_value = self.resource_values[j]
                        
                        if self.reward_type == "global":
                            # all agents get the reward, scaled by distance to encourage efficiency
                            efficiency_bonus = max(0, 2.0 - distance)  # bonus for quick collection
                            total_reward = reward_value + efficiency_bonus
                            for k in range(self.num_agents):
                                rewards[k] += total_reward / self.num_agents
                                
                        elif self.reward_type == "local":
                            # only collecting agent gets reward
                            rewards[i] += reward_value + efficiency_bonus
                                
                        else:  # mixed
                            # collecting agent gets more, others get less
                            rewards[i] += reward_value * 0.7 + efficiency_bonus
                            for k in range(self.num_agents):
                                if k != i:
                                    rewards[k] += reward_value * 0.3 / (self.num_agents - 1)
        
        # resource respawn logic
        if np.random.random() < self.respawn_probability:
            # respawn some collected resources
            for j in range(self.num_resources):
                if self.collected_resources[j] and np.random.random() < 0.3:  # 30% chance per resource
                    self.collected_resources[j] = False
                    # move resource to new position
                    self.resource_positions[j] = np.random.uniform(0, self.grid_size, 2)
        
        # small penalty for time steps to encourage efficiency
        if self.reward_type != "local":
            penalty = -0.02
            for i in range(self.num_agents):
                rewards[i] += penalty
        
        return rewards
    
    def _target_coverage_rewards(self) -> List[float]:
        """Calculate rewards for target coverage task"""
        rewards = [0.0] * self.num_agents
        
        # check which targets are covered
        self.covered_targets = np.zeros(self.num_targets, dtype=bool)
        for i, target_pos in enumerate(self.target_positions):
            for agent_pos in self.agent_positions:
                if np.linalg.norm(agent_pos - target_pos) < self.coverage_radius:
                    self.covered_targets[i] = True
                    break
        
        # reward based on coverage
        coverage_ratio = np.sum(self.covered_targets) / self.num_targets
        
        if self.reward_type == "global":
            reward = coverage_ratio * 10.0
            rewards = [reward] * self.num_agents
        elif self.reward_type == "local":
            for i, agent_pos in enumerate(self.agent_positions):
                # reward for covering nearby targets
                local_reward = 0
                for j, target_pos in enumerate(self.target_positions):
                    if np.linalg.norm(agent_pos - target_pos) < self.coverage_radius:
                        local_reward += 2.0
                rewards[i] = local_reward
        else:  # mixed
            global_reward = coverage_ratio * 5.0
            for i in range(self.num_agents):
                rewards[i] = global_reward
                # add local bonus
                for j, target_pos in enumerate(self.target_positions):
                    if np.linalg.norm(self.agent_positions[i] - target_pos) < self.coverage_radius:
                        rewards[i] += 1.0
        
        return rewards
    
    def _formation_control_rewards(self) -> List[float]:
        """Calculate rewards for formation control task"""
        rewards = [0.0] * self.num_agents
        
        # calculate formation error
        formation_error = 0
        for i, agent_pos in enumerate(self.agent_positions):
            target_pos = self.target_formation[i]
            distance = np.linalg.norm(agent_pos - target_pos)
            formation_error += distance
        
        formation_error /= self.num_agents
        
        # reward based on formation accuracy
        if self.reward_type == "global":
            reward = -formation_error + 5.0  # baseline reward
            rewards = [reward] * self.num_agents
        elif self.reward_type == "local":
            for i, agent_pos in enumerate(self.agent_positions):
                target_pos = self.target_formation[i]
                distance = np.linalg.norm(agent_pos - target_pos)
                rewards[i] = -distance + 2.0
        else:  # mixed
            global_reward = -formation_error * 0.5 + 2.0
            for i in range(self.num_agents):
                target_pos = self.target_formation[i]
                distance = np.linalg.norm(agent_pos - target_pos)
                rewards[i] = global_reward - distance * 0.5
        
        return rewards
    
    def _cooperative_navigation_rewards(self) -> List[float]:
        """Calculate rewards for cooperative navigation task"""
        rewards = [0.0] * self.num_agents
        
        # reward for reaching goal
        for i, agent_pos in enumerate(self.agent_positions):
            distance_to_goal = np.linalg.norm(agent_pos - self.goal_position)
            
            if distance_to_goal < 1.0:  # reached goal
                if self.reward_type == "global":
                    goal_reward = 10.0
                    rewards = [goal_reward] * self.num_agents
                elif self.reward_type == "local":
                    rewards[i] = 10.0
                else:  # mixed
                    rewards[i] = 7.0
                    for j in range(self.num_agents):
                        if j != i:
                            rewards[j] += 1.0
                return rewards  # early return if goal reached
            
            # progress reward
            progress_reward = -distance_to_goal * 0.1
            
            if self.reward_type == "global":
                for j in range(self.num_agents):
                    rewards[j] += progress_reward
            elif self.reward_type == "local":
                rewards[i] += progress_reward
            else:  # mixed
                for j in range(self.num_agents):
                    rewards[j] += progress_reward * 0.5
                rewards[i] += progress_reward * 0.5
        
        # penalty for collisions with obstacles
        for i, agent_pos in enumerate(self.agent_positions):
            for obstacle_pos in self.obstacle_positions:
                if np.linalg.norm(agent_pos - obstacle_pos) < self.obstacle_radius:
                    if self.reward_type == "global":
                        for j in range(self.num_agents):
                            rewards[j] -= 1.0
                    elif self.reward_type == "local":
                        rewards[i] -= 1.0
                    else:  # mixed
                        rewards[i] -= 0.7
                        for j in range(self.num_agents):
                            if j != i:
                                rewards[j] -= 0.1
        
        return rewards
    
    def _get_observations(self) -> np.ndarray:
        """Get observations for all agents"""
        observations = []
        
        for i, agent_pos in enumerate(self.agent_positions):
            if self.observation_type == "local":
                obs = self._get_local_observation(i)
            elif self.observation_type == "global":
                obs = self._get_global_observation()
            else:  # partial
                obs = self._get_partial_observation(i)
            
            observations.append(obs)
        
        return np.array(observations, dtype=np.float32)
    
    def _get_local_observation(self, agent_idx: int) -> np.ndarray:
        """Get local observation for an agent"""
        agent_pos = self.agent_positions[agent_idx]
        obs = []
        
        # agent position
        obs.extend(agent_pos)
        
        # nearby agents (up to 4 closest)
        distances_to_agents = []
        for i, other_pos in enumerate(self.agent_positions):
            if i != agent_idx:
                dist = np.linalg.norm(agent_pos - other_pos)
                distances_to_agents.append((dist, other_pos))
        
        distances_to_agents.sort(key=lambda x: x[0])
        for i in range(min(4, len(distances_to_agents))):
            obs.extend(distances_to_agents[i][1])
        for i in range(4 - len(distances_to_agents)):
            obs.extend([0, 0])
        
        # task-specific local information
        if self.task_type == CoordinationTask.RESOURCE_COLLECTION:
            # nearby resources (up to 4 closest)
            distances_to_resources = []
            for j, resource_pos in enumerate(self.resource_positions):
                if not self.collected_resources[j]:
                    dist = np.linalg.norm(agent_pos - resource_pos)
                    distances_to_resources.append((dist, resource_pos))
            
            distances_to_resources.sort(key=lambda x: x[0])
            for i in range(min(4, len(distances_to_resources))):
                obs.extend(distances_to_resources[i][1])
            for i in range(4 - len(distances_to_resources)):
                obs.extend([0, 0])
        
        return np.array(obs, dtype=np.float32)
    
    def _get_global_observation(self) -> np.ndarray:
        """Get global observation"""
        obs = []
        
        # all agent positions
        for agent_pos in self.agent_positions:
            obs.extend(agent_pos)
        
        # task-specific global information
        if self.task_type == CoordinationTask.RESOURCE_COLLECTION:
            # all resource positions and collection status
            for i, resource_pos in enumerate(self.resource_positions):
                obs.extend(resource_pos)
                obs.append(1.0 if not self.collected_resources[i] else 0.0)
        
        return np.array(obs, dtype=np.float32)
    
    def _get_partial_observation(self, agent_idx: int) -> np.ndarray:
        """Get partial observation (limited range)"""
        agent_pos = self.agent_positions[agent_idx]
        obs = []
        
        # agent position
        obs.extend(agent_pos)
        
        # agents within communication range
        nearby_agents = []
        for i, other_pos in enumerate(self.agent_positions):
            if i != agent_idx:
                dist = np.linalg.norm(agent_pos - other_pos)
                if dist <= self.communication_range:
                    nearby_agents.append(other_pos)
        
        for i in range(min(8, len(nearby_agents))):
            obs.extend(nearby_agents[i])
        for i in range(8 - len(nearby_agents)):
            obs.extend([0, 0])
        
        return np.array(obs, dtype=np.float32)
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        if self.task_type == CoordinationTask.RESOURCE_COLLECTION:
            return np.all(self.collected_resources)
        elif self.task_type == CoordinationTask.TARGET_COVERAGE:
            return np.all(self.covered_targets)
        elif self.task_type == CoordinationTask.FORMATION_CONTROL:
            # check if all agents are close to target formation
            for i, agent_pos in enumerate(self.agent_positions):
                target_pos = self.target_formation[i]
                if np.linalg.norm(agent_pos - target_pos) > 0.5:
                    return False
            return True
        elif self.task_type == CoordinationTask.COOPERATIVE_NAVIGATION:
            # check if all agents reached goal
            for agent_pos in self.agent_positions:
                if np.linalg.norm(agent_pos - self.goal_position) > 1.0:
                    return False
            return True
        
        return False
    
    def _get_info(self) -> Dict:
        """Get additional information"""
        info = {
            'step': self.current_step,
            'agent_positions': self.agent_positions.copy(),
        }
        
        if self.task_type == CoordinationTask.RESOURCE_COLLECTION:
            info['resources_collected'] = np.sum(self.collected_resources)
            info['total_resources'] = self.num_resources
        elif self.task_type == CoordinationTask.TARGET_COVERAGE:
            info['targets_covered'] = np.sum(self.covered_targets)
            info['total_targets'] = self.num_targets
        elif self.task_type == CoordinationTask.FORMATION_CONTROL:
            formation_error = 0
            for i, agent_pos in enumerate(self.agent_positions):
                target_pos = self.target_formation[i]
                formation_error += np.linalg.norm(agent_pos - target_pos)
            info['formation_error'] = formation_error / self.num_agents
        elif self.task_type == CoordinationTask.COOPERATIVE_NAVIGATION:
            distances_to_goal = [np.linalg.norm(pos - self.goal_position) 
                               for pos in self.agent_positions]
            info['avg_distance_to_goal'] = np.mean(distances_to_goal)
        
        return info
    
    def get_communication_graph(self) -> np.ndarray:
        """Get communication graph based on agent distances"""
        adj_matrix = np.zeros((self.num_agents, self.num_agents))
        
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i != j:
                    distance = np.linalg.norm(
                        self.agent_positions[i] - self.agent_positions[j]
                    )
                    if distance <= self.communication_range:
                        adj_matrix[i][j] = 1.0 / (1.0 + distance)  # weight by distance
        
        return adj_matrix
    
    def render(self, mode='human'):
        """Render the environment (optional)"""
        pass  # implementation omitted for brevity
