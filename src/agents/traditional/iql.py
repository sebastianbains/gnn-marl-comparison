import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Any


class IQLAgent:
    # independent q-learning agent - traditional baseline
    
    def __init__(self, obs_dim: int, action_dim: int, lr: float = 1e-3, 
                 gamma: float = 0.99, epsilon: float = 0.1):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        
        # q-network
        self.q_network = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # target network
        self.target_network = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        # update target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def select_action(self, obs: np.ndarray, training: bool = True) -> int:
        # select action using epsilon-greedy policy
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            q_values = self.q_network(obs_tensor)
            return q_values.argmax().item()
    
    def update(self, batch: List[Tuple]) -> float:
        # update q-network using experience replay
        obs, actions, rewards, next_obs, dones = zip(*batch)
        
        # convert to numpy arrays first (faster than list of arrays)
        obs = torch.FloatTensor(np.array(obs))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_obs = torch.FloatTensor(np.array(next_obs))
        dones = torch.BoolTensor(np.array(dones))
        
        # current q-values
        current_q_values = self.q_network(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # next q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_obs).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # compute loss (both tensors now have same shape)
        loss = self.loss_fn(current_q_values, target_q_values)
        
        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        # update target network parameters
        self.target_network.load_state_dict(self.q_network.state_dict())


class MultiAgentIQL:
    # multi-agent independent q-learning
    
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, **kwargs):
        self.num_agents = num_agents
        self.agents = [IQLAgent(obs_dim, action_dim, **kwargs) for _ in range(num_agents)]
        
    def select_actions(self, observations: List[np.ndarray], training: bool = True) -> List[int]:
        # select actions for all agents
        return [agent.select_action(obs, training) for agent, obs in zip(self.agents, observations)]
    
    def update_agents(self, agent_batches: List[List[Tuple]]) -> List[float]:
        # update all agents
        losses = []
        for agent, batch in zip(self.agents, agent_batches):
            if batch:  # only update if batch is not empty
                loss = agent.update(batch)
                losses.append(loss)
        return losses
    
    def update_target_networks(self):
        # update all target networks
        for agent in self.agents:
            agent.update_target_network()
    
    def set_epsilon(self, epsilon: float):
        # set exploration rate for all agents
        for agent in self.agents:
            agent.epsilon = epsilon
    
    def update(self, batch: List[Tuple]) -> float:
        # update multiagent iql using experience replay
        # for multiagentiql, we need to handle the batch differently
        # since we stored experiences per agent, we need to reconstruct agent-specific batches
        
        obs, actions, rewards, next_obs, dones = zip(*batch)
        
        # create agent-specific batches
        agent_batches = [[] for _ in range(self.num_agents)]
        for i in range(len(batch)):
            # assign each experience to a random agent
            agent_idx = i % self.num_agents
            agent_batches[agent_idx].append((obs[i], actions[i], rewards[i], next_obs[i], dones[i]))
        
        # update each agent
        losses = self.update_agents(agent_batches)
        
        # return average loss
        return sum(losses) / len(losses) if losses else 0.0
    
    def get_epsilon(self):
        # get current exploration rate (average of all agents)
        if self.agents:
            return sum(agent.epsilon for agent in self.agents) / len(self.agents)
        return 0.1
