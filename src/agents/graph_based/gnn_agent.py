import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import List, Dict, Tuple, Any


class GraphAgentNetwork(nn.Module):
    """Graph-based agent network using GNN"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64, 
                 gnn_type: str = 'gat', num_heads: int = 4):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gnn_type = gnn_type
        
        # observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # gNN layers
        if gnn_type == 'gcn':
            self.gnn1 = GCNConv(hidden_dim, hidden_dim)
            self.gnn2 = GCNConv(hidden_dim, hidden_dim)
        elif gnn_type == 'gat':
            self.gnn1 = GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False)
            self.gnn2 = GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        # action decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, observations: torch.Tensor, edge_index: torch.Tensor, 
                batch: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through graph network"""
        # handle different input shapes
        if observations.dim() == 3:
            # batch of observations: [batch_size, num_agents, obs_dim]
            batch_size, num_agents, obs_dim = observations.shape
            observations_flat = observations.view(-1, obs_dim)  # [batch_size * num_agents, obs_dim]
            
            # create batch-specific edge indices for each graph in batch
            if edge_index.numel() == 0:  # empty edge index
                edge_indices_batch = edge_index
            else:
                # repeat edge indices for each batch with offset
                edge_indices_list = []
                for b in range(batch_size):
                    offset = b * num_agents
                    edge_indices_b = edge_index + offset
                    edge_indices_list.append(edge_indices_b)
                edge_indices_batch = torch.cat(edge_indices_list, dim=1)
            
            # encode observations
            x = self.obs_encoder(observations_flat)
            
            # apply GNN layers
            x = F.relu(self.gnn1(x, edge_indices_batch))
            x = F.relu(self.gnn2(x, edge_indices_batch))
            
            # decode to actions
            q_values = self.action_decoder(x)
            
            # reshape back to [batch_size, num_agents, action_dim]
            q_values = q_values.view(batch_size, num_agents, self.action_dim)
            
        else:
            # single observation: [num_agents, obs_dim]
            num_agents = observations.shape[0]
            
            # filter edge_index to be within valid range
            if edge_index.numel() > 0:
                valid_mask = (edge_index[0] < num_agents) & (edge_index[1] < num_agents)
                edge_index = edge_index[:, valid_mask]
            
            # encode observations
            x = self.obs_encoder(observations)
            
            # apply GNN layers
            x = F.relu(self.gnn1(x, edge_index))
            x = F.relu(self.gnn2(x, edge_index))
            
            # decode to actions
            q_values = self.action_decoder(x)
            
            # ensure output is [num_agents, action_dim]
            if q_values.shape[0] != num_agents:
                print(f"Warning: Output shape {q_values.shape} doesn't match num_agents {num_agents}")
        
        return q_values


class GraphCommunicationNetwork(nn.Module):
    """
    Learnable communication graph between agents.
    
    This network learns which agents should communicate with each other
    by learning edge weights in the communication graph.
    """
    
    def __init__(self, num_agents: int, hidden_dim: int):
        super().__init__()
        self.num_agents = num_agents
        
        # learnable node embeddings
        self.node_embeddings = nn.Parameter(torch.randn(num_agents, hidden_dim))
        
        # attention mechanism for edge weights
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self) -> torch.Tensor:
        """Generate communication graph"""
        # create all possible edges
        rows = []
        cols = []
        edge_weights = []
        
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i != j:
                    rows.append(i)
                    cols.append(j)
                    
                    # compute edge weight using attention
                    emb_i = self.node_embeddings[i]
                    emb_j = self.node_embeddings[j]
                    edge_input = torch.cat([emb_i, emb_j])
                    weight = self.attention(edge_input)
                    edge_weights.append(weight)
        
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_weights = torch.cat(edge_weights)
        
        return edge_index, edge_weights


class MultiAgentGNN:
    """Multi-Agent Graph Neural Network implementation"""
    
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, 
                 lr: float = 1e-3, gamma: float = 0.99, epsilon: float = 0.1,
                 gnn_type: str = 'gat', hidden_dim: int = 64, 
                 communication_graph: str = 'learnable'):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.gnn_type = gnn_type
        self.hidden_dim = hidden_dim
        self.communication_graph = communication_graph
        
        # graph agent network
        self.agent_network = GraphAgentNetwork(
            obs_dim, action_dim, hidden_dim, gnn_type
        )
        
        # communication graph
        if communication_graph == 'learnable':
            self.comm_graph = GraphCommunicationNetwork(num_agents, hidden_dim)
        else:
            self.comm_graph = None
        
        # target network
        self.target_network = GraphAgentNetwork(
            obs_dim, action_dim, hidden_dim, gnn_type
        )
        self.target_network.load_state_dict(self.agent_network.state_dict())
        
        # optimizers with gradient clipping for stability
        self.optimizer = optim.Adam(self.agent_network.parameters(), lr=lr)
        self.max_grad_norm = 1.0  # gradient clipping threshold
        if self.comm_graph:
            self.comm_optimizer = optim.Adam(self.comm_graph.parameters(), lr=lr)
        
        self.loss_fn = nn.MSELoss()
        
        # cache static graph structure for speed
        self._cached_edge_index = None
        if communication_graph in ('fully_connected', 'full'):
            rows, cols = [], []
            for i in range(num_agents):
                for j in range(num_agents):
                    if i != j:
                        rows.append(i)
                        cols.append(j)
            self._cached_edge_index = torch.tensor([rows, cols], dtype=torch.long)
        # 'distance' means the caller (wrapper) supplies the edge index per step;
        # _get_edge_index() returns empty — wrapper overrides it via direct calls.
        
    def _get_edge_index(self):
        """Get edge index (cached for speed)"""
        if self._cached_edge_index is not None:
            return self._cached_edge_index
        
        # for learnable graphs, compute on demand
        if self.communication_graph == 'learnable' and self.comm_graph:
            edge_index, _ = self.comm_graph()
            return edge_index
        
        # ring topology (rarely used, not cached)
        if self.communication_graph == 'ring':
            rows, cols = [], []
            for i in range(self.num_agents):
                rows.append(i)
                cols.append((i + 1) % self.num_agents)
                rows.append(i)
                cols.append((i - 1) % self.num_agents)
            return torch.tensor([rows, cols], dtype=torch.long)
        
        # 'distance' or unknown: caller supplies edge index externally
        return torch.empty((2, 0), dtype=torch.long)
    
    def select_actions(self, observations: List[np.ndarray], training: bool = True) -> List[int]:
        """Select actions for all agents using graph communication"""
        obs_tensor = torch.FloatTensor(observations)
        edge_index = self._get_edge_index()
        
        with torch.no_grad():
            q_values = self.agent_network(obs_tensor, edge_index)
            
            if training and np.random.random() < self.epsilon:
                # random exploration
                return [np.random.randint(self.action_dim) for _ in range(self.num_agents)]
            else:
                # greedy actions
                return q_values.argmax(dim=1).cpu().numpy().tolist()
    
    def update(self, batch: List[Tuple]) -> float:
        """
        Update GNN using batch data with proper multi-agent experiences.
        OPTIMIZED: Vectorized batch processing for speed.
        
        Args:
            batch: List of tuples (obs_all, actions_all, rewards_all, next_obs_all, done)
        
        Returns:
            Average loss across batch
        """
        # vectorize batch processing - stack all experiences
        obs_list, action_list, reward_list, next_obs_list, done_list = [], [], [], [], []
        
        for experience in batch:
            obs_all, actions_all, rewards_all, next_obs_all, done = experience
            obs_list.append(obs_all if isinstance(obs_all, np.ndarray) else np.array(obs_all))
            next_obs_list.append(next_obs_all if isinstance(next_obs_all, np.ndarray) else np.array(next_obs_all))
            action_list.append(actions_all if isinstance(actions_all, np.ndarray) else np.array(actions_all))
            reward_list.append(rewards_all if isinstance(rewards_all, np.ndarray) else np.array(rewards_all))
            done_list.append(done)
        
        # stack into tensors [batch_size, num_agents, ...]
        obs_tensor = torch.FloatTensor(np.stack(obs_list))  # [B, N, obs_dim]
        next_obs_tensor = torch.FloatTensor(np.stack(next_obs_list))  # [B, N, obs_dim]
        action_tensor = torch.LongTensor(np.stack(action_list))  # [B, N]
        reward_tensor = torch.FloatTensor(np.stack(reward_list))  # [B, N]
        done_tensor = torch.BoolTensor([[d] * self.num_agents for d in done_list])  # [B, N]
        
        # get cached edge index
        edge_index = self._get_edge_index()
        
        # process entire batch at once
        batch_size = obs_tensor.shape[0]
        all_current_q = []
        all_target_q = []
        
        for i in range(batch_size):
            # current Q-values
            current_q = self.agent_network(obs_tensor[i], edge_index)  # [N, action_dim]
            current_q_action = current_q.gather(1, action_tensor[i].unsqueeze(1)).squeeze(1)  # [N]
            all_current_q.append(current_q_action)
            
            # target Q-values
            with torch.no_grad():
                next_q = self.target_network(next_obs_tensor[i], edge_index)  # [N, action_dim]
                next_q_max = next_q.max(1)[0]  # [N]
                target_q = reward_tensor[i] + (self.gamma * next_q_max * ~done_tensor[i])
                all_target_q.append(target_q)
        
        # stack and compute loss
        current_q_stacked = torch.stack(all_current_q)  # [B, N]
        target_q_stacked = torch.stack(all_target_q)  # [B, N]
        total_loss = self.loss_fn(current_q_stacked, target_q_stacked)
        
        # optimize
        self.optimizer.zero_grad()
        if self.comm_graph:
            self.comm_optimizer.zero_grad()
        
        total_loss.backward()
        
        # gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.agent_network.parameters(), self.max_grad_norm)
        if self.comm_graph:
            torch.nn.utils.clip_grad_norm_(self.comm_graph.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        if self.comm_graph:
            self.comm_optimizer.step()
        
        return total_loss.item()
    
    def update_target_network(self):
        """Update target network"""
        self.target_network.load_state_dict(self.agent_network.state_dict())
    
    def set_epsilon(self, epsilon: float):
        """Set exploration rate"""
        self.epsilon = epsilon
    
    def get_communication_graph(self) -> torch.Tensor:
        """Get current communication graph (if learnable)"""
        if self.comm_graph:
            edge_index, edge_weights = self.comm_graph()
            return edge_index, edge_weights
        return None, None
