import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool
import numpy as np
from typing import List, Dict, Tuple, Any


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for agent communication"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()
        
        # project to queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        output = self.out_proj(attn_output)
        
        return output, attn_weights


class GraphAttentionAgent(nn.Module):
    """Agent with graph attention mechanism"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64, 
                 num_heads: int = 8):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # multi-head attention for communication
        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        
        # gAT layers for spatial reasoning
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        
        # action decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, observations: torch.Tensor, edge_index: torch.Tensor, 
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass with attention and GAT"""
        batch_size, num_agents = observations.size(0), observations.size(1)
        
        # encode observations
        x = self.obs_encoder(observations)  # [batch_size, num_agents, hidden_dim]
        
        # apply multi-head attention (agent-wise communication)
        attn_out, attn_weights = self.attention(x, attention_mask)
        x = self.layer_norm1(x + attn_out)  # residual connection
        
        # reshape for GAT processing
        x_flat = x.view(-1, self.hidden_dim)  # [batch_size * num_agents, hidden_dim]
        
        # apply GAT layers (spatial communication)
        x_gat = F.relu(self.gat1(x_flat, edge_index))
        x_gat = F.relu(self.gat2(x_gat, edge_index))
        
        # reshape back
        x_gat = x_gat.view(batch_size, num_agents, self.hidden_dim)
        x = self.layer_norm2(x + x_gat)  # residual connection
        
        # decode to actions
        q_values = self.action_decoder(x)
        
        return q_values, attn_weights


class AdaptiveGraphTopology(nn.Module):
    """Adaptive graph topology learning"""
    
    def __init__(self, num_agents: int, hidden_dim: int):
        super().__init__()
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        
        # node embeddings
        self.node_embeddings = nn.Parameter(torch.randn(num_agents, hidden_dim))
        
        # edge prediction network
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # temperature for edge thresholding
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, training: bool = True) -> torch.Tensor:
        """Generate adaptive edge index"""
        # create all possible edges
        edge_indices = []
        edge_weights = []
        
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i != j:
                    # predict edge weight
                    emb_i = self.node_embeddings[i]
                    emb_j = self.node_embeddings[j]
                    edge_input = torch.cat([emb_i, emb_j])
                    weight = self.edge_predictor(edge_input)
                    
                    # apply temperature and threshold during training
                    if training:
                        # gumbel-softmax for differentiable sampling
                        weight = torch.sigmoid(weight / self.temperature)
                        
                        # random thresholding during training
                        if torch.rand(1) < weight:
                            edge_indices.append([i, j])
                            edge_weights.append(weight)
                    else:
                        # deterministic thresholding during evaluation
                        if weight > 0.5:
                            edge_indices.append([i, j])
                            edge_weights.append(weight)
        
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
            edge_weights = torch.cat(edge_weights)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weights = torch.empty(0)
        
        return edge_index, edge_weights


class AdvancedMultiAgentGNN:
    """Advanced Multi-Agent GNN with attention and adaptive topology"""
    
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, 
                 lr: float = 1e-3, gamma: float = 0.99, epsilon: float = 0.1,
                 hidden_dim: int = 64, num_heads: int = 8, 
                 adaptive_topology: bool = True):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.hidden_dim = hidden_dim
        self.adaptive_topology = adaptive_topology
        
        # graph attention agent
        self.agent_network = GraphAttentionAgent(
            obs_dim, action_dim, hidden_dim, num_heads
        )
        
        # adaptive topology
        if adaptive_topology:
            self.topology_network = AdaptiveGraphTopology(num_agents, hidden_dim)
        
        # target network
        self.target_network = GraphAttentionAgent(
            obs_dim, action_dim, hidden_dim, num_heads
        )
        self.target_network.load_state_dict(self.agent_network.state_dict())
        
        # optimizers with gradient clipping for stability
        self.optimizer = torch.optim.Adam(self.agent_network.parameters(), lr=lr)
        self.max_grad_norm = 1.0  # gradient clipping threshold
        if adaptive_topology:
            self.topology_optimizer = torch.optim.Adam(
                self.topology_network.parameters(), lr=lr * 0.1
            )
        
        self.loss_fn = nn.MSELoss()
        
        # experience replay buffer
        self.replay_buffer = []
        self.buffer_size = 10000
        
    def _get_edge_index(self, training: bool = True) -> torch.Tensor:
        """Get current edge index"""
        if self.adaptive_topology:
            edge_index, _ = self.topology_network(training)
            return edge_index
        else:
            # default to fully connected
            rows, cols = [], []
            for i in range(self.num_agents):
                for j in range(self.num_agents):
                    if i != j:
                        rows.append(i)
                        cols.append(j)
            return torch.tensor([rows, cols], dtype=torch.long)
    
    def select_actions(self, observations: List[np.ndarray], training: bool = True) -> Tuple[List[int], torch.Tensor]:
        """Select actions with attention weights"""
        obs_tensor = torch.FloatTensor(observations).unsqueeze(0)  # [1, num_agents, obs_dim]
        edge_index = self._get_edge_index(training)
        
        with torch.no_grad():
            q_values, attn_weights = self.agent_network(
                obs_tensor, edge_index
            )
            
            if training and np.random.random() < self.epsilon:
                actions = [np.random.randint(self.action_dim) for _ in range(self.num_agents)]
            else:
                actions = q_values.squeeze(0).argmax(dim=-1).cpu().numpy().tolist()
        
        return actions, attn_weights.squeeze(0)
    
    def update(self, batch: List[Tuple]) -> Tuple[float, torch.Tensor]:
        """
        Update network with batch data using proper multi-agent experiences.
        
        Args:
            batch: List of tuples (obs_all, actions_all, rewards_all, next_obs_all, done)
        
        Returns:
            Tuple of (loss, attention_weights)
        """
        current_losses = []
        attention_weights = []
        
        # process each multi-agent experience
        for experience in batch:
            obs_all, actions_all, rewards_all, next_obs_all, done = experience
            
            # convert to proper format
            if isinstance(obs_all, np.ndarray):
                obs_batch = [obs_all[i] for i in range(self.num_agents)]
                next_obs_batch = [next_obs_all[i] for i in range(self.num_agents)]
            else:
                obs_batch = obs_all
                next_obs_batch = next_obs_all
            
            if isinstance(actions_all, np.ndarray):
                action_batch = actions_all.tolist() if hasattr(actions_all, 'tolist') else list(actions_all)
            else:
                action_batch = actions_all
                
            if isinstance(rewards_all, np.ndarray):
                reward_batch = rewards_all.tolist() if hasattr(rewards_all, 'tolist') else list(rewards_all)
            else:
                reward_batch = rewards_all
            
            # get edge index
            edge_index, _ = self._get_edge_index(training=True)
            
            # current Q-values
            obs_tensor = torch.FloatTensor(obs_batch).unsqueeze(0)
            current_q, current_attn = self.agent_network(obs_tensor, edge_index)
            attention_weights.append(current_attn)
            
            # get Q-values for taken actions
            action_tensor = torch.LongTensor(action_batch)
            
            if current_q.dim() == 3:
                current_q_squeezed = current_q.squeeze(0)
                current_q_action = current_q_squeezed.gather(1, action_tensor.unsqueeze(1))
            else:
                current_q_action = current_q.gather(1, action_tensor.unsqueeze(1))
            
            # target Q-values
            with torch.no_grad():
                next_obs_tensor = torch.FloatTensor(next_obs_batch).unsqueeze(0)
                next_q, _ = self.target_network(next_obs_tensor, edge_index)
                
                if next_q.dim() == 3:
                    next_q_max = next_q.squeeze(0).max(1)[0]
                else:
                    next_q_max = next_q.max(1)[0]
                
                reward_tensor = torch.FloatTensor(reward_batch)
                done_tensor = torch.BoolTensor([done] * self.num_agents)
                target_q = reward_tensor + (self.gamma * next_q_max * ~done_tensor)
            
            # compute loss
            loss = self.loss_fn(current_q_action.squeeze(), target_q)
            current_losses.append(loss)
        
        # average loss across batch
        total_loss = torch.stack(current_losses).mean()
        
        # optimize
        self.optimizer.zero_grad()
        if self.adaptive_topology:
            self.topology_optimizer.zero_grad()
        
        total_loss.backward()
        
        # gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.agent_network.parameters(), self.max_grad_norm)
        if self.adaptive_topology:
            torch.nn.utils.clip_grad_norm_(self.topology_network.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        if self.adaptive_topology:
            self.topology_optimizer.step()
        
        # average attention weights
        avg_attention = torch.stack(attention_weights).mean(dim=0) if attention_weights else torch.zeros(self.num_agents, self.num_agents)
        
        return total_loss.item(), avg_attention
    
    def update_target_network(self):
        """Update target network"""
        self.target_network.load_state_dict(self.agent_network.state_dict())
    
    def set_epsilon(self, epsilon: float):
        """Set exploration rate"""
        self.epsilon = epsilon
    
    def get_topology_importance(self) -> torch.Tensor:
        """Get importance of different communication links"""
        if self.adaptive_topology:
            edge_index, edge_weights = self.topology_network(training=False)
            return edge_index, edge_weights
        return None, None
