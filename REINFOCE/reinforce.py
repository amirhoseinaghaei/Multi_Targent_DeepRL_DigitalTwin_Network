import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from Config import SimulationParameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SimulationParams = SimulationParameters("Configs.json")
SimulationParams.Configure()
# Define the Policy Network for Continuous Action Space
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = 1 * torch.tanh(self.fc3(x))
        return abs(x)

# REINFORCE Algorithm for Continuous Action Space
class REINFORCE:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
    
    def select_action(self, state):
        if state[2] <= SimulationParams.NumberOfCpuCycles[0]:
            return np.array([0.0]), torch.tensor([0.0], dtype=torch.float32)
        
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Get action mean from the policy
        action_mean = self.policy(state)
        action_std = torch.tensor([0.1], dtype=torch.float32).to(device)  # Standard deviation for exploration
        
        # Create a normal distribution with mean and std
        distribution = torch.distributions.Normal(action_mean, action_std)
        
        # Sample an action
        action = distribution.sample()
        
        # Clamp action to be within the range [-1, 1]
        action = torch.clamp(action, 0, 1)
        
        # Compute log probability
        log_prob = distribution.log_prob(action).sum()
        return action.detach().cpu().numpy().flatten(), log_prob


    def compute_returns(self, rewards):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32).to(device)
        
    def update_policy(self, log_probs, returns):
        if not log_probs:
            print("log_probs is empty. Skipping policy update.")
            return        
        log_probs = [lp.unsqueeze(0) if lp.dim() == 0 else lp for lp in log_probs]
        log_probs = torch.cat(log_probs)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        if len(log_probs) != len(returns):
            print(f"Dimension mismatch: log_probs size {len(log_probs)}, returns size {len(returns)}")
            return
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        loss = -torch.sum(log_probs * returns)
        print(f"Computed loss: {loss.item()}")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


      
