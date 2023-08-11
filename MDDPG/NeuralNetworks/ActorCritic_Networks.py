import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400,300)
        self.layer3 = nn.Linear(300, action_dim)
        self.max_action = max_action
    def forward(self,x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.max_action * torch.tanh(self.layer3(x))
        return x
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, N):
        super(Critic, self).__init__()
        self.NumOfAgents = N
        self.layer1 = nn.Linear(N*(state_dim + action_dim), 400)
        self.layer2 = nn.Linear(400,300)
        self.layer3 = nn.Linear(300,1)
        self.layer4 = nn.Linear(N*(state_dim + action_dim), 400)
        self.layer5 = nn.Linear(400,300)
        self.layer6 = nn.Linear(300,1)
    def forward(self,x):
        # x = torch.cat([states, actions] , 1)
        x1 = F.relu(self.layer1(x))
        x1 = F.relu(self.layer2(x1))
        x1 = self.layer3(x1)
        x2 = F.relu(self.layer4(x))
        x2 = F.relu(self.layer5(x2))
        x2 = self.layer6(x2)
        return x1, x2
    def Q1(self,x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
