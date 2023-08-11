from MDDPG.NeuralNetworks.ActorCritic_Networks import Critic
import torch
import torch.nn.functional as F
from Config import SimulationParameters
from torch.optim import lr_scheduler

SimulationParams = SimulationParameters("Configs.json")
SimulationParams.Configure()

device = torch.device("cude" if torch.cuda.is_available() else "cpu")

class Central_Critic(object):
    def __init__(self, state_dim, action_dim, N):
        self.Critic = Critic(state_dim = state_dim, action_dim = action_dim , N = N).to(device = device)
        self.Critic_Target = Critic(state_dim = state_dim, action_dim = action_dim, N = N).to(device = device)
        self.Critic_Target.load_state_dict(self.Critic.state_dict())
        self.Critic_optimizer = torch.optim.Adam(self.Critic.parameters(), lr= 0.0000003)
        self.critic_scheduler = lr_scheduler.LinearLR(optimizer= self.Critic_optimizer, start_factor= 1.0, end_factor= 0.001 , total_iters = 1000)
    def update(self,critic, critic_target, critic_optimizer, critic_scheduler):
        self.Critic = critic
        self.Critic_Target = critic_target
        self.Critic_optimizer = critic_optimizer
        self.critic_scheduler = critic_scheduler