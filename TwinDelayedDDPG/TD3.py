from NeuralNetworks.ActorCritic_Networks import Actor, Critic
from Replay_Buffer import Replay_Buffer
import torch
import torch.nn.functional as F
import numpy as np 
device = torch.device("cude" if torch.cuda.is_available() else "cpu")

class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.Actor = Actor(state_dim = state_dim, action_dim = action_dim, max_action = max_action).to(device = device)
        self.Actor_Target = Actor(state_dim = state_dim, action_dim = action_dim, max_action = max_action).to(device = device)
        self.Actor_Target.load_state_dict(self.Actor.state_dict())
        self.Actor_optimizer = torch.optim.Adam(self.Actor.parameters())
        self.Critic = Critic(state_dim = state_dim, action_dim = action_dim).to(device = device)
        self.Critic_Target = Critic(state_dim = state_dim, action_dim = action_dim).to(device = device)
        self.Critic_Target.load_state_dict(self.Critic.state_dict())
        self.Critic_optimizer = torch.optim.Adam(self.Critic.parameters())
        self.max_action = max_action
    def select_action(self, state):
        state = torch.tensor(state.reshape(1,-1)).to(device= device)
        return self.Actor(state).cpu().data.numpy().flatten()
    def train(self, replay_buffer, iterations, batch_size = 100, discount = 0.99, tau = 0.005, policy_noise = 0.2, noise_clip = 0.5, policy_freq = 2):
        for it in range(iterations):
            batch_state, batch_action, batch_next_state, batch_reward, batch_done = replay_buffer.sample(batch_size)
            
            states = torch.Tensor(batch_state).to(device = device)
            actions = torch.Tensor(batch_action).to(device = device)
            next_states = torch.Tensor(batch_next_state).to(device = device)
            rewards = torch.Tensor(batch_reward).to(device = device)
            dones = torch.Tensor(batch_done).to(device = device)
            next_actions = self.Actor_Target.forward(next_states)
            noise = torch.Tensor(next_actions).data.normal_(0, policy_noise).to(device = device)
            noise = noise.clamp(-noise_clip, + noise_clip)
            next_actions = (next_actions + noise).clamp(-self.max_action, self.max_action)
            target_Q = torch.min(self.Critic_Target.forward(next_states, next_actions))
            target = rewards + (discount*target_Q*(1-dones)).detach()
            Q1_current , Q2_current = self.Critic.forward(states,actions)
            Critic_Loss = F.mse_loss(Q1_current,target) + F.mse_loss(Q2_current,target)
            self.Critic_optimizer.zero_grad()
            Critic_Loss.backward()
            self.Critic_optimizer.step()
            if it % policy_freq == 0:
                Actor_loss = - self.Critic.Q1(state = states, action = self.Actor.forward(states)).mean()
                self.Actor_optimizer.zero_grad()
                Actor_loss.backward()
                self.Actor_optimizer.step()
                for param, target_param in zip(self.Critic.parameters(), self.Critic_Target.parameters()):
                    target_param.data.copy_(tau*param.data + (1-tau)*target_param.data)
                for param, target_param in zip(self.Actor.parameters(), self.Actor_Target.parameters()):
                    target_param.data.copy_(tau*param.data + (1-tau)*target_param.data)
