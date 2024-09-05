import time
from MDDPG.NeuralNetworks.ActorCritic_Networks import Actor, Critic
from MDDPG.Replay_Buffer import Replay_Buffer
import torch
import torch.nn.functional as F
import numpy as np 
from Config import SimulationParameters
from torch.optim import lr_scheduler

SimulationParams = SimulationParameters("Configs.json")
SimulationParams.Configure()
NumOfPSs = SimulationParams.NumberOfPS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, epsilon):
        self.policy_noise = 0.5
        self.policy_noise_min = 0.1
        self.epsilon = epsilon

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.0001)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.0001)

        self.actor_scheduler = lr_scheduler.LinearLR(self.actor_optimizer, start_factor=1.0, end_factor=0.001, total_iters=1000)
        self.critic_scheduler = lr_scheduler.LinearLR(self.critic_optimizer, start_factor=1.0, end_factor=0.001, total_iters=1000)

        self.max_action = max_action
        self.it = 0

    def select_action(self, state):
        state = torch.tensor(state.reshape(1, -1)).to(device).float()
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256, discount=0.99, tau=0.005, noise_clip=0.5, policy_freq=2):
        for it in range(batch_size):
            # Sample a batch of transitions from the replay buffer
            batch_state, batch_action, batch_next_state, batch_reward, batch_done = replay_buffer.sample(batch_size)

            state = torch.Tensor(batch_state).to(device)
            action = torch.Tensor(batch_action).to(device)
            next_state = torch.Tensor(batch_next_state).to(device)
            reward = torch.Tensor(batch_reward).to(device)
            done = torch.Tensor(batch_done).to(device)

            # Select action according to policy and add clipped noise
            noise = torch.Tensor(batch_action).data.normal_(0, self.policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:
                # Compute actor loss
                actor_loss = -self.critic(state, self.actor(state))[0].mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            self.policy_noise = max(self.policy_noise - self.epsilon, self.policy_noise_min)
            self.it += 1

        self.actor_scheduler.step()
        self.critic_scheduler.step()

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{directory}/{filename}_critic.pth"))