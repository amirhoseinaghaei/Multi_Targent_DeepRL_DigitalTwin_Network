import time
from TwinDelayedDDPG.NeuralNetworks.ActorCritic_Networks import Actor, Critic
from TwinDelayedDDPG.Replay_Buffer import Replay_Buffer
import torch
import torch.nn.functional as F
import numpy as np 
from Config import SimulationParameters
from torch.optim import lr_scheduler

SimulationParams = SimulationParameters("Configs.json")
SimulationParams.Configure()

device = torch.device("cude" if torch.cuda.is_available() else "cpu")

class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, epsilon):
        self.policy_noise = 0.005
        self.policy_noise_min = 0.004
        self.epsilon = epsilon 
        self.Actor = Actor(state_dim = state_dim, action_dim = action_dim, max_action = max_action).to(device = device)
        self.Actor_Target = Actor(state_dim = state_dim, action_dim = action_dim, max_action = max_action).to(device = device)
        self.Actor_Target.load_state_dict(self.Actor.state_dict())
        self.Actor_optimizer = torch.optim.Adam(self.Actor.parameters() , lr= 0.0000003)
        self.Critic = Critic(state_dim = state_dim, action_dim = action_dim).to(device = device)
        self.Critic_Target = Critic(state_dim = state_dim, action_dim = action_dim).to(device = device)
        self.Critic_Target.load_state_dict(self.Critic.state_dict())
        self.Critic_optimizer = torch.optim.Adam(self.Critic.parameters(), lr= 0.0000005)
        self.actor_scheduler = lr_scheduler.LinearLR(optimizer= self.Actor_optimizer, start_factor= 1.0, end_factor= 0.001 , total_iters = 100)
        self.critic_scheduler = lr_scheduler.LinearLR(optimizer= self.Critic_optimizer, start_factor= 1.0, end_factor= 0.001 , total_iters = 100)
        self.max_action = max_action
    def select_action(self, state):
        if state[2] <= SimulationParams.NumberOfCpuCycles[0]:
            return np.array([0])
        else:
            state = torch.tensor(state.reshape(1,-1)).to(device= device)
            state = state.to(torch.float32)
        # print(f"Action:  {type(self.Actor(state).cpu().data.numpy().flatten())}")
            return self.Actor(state).cpu().data.numpy().flatten()
    
    def train(self, replay_buffer, iterations, batch_size = 100, discount = 0.99, tau = 0.005, noise_clip = 0.05, policy_freq = 200):
        for it in range(iterations):
            
            batch_state, batch_action, batch_next_state, batch_reward, batch_done = replay_buffer.sample(batch_size , )
            
            states = torch.Tensor(batch_state).to(device = device)
            actions = torch.Tensor(batch_action).to(device = device)
            next_states = torch.Tensor(batch_next_state).to(device = device)
            rewards = torch.Tensor(batch_reward).to(device = device)
            dones = torch.Tensor(batch_done).to(device = device)
            next_actions = self.Actor_Target.forward(next_states)
            # next_actions = abs(next_actions)
            # time.sleep(10)
            # print("Next action is as follows: ")
            # print(abs(next_actions))
            noise = torch.Tensor(batch_action).data.normal_(0, self.policy_noise).to(device = device)
            noise = noise.clamp(-noise_clip, + noise_clip)
            # print("Noise is as follows: ")
            # print(next_actions + noise)
            next_actions = (next_actions + noise).clamp(-self.max_action, self.max_action)
            # print("New next action is as follows: ")
            next_actions = abs(next_actions)
            target_Q1 , target_Q2 = self.Critic_Target.forward(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = torch.reshape(target_Q, (-1,))
            # print("-------------------------------------------")
            # print(states)
            # print(dones)
            # print(rewards)
            target_Q = rewards + ((1-dones)*discount*target_Q).detach()
            # print(target_Q)
            Q1_current , Q2_current = self.Critic.forward(states,actions)
            # print(f"Q1_Current: {Q1_current}")
            # print(f"Q1_Current: {Q2_current}")
            # time.sleep(10)
            # print("-------------------------------------------")

            Q1_current = torch.reshape(Q1_current, (-1,))
            Q2_current = torch.reshape(Q1_current, (-1,))
            Critic_Loss = F.mse_loss(Q1_current,target_Q) + F.mse_loss(Q2_current,target_Q)

            self.Critic_optimizer.zero_grad()
            Critic_Loss.backward()
            self.Critic_optimizer.step()
            if it % policy_freq == 0:

                Actor_loss = - self.Critic.Q1(state = states, action = abs(self.Actor.forward(states))).mean()
                # print(f"Actions: {self.Actor.forward(states)}")
                # print(f"Actor loss: {Actor_loss}")
                # time.sleep(1)
                self.Actor_optimizer.zero_grad()
                Actor_loss.backward()
                self.Actor_optimizer.step()
                for param, target_param in zip(self.Critic.parameters(), self.Critic_Target.parameters()):
                    target_param.data.copy_(tau*param.data + (1-tau)*target_param.data)
                for param, target_param in zip(self.Actor.parameters(), self.Actor_Target.parameters()):
                    target_param.data.copy_(tau*param.data + (1-tau)*target_param.data)
            self.policy_noise = self.policy_noise - self.epsilon if self.policy_noise > self.policy_noise_min else self.policy_noise_min
        ACTOR_before_lr = self.Actor_optimizer.param_groups[0]["lr"]
        CRITIC_before_lr = self.Critic_optimizer.param_groups[0]["lr"]
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        ACTOR_after_lr = self.Actor_optimizer.param_groups[0]["lr"]
        CRITIC_after_lr = self.Critic_optimizer.param_groups[0]["lr"]
        print(f"Actor before:{ACTOR_before_lr}, after: {ACTOR_after_lr}")
        print(f"Critic before:{CRITIC_before_lr}, after: {CRITIC_after_lr}")
    def save(self,filename, directory):
        torch.save(self.Actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.Critic.state_dict(), f"{directory}/{filename}_critic.pth")
    def load(self,filename, directory):
        self.Actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth"))
        self.Critic.load_state_dict(torch.load(f"{directory}/{filename}_critic.pth"))
