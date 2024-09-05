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
        self.Actor = Actor(state_dim = state_dim, action_dim = action_dim, max_action = max_action).to(device = device)
        self.Actor_Target = Actor(state_dim = state_dim, action_dim = action_dim, max_action = max_action).to(device = device)
        self.Actor_Target.load_state_dict(self.Actor.state_dict())
        self.Actor_optimizer = torch.optim.Adam(self.Actor.parameters() , lr= 0.0001)
        self.Critic = Critic(state_dim = state_dim, action_dim = action_dim , N = 1).to(device = device)
        self.Critic_Target = Critic(state_dim = state_dim, action_dim = action_dim, N = 1).to(device = device)
        self.Critic_Target.load_state_dict(self.Critic.state_dict())
        self.Critic_optimizer = torch.optim.Adam(self.Critic.parameters(), lr= 0.0001)
        self.actor_scheduler = lr_scheduler.LinearLR(optimizer= self.Actor_optimizer, start_factor= 1.0, end_factor= 0.001 , total_iters = 1000)
        self.critic_scheduler = lr_scheduler.LinearLR(optimizer= self.Critic_optimizer, start_factor= 1.0, end_factor= 0.001 , total_iters = 1000)
        self.max_action = max_action
        self.it = 0
    def select_action(self, state):
 #       if state[2] <= SimulationParams.NumberOfCpuCycles[0]/(SimulationParams.NumberOfCpuCycles[0] + SimulationParams.NumberOfBits[0]):

#            return np.array([0])

        if state[2] <= SimulationParams.NumberOfCpuCycles[0]:
            return np.array([0])

        else:
            state = torch.tensor(state.reshape(1,-1)).to(device= device)
            state = state.to(torch.float32)
        # print(f"Action:  {type(self.Actor(state).cpu().data.numpy().flatten())}")
            return self.Actor(state).cpu().data.numpy().flatten()
    def update_critic(self , central_critic):
        self.Critic = central_critic.Critic
        self.Critic_optimizer = central_critic.Critic_optimizer
        self.Critic_Target = central_critic.Critic_Target
        self.critic_scheduler = central_critic.critic_scheduler
    def train(self, iterations, replay_buffer,  batch_size = 256, discount = 0.99, tau = 0.005, noise_clip = 0.5, policy_freq = 2):
        # self.update_critic(self = central_critic)
        for it in range(iterations):
           
            minstorage = len(replay_buffer.storage)
            ind = np.random.randint(0,minstorage, batch_size)

            batch_state, batch_action, batch_next_state, batch_reward, batch_done, shaped_reward , shaped_reward_next = replay_buffer.sample(batch_size , ind )
            states = torch.Tensor(batch_state).to(device = device)
            actions = torch.Tensor(batch_action).to(device = device)
            nextStates = torch.Tensor(batch_next_state).to(device = device)
            rewards = torch.Tensor(batch_reward).to(device = device)
            dones = torch.Tensor(batch_done).to(device = device)
            shaped_rewards = torch.Tensor(shaped_reward).to(device = device)
            shaped_next_rewards = torch.Tensor(shaped_reward_next).to(device = device)

            nextactions = self.Actor_Target.forward(nextStates)            
            noise = torch.Tensor(batch_action).data.normal_(0, self.policy_noise).to(device = device)
            noise = noise.clamp(-noise_clip, + noise_clip)
            nextactions = (nextactions + noise).clamp(0, self.max_action)
            # OGnext_actions[ps] = abs(OGnext_actions[ps])
            input = []
            input.append(nextStates)
            input.append(nextactions)
            input = torch.cat(input , 1)
            target_Q1 , target_Q2 = self.Critic_Target.forward(input)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = torch.reshape(target_Q, (-1,))
            target_Q = rewards + ((1-dones)*discount*target_Q).detach()
            currentinput = []
            currentinput.append(states)
            currentinput.append(actions)
            currentinput = torch.cat(currentinput , 1)
            Q1_current , Q2_current = self.Critic.forward(currentinput)
            Q1_current = torch.reshape(Q1_current, (-1,))
            Q2_current = torch.reshape(Q1_current, (-1,))
            Critic_Loss = F.mse_loss(Q1_current,target_Q) + F.mse_loss(Q2_current,target_Q)

            self.Critic_optimizer.zero_grad()
            Critic_Loss.backward()
            self.Critic_optimizer.step()
            if self.it % policy_freq == 0:
                actions = abs(self.Actor.forward(states))
                input2 = []
                input2.append(states)
                input2.append(actions)
                input2 = torch.cat(input2 , 1)
                Actor_loss = - self.Critic.Q1(input2).mean()
                self.Actor_optimizer.zero_grad()
                Actor_loss.backward()
                self.Actor_optimizer.step()
                for param, target_param in zip(self.Critic.parameters(), self.Critic_Target.parameters()):
                    target_param.data.copy_(tau*param.data + (1-tau)*target_param.data)
                for param, target_param in zip(self.Actor.parameters(), self.Actor_Target.parameters()):
                    target_param.data.copy_(tau*param.data + (1-tau)*target_param.data)
            self.policy_noise = self.policy_noise - self.epsilon if self.policy_noise > self.policy_noise_min else self.policy_noise_min
            self.it += 1
       # ACTOR_before_lr = self.Actor_optimizer.param_groups[0]["lr"]
       # CRITIC_before_lr = central_critic.Critic_optimizer.param_groups[0]["lr"]
        self.actor_scheduler.step()
        self.critic_scheduler.step()
      #  ACTOR_after_lr = self.Actor_optimizer.param_groups[0]["lr"]
       # CRITIC_after_lr = central_critic.Critic_optimizer.param_groups[0]["lr"]
        #print(f"Actor before:{ACTOR_before_lr}, after: {ACTOR_after_lr}")
       # print(f"Critic before:{CRITIC_before_lr}, after: {CRITIC_after_lr}")
        # central_critic.update(self.Critic, self.Critic_Target, self.Critic_optimizer, self.actor_scheduler)
    def save(self,filename, directory):
        torch.save(self.Actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.Critic.state_dict(), f"{directory}/{filename}_critic.pth")
    def load(self,filename, directory):
        self.Actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth"))
        self.Critic.load_state_dict(torch.load(f"{directory}/{filename}_critic.pth"))
