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

device = torch.device("cude" if torch.cuda.is_available() else "cpu")

class NewTD3(object):
    def __init__(self, state_dim, action_dim, N, max_action, epsilon):
        self.policy_noise = 0.005
        self.policy_noise_min = 0.004
        self.epsilon = epsilon 
        self.Actor = Actor(state_dim = state_dim, action_dim = action_dim, max_action = max_action).to(device = device)
        self.Actor_Target = Actor(state_dim = state_dim, action_dim = action_dim, max_action = max_action).to(device = device)
        self.Actor_Target.load_state_dict(self.Actor.state_dict())
        self.Actor_optimizer = torch.optim.Adam(self.Actor.parameters() , lr= 0.0000001)
        self.Critic = Critic(state_dim = state_dim, action_dim = action_dim , N = N).to(device = device)
        self.Critic_Target = Critic(state_dim = state_dim, action_dim = action_dim, N = N).to(device = device)
        self.Critic_Target.load_state_dict(self.Critic.state_dict())
        self.Critic_optimizer = torch.optim.Adam(self.Critic.parameters(), lr= 0.0000003)
        self.actor_scheduler = lr_scheduler.LinearLR(optimizer= self.Actor_optimizer, start_factor= 1.0, end_factor= 0.001 , total_iters = 1000)
        self.critic_scheduler = lr_scheduler.LinearLR(optimizer= self.Critic_optimizer, start_factor= 1.0, end_factor= 0.001 , total_iters = 1000)
        self.max_action = max_action
    def select_action(self, state):
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
    def train(self, central_critic, iterations, policies, replay_buffers, ps,  batch_size = 100, discount = 0.99, tau = 0.005, noise_clip = 0.05, policy_freq = 200):
        self.update_critic(central_critic = central_critic)
        for it in range(iterations):
            OGstates = {}
            OGactions = {}
            OGrewards = {}
            OGdones = {}
            OGnext_states = {}
            OGnext_actions = {}
            OGnoise = {}
            minstorage = len(replay_buffers[1].storage)
            for i in replay_buffers.keys():
                minstorage = min(len(replay_buffers[i].storage), minstorage)
            ind = np.random.randint(0,minstorage, batch_size)

            for p in range(1,NumOfPSs+1):
                batch_state, batch_action, batch_next_state, batch_reward, batch_done = replay_buffers[p].sample(batch_size , ind )
                OGstates[p] = torch.Tensor(batch_state).to(device = device)
                OGactions[p] = torch.Tensor(batch_action).to(device = device)
                OGnext_states[p] = torch.Tensor(batch_next_state).to(device = device)
                OGrewards[p] = torch.Tensor(batch_reward).to(device = device)
                OGdones[p] = torch.Tensor(batch_done).to(device = device)
                OGnext_actions[p] = policies[p].Actor_Target.forward(OGnext_states[p])            
                OGnoise[p] = torch.Tensor(batch_action).data.normal_(0, self.policy_noise).to(device = device)
                OGnoise[p] = OGnoise[p].clamp(-noise_clip, + noise_clip)
            OGnext_actions[ps] = (OGnext_actions[ps] + OGnoise[ps]).clamp(0, self.max_action)
            # OGnext_actions[ps] = abs(OGnext_actions[ps])
            input = []
            for p in range(1,NumOfPSs+1):
                input.append(OGnext_states[p])
            for p in range(1,NumOfPSs+1):
                input.append(OGnext_actions[p])
            input = torch.cat(input , 1)
            target_Q1 , target_Q2 = central_critic.Critic_Target.forward(input)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = torch.reshape(target_Q, (-1,))
            target_Q = OGrewards[ps] + ((1-OGdones[ps])*discount*target_Q).detach()
            currentinput = []
            for p in range(1,NumOfPSs+1):
                currentinput.append(OGstates[p])
            for p in range(1,NumOfPSs+1):
                currentinput.append(OGactions[p])
            currentinput = torch.cat(currentinput , 1)
            Q1_current , Q2_current = central_critic.Critic.forward(currentinput)
            Q1_current = torch.reshape(Q1_current, (-1,))
            Q2_current = torch.reshape(Q1_current, (-1,))
            Critic_Loss = F.mse_loss(Q1_current,target_Q) + F.mse_loss(Q2_current,target_Q)

            central_critic.Critic_optimizer.zero_grad()
            Critic_Loss.backward()
            central_critic.Critic_optimizer.step()
            if it % policy_freq == 0:
                OGactions[ps] = abs(self.Actor.forward(OGstates[ps]))
                input2 = []
                for p in range(1,NumOfPSs+1):
                    input2.append(OGstates[p])
                for p in range(1,NumOfPSs+1):
                    input2.append(OGactions[p])
                input2 = torch.cat(input2 , 1)
                Actor_loss = - central_critic.Critic.Q1(input2).mean()
                self.Actor_optimizer.zero_grad()
                Actor_loss.backward()
                self.Actor_optimizer.step()
                for param, target_param in zip(central_critic.Critic.parameters(), central_critic.Critic_Target.parameters()):
                    target_param.data.copy_(tau*param.data + (1-tau)*target_param.data)
                for param, target_param in zip(self.Actor.parameters(), self.Actor_Target.parameters()):
                    target_param.data.copy_(tau*param.data + (1-tau)*target_param.data)
            self.policy_noise = self.policy_noise - self.epsilon if self.policy_noise > self.policy_noise_min else self.policy_noise_min
        ACTOR_before_lr = self.Actor_optimizer.param_groups[0]["lr"]
        CRITIC_before_lr = central_critic.Critic_optimizer.param_groups[0]["lr"]
        self.actor_scheduler.step()
        central_critic.critic_scheduler.step()
        ACTOR_after_lr = self.Actor_optimizer.param_groups[0]["lr"]
        CRITIC_after_lr = central_critic.Critic_optimizer.param_groups[0]["lr"]
        print(f"Actor before:{ACTOR_before_lr}, after: {ACTOR_after_lr}")
        print(f"Critic before:{CRITIC_before_lr}, after: {CRITIC_after_lr}")
        # central_critic.update(self.Critic, self.Critic_Target, self.Critic_optimizer, self.actor_scheduler)
        return central_critic
    def save(self,filename, directory):
        torch.save(self.Actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.Critic.state_dict(), f"{directory}/{filename}_critic.pth")
    def load(self,filename, directory):
        self.Actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth"))
        self.Critic.load_state_dict(torch.load(f"{directory}/{filename}_critic.pth"))
