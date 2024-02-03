import numpy as np 

class Replay_Buffer(object):
    def __init__(self,max_size = 1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
    def add(self,transition):
        if len(self.storage) == self.max_size:
            self.storage[self.ptr] = transition
            self.ptr += 1
        else:
            self.storage.append(transition)
    def sample(self,batch_size , ind):
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_done , batch_episode_reward, batch_episode_reward_next_state= [], [], [], [], [], [], []
        for i in ind:
            state, action, next_state, reward, done, episode_reward = self.storage[i]
            if done != True and i < len(self.storage) -1:
                _,_,_,_,_,episode_reward_prime = self.storage[i+1]
            else:
                episode_reward_prime = episode_reward
            batch_states.append(np.array(state, copy= False))
            batch_actions.append(np.array(action, copy= False))
            batch_next_states.append(np.array(next_state, copy = False))
            batch_rewards.append(np.array(reward, copy= False))
            batch_done.append(np.array(done, copy= False))
            batch_episode_reward.append(np.array(episode_reward, copy= False))
            batch_episode_reward_next_state.append(np.array(episode_reward_prime, copy= False))

        return np.array(batch_states), np.array(batch_actions), np.array(batch_next_states), np.array(batch_rewards), np.array(batch_done) , np.array(batch_episode_reward) ,np.array(batch_episode_reward_next_state)
    