import torch
from torch.distributions import Normal


class KBandit:
    def __init__(self, k, is_stationary=True, steps=1000):
        self.k = k
        self.is_stationary = is_stationary
        self.std = 1.0
        self.current_step = 0
        self.steps = steps
        # self.action_values = q*(t)
        self.action_values = (torch.rand(k)*99.0) + 1.0
        
    def give_reward(self, arm_selected):
        self.reward_t = Normal(loc = self.action_values[arm_selected], scale=self.std).sample()
        self.current_step += 1
        return self.reward_t.item()

    def reset(self):
        self.current_step = 0
        self.action_values = (torch.rand(self.k)*99.0) + 1.0
