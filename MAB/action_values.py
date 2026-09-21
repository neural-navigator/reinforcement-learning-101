import torch


class EpsilonGreedy:
    def __init__(self, epsilon=0.0, k_arms=10, steps=1000):
        # REVIEW: Validate epsilon is in [0, 1] and that k_arms/steps are
        # positive before allocating state.
        self.epsilon = epsilon
        self.k_arms = k_arms
        self.steps = steps
        self.rewards = torch.zeros((self.k_arms, self.steps))
        self.current_step = 0
        self.arm_selection_count = torch.zeros(self.k_arms)
        self.Q_t = torch.zeros(self.k_arms)

    def try_action(self):
        tmp = torch.rand(1).item()
        if self.epsilon == 0.0 or tmp>= self.epsilon:
            global_max = torch.max(self.Q_t).item()
            match_indices = torch.nonzero(self.Q_t==global_max)
            selected_idx = torch.randint(0, match_indices.size(0), (1,)).item()
            selected_action = match_indices[selected_idx]
            selected_action_idx = selected_action[0].item()
        else:
            selected_action_idx = torch.randint(0, self.k_arms, (1,)).item()
        return selected_action_idx

    def collect_reward(self, env, selected_action_idx):
        reward = env.give_reward(selected_action_idx)
        self.rewards[selected_action_idx, self.current_step] = reward
        self.arm_selection_count[selected_action_idx] += 1
        self.Q_t[selected_action_idx] = self.rewards[selected_action_idx].sum().item() / self.arm_selection_count[selected_action_idx].item()
        self.current_step += 1
        return reward

    def reset(self):
        self.current_step = 0
        self.rewards = torch.zeros((self.k_arms, self.steps))
        self.Q_t = torch.zeros(self.k_arms)
        self.arm_selection_count = torch.zeros(self.k_arms)


class UCB:
    def __init__(self, c, k_arms=10, steps=1000):
        self.c = c
        self.k_arms = k_arms
        self.steps = steps
        self.current_step = 0
        self.arm_pull_count = torch.zeros(self.k_arms)
        self.Q_t = torch.zeros(self.k_arms)
        self.rewards = torch.zeros((self.k_arms, self.steps))

    def try_action(self):
        mask_arr = (self.arm_pull_count == 0)
        if mask_arr.any():
            return mask_arr.argmax().item()
        else:
            uncertainity_val = self.c * torch.sqrt(torch.log(torch.tensor(self.current_step, dtype=torch.float32))/ (self.arm_pull_count))
            updated_Q_t = self.Q_t + uncertainity_val
            arm_pulled = torch.argmax(updated_Q_t).item()
            return arm_pulled

    def collect_reward(self, env, arm_pulled):
        reward = env.give_reward(arm_pulled)
        self.current_step += 1
        self.arm_pull_count[arm_pulled] += 1
        self.rewards[arm_pulled, self.current_step] = reward
        self.Q_t[arm_pulled] = self.rewards[arm_pulled].sum().item() / self.arm_pull_count[arm_pulled]

    def reset(self):
        self.current_step = 0
        self.rewards = torch.zeros((self.k_arms, self.steps))
        self.Q_t = torch.zeros(self.k_arms)
        self.arm_pull_count = torch.zeros(self.k_arms)


class GradientBandit:
    def __init__(self, alpha, k_arms, steps=1000):
        self.alpha = alpha
        self.steps = steps
        self.k_arms = k_arms
        self.current_step = 0
        self.baseline = 0
        self.rewards = torch.zeros((self.k_arms, self.steps))
        self.preferences = torch.zeros(self.k_arms)

    def try_action(self):
        self.action_probabilities = torch.softmax(self.preferences, dim=0)
        action = torch.distributions.Categorical(self.action_probabilities).sample().item()
        return action

    def collect_reward(self, arm_pulled, env):
        reward = env.give_reward(arm_pulled)
        self.rewards[arm_pulled, self.current_step] = reward
        self.current_step += 1
        baseline_error = reward - self.baseline
        self.baseline = self.baseline + (baseline_error / self.current_step)
        tmp = torch.zeros(self.k_arms)
        tmp[arm_pulled] = 1
        self.preferences = self.preferences + self.alpha * baseline_error * (tmp - self.action_probabilities)
        return reward

    def reset(self):
        self.current_step = 0
        self.baseline = 0
        self.rewards = torch.zeros(self.k_arms)
        self.preferences = torch.zeros(self.k_arms)
        
