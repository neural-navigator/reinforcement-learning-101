import os
from tqdm import tqdm

import torch
from k_bandit import KBandit
from action_values import EpsilonGreedy, UCB, GradientBandit


NUM_STEPS = 1000
NUM_RUNS = 2000
NUM_ARMS = 10
NUM_AGENTS = 6

OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

reward_values = torch.zeros((NUM_AGENTS, NUM_RUNS, NUM_STEPS))
action_values = torch.zeros((NUM_AGENTS, NUM_RUNS, NUM_STEPS))
arm_values = torch.zeros(NUM_RUNS, NUM_ARMS)
optimal_action_values = torch.zeros((NUM_AGENTS, NUM_RUNS, NUM_STEPS))

k_bandit_env = KBandit(k=NUM_ARMS, steps=NUM_STEPS)
greedy_agent = EpsilonGreedy(epsilon=0.0, k_arms=NUM_ARMS, steps=NUM_STEPS)
epsilon_agent_0_1 = EpsilonGreedy(epsilon=0.1, k_arms=NUM_ARMS, steps=NUM_STEPS)
epsilon_agent_0_0_1 = EpsilonGreedy(epsilon=0.01, k_arms=NUM_ARMS, steps=NUM_STEPS)
ucb_1 = UCB(c=1, k_arms=NUM_ARMS, steps=NUM_STEPS)
ucb_5 = UCB(c=5, k_arms=NUM_ARMS, steps=NUM_STEPS)
grad_agent = GradientBandit(alpha=0.1, k_arms=NUM_ARMS, steps=NUM_STEPS)


for z in tqdm(range(NUM_RUNS)):
    # initialize all agents
    k_bandit_env.reset()
    arm_values[z] = k_bandit_env.action_values
    optimal_value = torch.argmax(k_bandit_env.action_values).item()

    AGENTS = [greedy_agent, epsilon_agent_0_1, epsilon_agent_0_0_1, ucb_1, ucb_5, grad_agent]

    for idx, agent in enumerate(AGENTS):
        agent.reset()
        k_bandit_env.reset_steps()
        for x in tqdm(range(NUM_STEPS)):
            action = agent.try_action()
            reward = agent.collect_reward(env=k_bandit_env, arm_pulled=action)
            reward_values[idx, z, x] = reward
            action_values[idx, z, x] = action
            if action == optimal_value:
                optimal_action_values[idx, z, x] = 1
        print("completed for ith agent ", idx)

# save the values
torch.save(reward_values, f"{OUTPUT_FOLDER}/reward_values.pt")
torch.save(optimal_action_values, f"{OUTPUT_FOLDER}/optimal_action_values.pt")
torch.save(action_values, f"{OUTPUT_FOLDER}/action_values.pt")
torch.save(arm_values, f"{OUTPUT_FOLDER}/arm_values.pt")
