import torch
import matplotlib.pyplot as plt
import os

DATA_FOLDER = "output"
PLOTS_FOLDER = "plots"
NUM_AGENTS = 6

def load_and_plot():
    os.makedirs(PLOTS_FOLDER, exist_ok=True)

    try:
        reward_values = torch.load(f"{DATA_FOLDER}/reward_values.pt")
        optimal_action_values = torch.load(f"{DATA_FOLDER}/optimal_action_values.pt")
    except FileNotFoundError:
        print(f"Error: Could not find data files in '{DATA_FOLDER}'. Run testbed.py first.")
        return

    mean_rewards = reward_values.mean(dim=1)
    mean_optimal_actions = optimal_action_values.mean(dim=1) * 100  

    labels = [
        "Greedy (\u03b5=0.0)", 
        "Epsilon-Greedy (\u03b5=0.1)", 
        "Epsilon-Greedy (\u03b5=0.01)", 
        "UCB (c=1)", 
        "UCB (c=5)", 
        "Gradient (\u03b1=0.1)"
    ]

    # Define 6 distinct markers (circle, x, triangle, square, diamond, star)
    markers = ['o', 'x', '^', 's', 'd', '*']

    plt.figure(figsize=(14, 6))

    # Plot A: Average Rewards
    plt.subplot(1, 2, 1)
    for i in range(NUM_AGENTS):
        plt.plot(
            mean_rewards[i].numpy(), 
            label=labels[i], 
            linewidth=1.5, 
            marker=markers[i], 
            markevery=100,  # Space symbols out every 100 steps
            markersize=6
        )
        
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title("Average Reward over 2,000 Runs")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot B: % Optimal Action
    plt.subplot(1, 2, 2)
    for i in range(NUM_AGENTS):
        plt.plot(
            mean_optimal_actions[i].numpy(), 
            label=labels[i], 
            linewidth=1.5, 
            marker=markers[i], 
            markevery=100,  # Space symbols out every 100 steps
            markersize=6
        )
        
    plt.xlabel("Steps")
    plt.ylabel("% Optimal Action")
    plt.title("Optimal Action % over 2,000 Runs")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    save_path = f"{PLOTS_FOLDER}/agent_comparison_results.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    load_and_plot()