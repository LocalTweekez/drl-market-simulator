import matplotlib.pyplot as plt
import pandas as pd

result_files = "../results/sb3_log/"
# csv_file = "../results/REWARD_FUNC_TEST_PPO/func_r/"
# csv_file = "../results/REWARD_FUNC_TEST_PPO/func_r/sb3_logs/"

def render_loss_function(csv_file=result_files):
    # Load the CSV data
    data = pd.read_csv(csv_file+"sb3_log/progress.csv")
    

    # Extract relevant columns
    iterations = data['time/iterations']
    value_loss = data['train/value_loss']
    policy_loss = data['train/policy_gradient_loss']
    total_loss = data['train/loss']

    # Create a new figure for the plot
    plt.figure(figsize=(10, 6))

    # Plot value loss
    plt.plot(iterations, value_loss, label='Value Loss', color='blue', linewidth=2)
    
    # Plot policy loss
    plt.plot(iterations, policy_loss, label='Policy Gradient Loss', color='green', linewidth=2)
    
    # Plot total loss
    plt.plot(iterations, total_loss, label='Total Loss', color='red', linewidth=2)
    
    # Add labels and title
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('PPO Agent Loss Function Over Time')
    plt.legend()
    plt.grid(True)

    # Save the figure as a PNG file
    plt.savefig(csv_file+"sb3_log/ppo_loss_function.png")

