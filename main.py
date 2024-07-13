from drl_modules.env import TradingEnv
from drl_modules.agent_render import render_loss_function
from drl_modules.data_extract import extract_data, extract_batched_data
from drl_modules.agents import ppo_run, ppo_eval
from drl_modules.rewards import RewardFunctions
import os

def get_path_from_input(path="results/"):
    folder = input("Enter name of the folder to save the results in: ")
    res_path = path+folder+"/"
    os.makedirs(res_path, exist_ok=True)
    return res_path

def multiple_batch(reward_idx, batches_amount=4):
    path = "results/"
    start_part = 0

    df = extract_data()
    batches = extract_batched_data(df, batch_divider=batches_amount)
    res_path = get_path_from_input()
    
    step_inp = int(input("Enter amount of steps: "))
    symbol = input("Enter currency pair symbol in 6 capital letters: ")
    vec_env = int(input("Enter amount of vectorized environments (zero for base): "))
    
    
    last_found_part = -1
    for i in range(batches_amount):
        if os.path.exists(res_path+f"part{i}"):
            last_found_part = i
        else:
            break

    start_part = last_found_part + 1 if last_found_part != -1 else 0
    print(f"Starting from part {start_part}")

    for i, part in enumerate(batches[start_part:], start=start_part):
        print(f"Training on batch {i + 1}")
        # ../results/REWARD_FUNC_TEST_PPO/func_r/
        ppo_run(dir=res_path+f"part{i}/", 
                reward_func_idx=reward_idx, 
                symbol=symbol,
                step_amount=step_inp, 
                df_path=part,
                batch_idx=i,
                save_model_after_each_batch=True,
                vectorized_environments=vec_env)
                
    eval_part = pd.concat(batches[-4:])
    ppo_eval(dir=res_path+f"evaluation/", 
             episodes=100, 
             symbol=symbol,
             reward_func_idx=reward_idx,
             render_modulo=1,
             df_path=eval_part)

def single_batch(reward_idx):
    path = "results/"
    df = extract_data()
    res_path = get_path_from_input()

    symbol = input("Enter currency pair symbol in 6 capital letters: ")
    step_inp = int(input("Enter amount of steps: "))
    vec_env = int(input("Enter amount of vectorized environments (zero for base): "))
    
    print("\n\nRUNNING THE TRAINING PHASE:")
    ppo_run(dir=res_path, 
        reward_func_idx=reward_idx, 
        symbol=symbol,
        step_amount=step_inp, 
        df_path=df,
        batch_idx=0,
        save_model_after_each_batch=True,
        vectorized_environments=vec_env)
        
    print("\n\nRUNNING THE EVALUATION PHASE:")
    ppo_eval(dir=res_path, 
                episodes=100, 
                reward_func_idx=reward_idx,
                symbol=symbol,
                model_path=res_path+"PPO_model_batch_0",
                render_modulo=1,
                df_path=df)


if __name__ == "__main__":
    batches = int(input("Enter amount of batches (zero for single, minimum 4): "))
    print("\nReward functions:")
    rewards = RewardFunctions().function_names

    for i, r in enumerate(rewards):
        print(f"\t{i}. {r}")
    rw_idx = int(input("\nEnter reward function index: "))
    
    if batches == 0:
        single_batch(rw_idx)
    else:
        multiple_batch(rw_idx, batches)
    