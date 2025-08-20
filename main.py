import pandas as pd
from drl_modules.env import TradingEnv
from drl_modules.agent_render import render_loss_function
from drl_modules.data_extract import extract_data, extract_batched_data
from drl_modules.ppo import (
    ppo_run,
    ppo_eval,
    a2c_run,
    a2c_eval,
    dqn_run,
    dqn_eval,
)
from drl_modules.input_config import get_user_input, get_user_sim_method
from drl_modules.rewards import RewardFunctions
import os
from drl_modules.export_model import export_to_onnx, export_to_onnx_dict
import yaml

def get_path_from_input(path="results/"):
    folder = input("Enter name of the folder to save the results in: ")
    res_path = path+folder+"/"
    os.makedirs(res_path, exist_ok=True)
    return res_path

ALGO_DISPATCH = {
    "PPO": (ppo_run, ppo_eval),
    "A2C": (a2c_run, a2c_eval),
    "DQN": (dqn_run, dqn_eval),
}


def run_drl_system(
    reward_idx,
    symbol,
    batches_amount,
    df_path,
    step_inp,
    vec_env,
    res_path,
    device,
    policy,
    algorithm,
    eval_only=False,
    eval_episodes=100,
):
    start_part = 0

    df = extract_data(csv_path=df_path)
    run_fn, eval_fn = ALGO_DISPATCH[algorithm]

    if not eval_only:
        if batches_amount >= 4:
            batches = extract_batched_data(df, batch_divider=batches_amount)
            
            last_found_part = -1 
            for i in range(batches_amount):
                if os.path.exists(res_path+f"part{i}"):
                    last_found_part = i
                else:
                    break

            start_part = last_found_part + 1 if last_found_part != -1 else 0
            
            print(f"\n\nRUNNING THE TRAINING PHASE ({batches_amount} batches):")
            print(f"Starting from part {start_part}")

            for i, part in enumerate(batches[start_part:], start=start_part):
                print(f"Training on batch {i + 1}")
                # ../results/REWARD_FUNC_TEST_PPO/func_r/
                run_fn(
                    dir=res_path,
                    reward_func_idx=reward_idx,
                    symbol=symbol,
                    step_amount=step_inp,
                    df_path=part,
                    batch_idx=i,
                    batches_amount=batches_amount,
                    save_model_after_each_batch=True,
                    vectorized_environments=vec_env,
                    agent_policy=policy,
                    device=device,
                )
            eval_part = pd.concat(batches[-4:])

        else:
            print("\n\nRUNNING THE TRAINING PHASE (Single batch):")
            run_fn(
                dir=res_path,
                reward_func_idx=reward_idx,
                symbol=symbol,
                step_amount=step_inp,
                df_path=df,
                batch_idx=0,
                batches_amount=batches_amount,
                save_model_after_each_batch=True,
                vectorized_environments=vec_env,
                agent_policy=policy,
                device=device,
            )

        model_zip = f"{algorithm}_model.zip"
        if algorithm == "PPO":
            if policy == "MultiInputPolicy":
                export_to_onnx_dict(res_path + model_zip)
            else:
                export_to_onnx(res_path + model_zip)

                
    print("\n\nRUNNING THE EVALUATION PHASE:")
    eval_fn(
        dir=res_path,
        model_path=res_path + f"{algorithm}_model",
        episodes=eval_episodes,
        symbol=symbol,
        reward_func_idx=reward_idx,
        render_modulo=1,
        agent_policy=policy,
        df_path=df,
        eval_only_setting=eval_only,
    )
    
if __name__ == "__main__":
    method = get_user_sim_method()
    if not method:
        inputs = get_user_input()

        run_drl_system(
            reward_idx=inputs["Reward"],
            symbol=inputs["Symbol"],
            batches_amount=inputs["Batches"],
            df_path=inputs["Dataset"],
            step_inp=inputs["Steps"],
            vec_env=inputs["VectEnvs"],
            res_path=inputs["Folder"],
            device=inputs["Device"],
            policy=inputs["Policy"],
            algorithm=inputs["Algorithm"],
            eval_only=False,
        )
    else:
        data = None
        with open(method+"configuration.yaml", "r") as f:
            data = yaml.safe_load(f)
            
        run_drl_system(
            reward_idx=data["Reward"],
            symbol=data["Symbol"],
            batches_amount=data["Batches"],
            df_path=data["Dataset"],
            step_inp=data["Steps"],
            vec_env=data["VectEnvs"],
            res_path=data["Folder"],
            device=data["Device"],
            policy=data["Policy"],
            algorithm=data["Algorithm"],
            eval_only=True,
            eval_episodes=200,
        )