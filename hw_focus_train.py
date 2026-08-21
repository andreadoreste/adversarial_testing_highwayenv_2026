import csv
import time
import datetime

import utils.register
from hw5_config import multi_env,retrain_env
from utils.focus_training_utils import read_failures, focus_train_agent_1, update_failures_file_after_focus_train

e = "highwayMA-v1"

#ego = "/home/andrea/adversarial_agent/code/out/HighwayEnvMA/DQNAgent/run_20250716-223535_2466157/checkpoint-best.tar"
ego = "/home/andrea/adversarial_testing_rl/out/HighwayEnv/DQNAgent/run_20250721-001120_6644/checkpoint-best.tar"


purely_adv = "out/HighwayEnvMA/DQNAgent/run_20260710-163729_2738587/checkpoint-best.tar"

failures_path_list = [
"out/HighwayEnvMA/DQNAgent/run_20260711-135902_3239623/run_20260711-135902_3239623.csv",
"out/HighwayEnvMA/DQNAgent/run_20260711-140342_3239623/run_20260711-140342_3239623.csv",
"out/HighwayEnvMA/DQNAgent/run_20260711-140824_3239623/run_20260711-140824_3239623.csv",
"out/HighwayEnvMA/DQNAgent/run_20260711-141300_3239623/run_20260711-141300_3239623.csv",
"out/HighwayEnvMA/DQNAgent/run_20260711-141750_3239623/run_20260711-141750_3239623.csv",
"out/HighwayEnvMA/DQNAgent/run_20260711-142230_3239623/run_20260711-142230_3239623.csv",
"out/HighwayEnvMA/DQNAgent/run_20260711-142710_3239623/run_20260711-142710_3239623.csv",
"out/HighwayEnvMA/DQNAgent/run_20260711-143150_3239623/run_20260711-143150_3239623.csv",
"out/HighwayEnvMA/DQNAgent/run_20260711-143629_3239623/run_20260711-143629_3239623.csv",
"out/HighwayEnvMA/DQNAgent/run_20260711-144106_3239623/run_20260711-144106_3239623.csv",
]

file_name = 'hw_sc5_evaluate_validity_purely_adv_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv'

for i in range(len(failures_path_list)):
    
    start_time = time.time()
    path = failures_path_list[i]
    failures, df = read_failures(path)

    new_failure_list, valid_failures = focus_train_agent_1(e,
                                                            ego,
                                                            purely_adv,
                                                            retrain_env,
                                                            multi_env,
                                                            "./models/dqn.json",
                                                            50,
                                                            failures,
                                                            0.5,
                                                            )
    

    save_path = f'{path}_focus_training.csv'
    df_updated = update_failures_file_after_focus_train(new_failure_list, df, save_path)
    
    end_time = time.time()
    print(f'Focus training info saved in: {save_path}')

    ##Calculate success rate after dropping invalid failures
    total_episodes = df.shape[0]
    total_success = total_episodes - valid_failures
    updated_success_rate = total_success/total_episodes

    execution_time = end_time - start_time  # Calculate time taken
    execution_time_min = execution_time/60
    print(f"Updated success rate: {updated_success_rate}")
    print(f"Execution time: {execution_time:.4f} seconds")
    print(f"Execution time: {execution_time_min:.4f} minutes")
    print(f"Valid failures: {valid_failures}")
    
    info = [path, save_path, updated_success_rate, execution_time_min, valid_failures]
    with open(file_name,'a') as f_object:
        writer_object = csv.writer(f_object)
        writer_object.writerow(info)

    f_object.close()