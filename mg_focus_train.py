import csv
import time
import datetime

import utils.register
from mg_config import multi_env,rt_env
from utils.focus_training_utils import read_failures, focus_train_agent_1, update_failures_file_after_focus_train

e = "mergeMA-v1"

ego = "/home/andrea/adversarial_agent/code/out/MergeEnv/DQNAgent/run_20250720-212513_399326/checkpoint-final.tar"
purely_adv = "out/MergeEnvMA/DQNAgent/run_20260714-143236_955323/checkpoint-final.tar"

failures_path_list = [
"out/MergeEnvMA/DQNAgent/run_20260715-031403_1323776/run_20260715-031403_1323776.csv",
"out/MergeEnvMA/DQNAgent/run_20260715-031932_1323776/run_20260715-031932_1323776.csv",
"out/MergeEnvMA/DQNAgent/run_20260715-032506_1323776/run_20260715-032506_1323776.csv",
"out/MergeEnvMA/DQNAgent/run_20260715-033033_1323776/run_20260715-033033_1323776.csv",
"out/MergeEnvMA/DQNAgent/run_20260715-033607_1323776/run_20260715-033607_1323776.csv",
"out/MergeEnvMA/DQNAgent/run_20260715-034142_1323776/run_20260715-034142_1323776.csv",
"out/MergeEnvMA/DQNAgent/run_20260715-034715_1323776/run_20260715-034715_1323776.csv",
"out/MergeEnvMA/DQNAgent/run_20260715-035243_1323776/run_20260715-035243_1323776.csv",
"out/MergeEnvMA/DQNAgent/run_20260715-035814_1323776/run_20260715-035814_1323776.csv",
"out/MergeEnvMA/DQNAgent/run_20260715-040345_1323776/run_20260715-040345_1323776.csv",
]

file_name = 'mg_evaluate_validity_purely_adv' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv'

for i in range(len(failures_path_list)):
    
    start_time = time.time()
    path = failures_path_list[i]
    failures, df = read_failures(path)

    new_failure_list, valid_failures = focus_train_agent_1(e,
                                                            ego,
                                                            purely_adv,
                                                            rt_env,
                                                            multi_env,
                                                            "./models/dqn.json",
                                                            50,
                                                            failures,
                                                            2.0,
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