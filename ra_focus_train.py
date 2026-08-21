import csv
import time
import datetime

import utils.register
from ra_config import multi_env,rt_env
from utils.focus_training_utils import read_failures, focus_train_agent_1, update_failures_file_after_focus_train

e = "roundaboutMA-v1"

ego = "/home/andrea/adversarial_agent/code/out/RoundaboutEnv/DQNAgent/run_20250807-224434_1390372/checkpoint-best.tar"
purely_adv = "out/RoundaboutEnvMA/DQNAgent/run_20260717-141703_2576922/checkpoint-best.tar"

failures_path_list = [
"out/RoundaboutEnvMA/DQNAgent/run_20260717-181956_3030687/run_20260717-181956_3030687.csv",
"out/RoundaboutEnvMA/DQNAgent/run_20260717-182721_3030687/run_20260717-182721_3030687.csv",
"out/RoundaboutEnvMA/DQNAgent/run_20260717-183449_3030687/run_20260717-183449_3030687.csv",
"out/RoundaboutEnvMA/DQNAgent/run_20260717-184206_3030687/run_20260717-184206_3030687.csv",
"out/RoundaboutEnvMA/DQNAgent/run_20260717-184935_3030687/run_20260717-184935_3030687.csv",
"out/RoundaboutEnvMA/DQNAgent/run_20260717-185656_3030687/run_20260717-185656_3030687.csv",
"out/RoundaboutEnvMA/DQNAgent/run_20260717-190421_3030687/run_20260717-190421_3030687.csv",
"out/RoundaboutEnvMA/DQNAgent/run_20260717-191137_3030687/run_20260717-191137_3030687.csv",
"out/RoundaboutEnvMA/DQNAgent/run_20260717-191857_3030687/run_20260717-191857_3030687.csv",
"out/RoundaboutEnvMA/DQNAgent/run_20260717-192621_3030687/run_20260717-192621_3030687.csv",
]

file_name = 'ra_evaluate_validity_purely_adv' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv'

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