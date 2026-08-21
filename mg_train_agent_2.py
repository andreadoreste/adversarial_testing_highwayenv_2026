import datetime

import utils.register
from mg_config import single_env, multi_env
from utils.my_utils import save_info_executions
from exec_functions.train_agent_2 import train_agent_2

e = "mergeMA-v1"

ego = "/Users/andrea.doreste/Documents/GitHub/pbt_v1/out/MergeEnv/DQNAgent/run_20251201-163254_10296/checkpoint-final.tar"

adv = "/Users/andrea.doreste/Documents/GitHub/pbt_v1/out/MergeEnv/DQNAgent/run_20251201-162149_10215/checkpoint-final.tar"

repetitions = 1

file_name = 'mg_train_adv' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv'

for i in range(repetitions):
    
    results = train_agent_2(e,
                ego,
                adv,                 #adv_trained_file = ego_trained_file
                single_env,
                multi_env,
                "./models/dqn.json",
                2000,
                2.0,
                mode = "adversarial_2")
    
    ego_adv_crashes, ego_crashes, adv_crashes, episode, directory, final_success_rate, mean_success_rate = results
    save_info_executions(file_name,
                         ego,
                         i,
                         directory,
                         ego_adv_crashes,
                         ego_crashes,
                         adv_crashes,
                         episode,
                         final_success_rate,
                         mean_success_rate,
                         0.5,
                        )