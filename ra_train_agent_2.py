import datetime

import utils.register
from ra_config import single_env, multi_env
from utils.my_utils import save_info_executions
from exec_functions.train_agent_2 import train_agent_2

e = "roundaboutMA-v1"

ego =  "/Users/andrea.doreste/Documents/GitHub/adversarial_agent/code/out/RoundaboutEnv/DQNAgent/run_20251205-154131_57013/checkpoint-best.tar"

repetitions = 2

file_name = 'ra_train_adv' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv'

for i in range(repetitions):
    
    results = train_agent_2(e,
                ego,
                None,                 #adv_trained_file = ego_trained_file
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
                         2.0,
                        )