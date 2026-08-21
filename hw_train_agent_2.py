import datetime

import utils.register
from hw3_config import single_env, multi_env
from utils.my_utils import save_info_executions
from exec_functions.train_agent_2 import train_agent_2

e = "highwayMA-v1"

#5NPCs
#ego = "/home/andrea/adversarial_testing_rl/out/HighwayEnv/DQNAgent/run_20250721-001120_6644/checkpoint-best.tar"
#3NPCs
ego = "/home/andrea/adversarial_agent/code/out/HighwayEnvMA/DQNAgent/run_20250716-223535_2466157/checkpoint-best.tar"
repetitions = 10

n_car_c11 = single_env["controlled_vehicles"] + single_env["vehicles_count"]
n_car_cma = multi_env["controlled_vehicles"] + multi_env["vehicles_count"]

file_name = 'hw_sc3_train_adv' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv'
for i in range(repetitions):
    
    results = train_agent_2(e,
                ego,
                None,                 #adv_trained_file = ego_trained_file
                single_env,
                multi_env,
                "./models/dqn.json",
                2000,
                0.5,
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
                         n_car_c11,
                         n_car_cma)


