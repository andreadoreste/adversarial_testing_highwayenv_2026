import datetime

import utils.register
from hw5_config import single_env, multi_env
from utils.my_utils import save_info_executions
from exec_functions.test_agent_2 import test_agent_2


e = "highwayMA-v1"

#5NPC
#ego = "/home/andrea/adversarial_testing_rl/out/HighwayEnv/DQNAgent/run_20250721-001120_6644/checkpoint-best.tar"
#3NPC
#ego = "/home/andrea/adversarial_agent/code/out/HighwayEnvMA/DQNAgent/run_20250716-223535_2466157/checkpoint-best.tar"
retrained_ego = "out/HighwayEnvMA/DQNAgent/run_20260731-191224_10745/checkpoint-best.tar"

adv_list = [
    "out/HighwayEnvMA/DQNAgent/run_20260710-161434_2739181/checkpoint-best.tar", #second best adv
    ]
repetitions = 10

file_name = 'hw_sc5_test_retrained_ego_second_best_adv' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv'

n_car_c11 = single_env["controlled_vehicles"] + single_env["vehicles_count"]
n_car_cma = multi_env["controlled_vehicles"] + multi_env["vehicles_count"]

for adv in adv_list:
    #adv_agent = f"./out/HighwayEnvMA/DQNAgent/{adv}/checkpoint-best.tar"
    adv_agent = adv
    print(f"adv: {adv}")
    for i in range(repetitions):
        results = test_agent_2(e,
                     retrained_ego,
                     adv_agent,
                     single_env,
                     multi_env,
                     "./models/dqn.json",
                     2000,
                     0.5,
                     mode = "adversarial_2")
        
        ego_adv_crashes, ego_crashes, adv_crashes, episode, directory, final_success_rate, mean_success_rate = results
        save_info_executions(file_name,
                             retrained_ego,
                             i,
                             directory,
                             ego_adv_crashes,
                             ego_crashes,
                             adv_crashes,
                             episode,
                             final_success_rate,
                             mean_success_rate,
                             adv_agent,
                             0.5,
                             n_car_c11,
                             n_car_cma)
        