import datetime

import utils.register
from hw3_config import single_env
from utils.my_utils import save_info_executions
from exec_functions.test_agent_1 import test_agent_1


e = "highwayMA-v1"

#ego = "/Users/andrea.doreste/Documents/GitHub/adversarial_agent/code/out/HighwayEnvMA/202507/run_20250721-001120_6644/checkpoint-best.tar"
#retrained_ego_adv = "out/HighwayEnvMA/DQNAgent/run_20260802-140142_142122/checkpoint-best.tar" #hw3
ego = "out/HighwayEnvMA/DQNAgent/run_20260819-162820_37230/checkpoint-best.tar"

repetitions = 1 

file_name = 'hw5_test_retrained_ego_purely_adv_npc' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv'

for i in range(repetitions):

    results = test_agent_1(e,
                           ego,
                           single_env,
                           "./models/dqn.json",
                            2000)

    ego_crashes, directory, episode, final_success_rate, mean_success_rate = results

    save_info_executions(file_name,
                         ego,
                         i,
                         directory,
                         ego_crashes, episode,
                         final_success_rate,
                         mean_success_rate)