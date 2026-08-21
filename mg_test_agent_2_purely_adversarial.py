import datetime

import utils.register
from mg_config import single_env, multi_env
from utils.my_utils import save_info_executions
from exec_functions.test_agent_2 import test_agent_2

e = "mergeMA-v1"

ego = "/Users/andrea.doreste/Documents/GitHub/pbt_v1/out/MergeEnv/DQNAgent/run_20251201-163254_10296/checkpoint-final.tar"

adv_list = [
    "./out/MergeEnvMA/DQNAgent/run_20260714-120041_78262/checkpoint-final.tar",
    "./out/MergeEnvMA/DQNAgent/run_20260714-120923_78262/checkpoint-final.tar",
]

repetitions = 2

file_name = 'mg_test_purely_adv' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv'

for adv in adv_list:
    adv_agent = adv
    print(f"adv: {adv}")
    for i in range(repetitions):
        results = test_agent_2(e,
                     ego,
                     adv_agent,
                     single_env,
                     multi_env,
                     "./models/dqn.json",
                     2000,
                     2.0,
                     mode = "purely_adversarial_2")
        
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
                             adv_agent,
                             0.5,
                            )