import datetime

import utils.register
from ra_config import single_env, multi_env
from utils.my_utils import save_info_executions
from exec_functions.test_agent_2 import test_agent_2

e = "roundaboutMA-v1"

ego = "/Users/andrea.doreste/Documents/GitHub/adversarial_agent/code/out/RoundaboutEnv/DQNAgent/run_20251205-154131_57013/checkpoint-best.tar"

adv_list = [
    "./out/RoundaboutEnvMA/DQNAgent/run_20260715-170448_10718/checkpoint-best.tar",
    "./out/RoundaboutEnvMA/DQNAgent/run_20260715-171907_10718/checkpoint-best.tar"
]

repetitions = 1

file_name = 'ra_test_purely_adv' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv'

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
                             2.0,
                            )