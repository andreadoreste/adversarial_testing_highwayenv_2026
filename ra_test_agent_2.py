import datetime

import utils.register
from ra_config import single_env, multi_env
from utils.my_utils import save_info_executions
from exec_functions.test_agent_2 import test_agent_2

e = "roundaboutMA-v1"

#ego = "/Users/andrea.doreste/Documents/GitHub/adversarial_agent/code/out/RoundaboutEnv/DQNAgent/run_20251205-154131_57013/checkpoint-best.tar"
#retrained_ego = "out/RoundaboutEnvMA/DQNAgent/run_20260813-191319_1375555/checkpoint-best.tar"

retrained_ego_list = [
"out/RoundaboutEnvMA/DQNAgent/run_20260813-181239_1375962/checkpoint-best.tar",
"out/RoundaboutEnvMA/DQNAgent/run_20260813-182752_1375962/checkpoint-best.tar",
"out/RoundaboutEnvMA/DQNAgent/run_20260813-184248_1375962/checkpoint-best.tar",
"out/RoundaboutEnvMA/DQNAgent/run_20260813-185740_1375962/checkpoint-best.tar",
"out/RoundaboutEnvMA/DQNAgent/run_20260813-191231_1375962/checkpoint-best.tar",
"out/RoundaboutEnvMA/DQNAgent/run_20260813-181221_1375555/checkpoint-best.tar",
"out/RoundaboutEnvMA/DQNAgent/run_20260813-182754_1375555/checkpoint-best.tar",
"out/RoundaboutEnvMA/DQNAgent/run_20260813-184314_1375555/checkpoint-best.tar",
"out/RoundaboutEnvMA/DQNAgent/run_20260813-185817_1375555/checkpoint-best.tar",
]

adv_list = [
        "out/RoundaboutEnvMA/DQNAgent/run_20260717-150121_2576212/checkpoint-best.tar" #second best adv
]

repetitions = 5

file_name = 'ra_test_retrained_ma_ego_second_best_adv_all' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv'

for retrained_ego in retrained_ego_list:
    adv_agent = adv_list[0]
    print(f"ego: {retrained_ego}")
    for i in range(repetitions):
        results = test_agent_2(e,
                     retrained_ego,
                     adv_agent,
                     single_env,
                     multi_env,
                     "./models/dqn.json",
                     2000,
                     2.0,
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
                             2.0,
                            )