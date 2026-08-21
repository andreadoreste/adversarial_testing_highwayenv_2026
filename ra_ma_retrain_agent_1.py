import random
import datetime

import utils.register
from ra_config import multi_env, rt_env
from utils.my_utils import save_info_executions
from exec_functions.ma_retrain_agent_1 import ma_retrain_agent_1

e = "roundaboutMA-v1"

ego = "/home/andrea/adversarial_agent/code/out/RoundaboutEnv/DQNAgent/run_20250807-224434_1390372/checkpoint-best.tar"

"""adv_list = [
"out/RoundaboutEnvMA/DQNAgent/run_20260717-135918_2576212/checkpoint-best.tar",
"out/RoundaboutEnvMA/DQNAgent/run_20260717-140803_2576212/checkpoint-best.tar",
"out/RoundaboutEnvMA/DQNAgent/run_20260717-141656_2576212/checkpoint-best.tar",
"out/RoundaboutEnvMA/DQNAgent/run_20260717-142551_2576212/checkpoint-best.tar",
"out/RoundaboutEnvMA/DQNAgent/run_20260717-143451_2576212/checkpoint-best.tar",
"out/RoundaboutEnvMA/DQNAgent/run_20260717-144358_2576212/checkpoint-best.tar",
"out/RoundaboutEnvMA/DQNAgent/run_20260717-145238_2576212/checkpoint-best.tar",
"out/RoundaboutEnvMA/DQNAgent/run_20260717-150121_2576212/checkpoint-best.tar",
"out/RoundaboutEnvMA/DQNAgent/run_20260717-151018_2576212/checkpoint-best.tar",
"out/RoundaboutEnvMA/DQNAgent/run_20260717-151911_2576212/checkpoint-best.tar",
]"""

n = 5
#chosen_adv = random.sample(adv_list, n)

chosen_adv = [
        'out/RoundaboutEnvMA/DQNAgent/run_20260717-144358_2576212/checkpoint-best.tar',
        'out/RoundaboutEnvMA/DQNAgent/run_20260717-142551_2576212/checkpoint-best.tar',
        'out/RoundaboutEnvMA/DQNAgent/run_20260717-141656_2576212/checkpoint-best.tar',
        'out/RoundaboutEnvMA/DQNAgent/run_20260717-151911_2576212/checkpoint-best.tar',
        'out/RoundaboutEnvMA/DQNAgent/run_20260717-135918_2576212/checkpoint-best.tar'
]

#print(chosen_adv)

repetitions = 5

file_name = 'ra_ma_retrain_ego_adv' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv'

for i in range(repetitions):
    result = ma_retrain_agent_1(e,
                                ego,
                                chosen_adv,
                                rt_env,
                                multi_env,
                                "./models/dqn.json",
                                2000,
                                2.0,
                                "retraining",
                                seed = None,
                                replay_episode = False,
                                budget = None,
                                run_folder = None
                                )

    save_info_executions(file_name,
                            ego,
                            i,
                            result["directory"],
                            result["ego_adv_crashes"],
                            result["ego_crashes"],
                            result["adv_crashes"],
                            result["episode"],
                            result["final_success_rate"],
                            result["mean_success_rate"],
                            result["advs_file_name"],
                            2.0,
                            chosen_adv
                            )
