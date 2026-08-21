import datetime

import utils.register
from mg_config import single_env
from utils.my_utils import save_info_executions
from exec_functions.test_agent_1 import test_agent_1

e = "mergeMA-v1"

retrained_ego_pure_adv = "out/MergeEnvMA/DQNAgent/run_20260803-213815_294385/checkpoint-final.tar"

repetitions = 10

file_name = 'mg_test_retrained_ego_purely_adv_npc' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv'

for i in range(repetitions):

    results = test_agent_1(e,
                        retrained_ego_pure_adv,
                        single_env,
                        "./models/dqn.json",
                        2000
                        )

    ego_crashes, directory, episode, final_success_rate, mean_success_rate = results

    save_info_executions(file_name,
                         retrained_ego_pure_adv,
                         i,
                         directory,
                         ego_crashes, episode,
                         final_success_rate,
                         mean_success_rate)
