import datetime

import utils.register
from ra_config import single_env
from utils.my_utils import save_info_executions
from exec_functions.test_agent_1 import test_agent_1

e = "roundaboutMA-v1"

#retrained_ego_adv = "out/RoundaboutEnvMA/DQNAgent/run_20260802-131618_141365/checkpoint-best.tar"
retrained_ego_ma_adv = "out/RoundaboutEnvMA/DQNAgent/run_20260813-191319_1375555/checkpoint-best.tar"

repetitions = 10

file_name = 'ra_test_retrained_ma_ego_adv_npc' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv'

for i in range(repetitions):

    results = test_agent_1(e,
                        retrained_ego_ma_adv,
                        single_env,
                        "./models/dqn.json",
                        2000
                        )

    ego_crashes, directory, episode, final_success_rate, mean_success_rate = results

    save_info_executions(file_name,
                         retrained_ego_ma_adv,
                         i,
                         directory,
                         ego_crashes, episode,
                         final_success_rate,
                         mean_success_rate)
