import datetime

import utils.register
from hw3_config import single_env
from utils.my_utils import save_info_executions
from exec_functions.train_agent_1 import train_agent_1

file_name = 'hw_sc3_trai_agent_1_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv'

e = 'highwayMA-v1'
repetitions = 1

for i in range(repetitions):
    results = train_agent_1(e,
                single_env,
                "./models/dqn.json",
                2000)

    ego_crashes, directory, episode, final_success_rate, mean_success_rate = results

    save_info_executions(file_name,
                             None,
                             i,
                             directory,
                             ego_crashes, episode,
                             final_success_rate,
                             mean_success_rate)