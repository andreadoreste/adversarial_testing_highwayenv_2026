import datetime

import utils.register
from ra_config import single_env
from utils.my_utils import save_info_executions
from exec_functions.train_agent_1 import train_agent_1

e = "roundaboutMA-v1"
file_name = 'ra_train_agent_1' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv'

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


    