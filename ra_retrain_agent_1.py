import datetime

import utils.register
from ra_config import multi_env, rt_env
from utils.my_utils import save_info_executions
from exec_functions.retrain_agent_1 import retrain_agent_1

e = "roundaboutMA-v1"


ego = "/home/andrea/adversarial_agent/code/out/RoundaboutEnv/DQNAgent/run_20250807-224434_1390372/checkpoint-best.tar"

best_purely_adv = "out/RoundaboutEnvMA/DQNAgent/run_20260717-141703_2576922/checkpoint-best.tar"
repetitions = 10

file_name = 'ra_retrain_ego_purely_adv' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv'



for i in range(repetitions):

    result = retrain_agent_1(e,
                    ego,
                    best_purely_adv,
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
                        best_purely_adv,
                        2.0
                        )

