import datetime

import utils.register
from mg_config import multi_env, rt_env
from utils.my_utils import save_info_executions
from exec_functions.retrain_agent_1 import retrain_agent_1

e = "mergeMA-v1"


ego = "/home/andrea/adversarial_agent/code/out/MergeEnv/DQNAgent/run_20250720-212513_399326/checkpoint-final.tar"

#best_adv = "out/MergeEnvMA/DQNAgent/run_20260714-135629_954738/checkpoint-final.tar"
best_purely_adv = "out/MergeEnvMA/DQNAgent/run_20260714-143236_955323/checkpoint-final.tar"
repetitions = 10

file_name = 'mg_retrain_ego_purely_adv' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv'


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

