import datetime

import utils.register
from mg_config import multi_env, rt_env
from utils.my_utils import save_info_executions
from exec_functions.test_retrained_agent import test_retrained_agent_1

e = "mergeMA-v1"


#best_adv = "out/MergeEnvMA/DQNAgent/run_20260714-135629_954738/checkpoint-final.tar"
best_purely_adv = "out/MergeEnvMA/DQNAgent/run_20260714-143236_955323/checkpoint-final.tar"
repetitions = 5

ego_list = [
"out/MergeEnvMA/DQNAgent/run_20260803-212356_294385/checkpoint-final.tar",
"out/MergeEnvMA/DQNAgent/run_20260803-213108_294385/checkpoint-final.tar",
"out/MergeEnvMA/DQNAgent/run_20260803-213815_294385/checkpoint-final.tar",
"out/MergeEnvMA/DQNAgent/run_20260803-214517_294385/checkpoint-final.tar",
"out/MergeEnvMA/DQNAgent/run_20260803-215218_294385/checkpoint-final.tar",
"out/MergeEnvMA/DQNAgent/run_20260803-215927_294385/checkpoint-final.tar",
"out/MergeEnvMA/DQNAgent/run_20260803-220641_294385/checkpoint-final.tar",
"out/MergeEnvMA/DQNAgent/run_20260803-221354_294385/checkpoint-final.tar",
"out/MergeEnvMA/DQNAgent/run_20260803-222103_294385/checkpoint-final.tar",
"out/MergeEnvMA/DQNAgent/run_20260803-222813_294385/checkpoint-final.tar",
]

file_name = 'mg_test_retrained_ego_purely_adv' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv'

for ego in ego_list:
    for i in range(repetitions):

        result = test_retrained_agent_1(e,
                            ego,
                            best_purely_adv,
                            rt_env,
                            multi_env,
                            "./models/dqn.json",
                            2000,
                            2.0,
                            "retraining",
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