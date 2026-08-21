import datetime

import utils.register
from hw5_config import multi_env, retrain_env
from utils.my_utils import save_info_executions
from exec_functions.test_retrained_agent import test_retrained_agent_1

e = "highwayMA-v1"

n_car_c11 = retrain_env["controlled_vehicles"] + retrain_env["vehicles_count"]
n_car_cma = multi_env["controlled_vehicles"] + multi_env["vehicles_count"]

assert n_car_cma == 6
assert n_car_cma == n_car_c11

#best_adv = "out/HighwayEnvMA/DQNAgent/run_20260713-140537_274789/checkpoint-best.tar"
best_purely_adv = "out/HighwayEnvMA/DQNAgent/run_20260710-163729_2738587/checkpoint-best.tar"
repetitions = 5

ego_list = [
"out/HighwayEnvMA/DQNAgent/run_20260804-175817_421106/checkpoint-best.tar",
"out/HighwayEnvMA/DQNAgent/run_20260804-181202_421106/checkpoint-best.tar",
"out/HighwayEnvMA/DQNAgent/run_20260804-182437_421106/checkpoint-best.tar",
"out/HighwayEnvMA/DQNAgent/run_20260804-183756_421106/checkpoint-best.tar",
"out/HighwayEnvMA/DQNAgent/run_20260804-185105_421106/checkpoint-best.tar",
"out/HighwayEnvMA/DQNAgent/run_20260804-190406_421106/checkpoint-best.tar",
"out/HighwayEnvMA/DQNAgent/run_20260804-191734_421106/checkpoint-best.tar",
"out/HighwayEnvMA/DQNAgent/run_20260804-193001_421106/checkpoint-best.tar",
"out/HighwayEnvMA/DQNAgent/run_20260804-194302_421106/checkpoint-best.tar",
"out/HighwayEnvMA/DQNAgent/run_20260804-195549_421106/checkpoint-best.tar",
]

file_name = 'hw_sc5_test_retrained_ego_purely_adv' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv'

for ego in ego_list:
    for i in range(repetitions):

        result = test_retrained_agent_1(e,
                            ego,
                            best_purely_adv,
                            retrain_env,
                            multi_env,
                            "./models/dqn.json",
                            2000,
                            0.5,
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
                                0.5
                                )