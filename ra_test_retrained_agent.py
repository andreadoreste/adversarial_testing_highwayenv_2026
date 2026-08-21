import datetime

import utils.register
from ra_config import multi_env, rt_env
from utils.my_utils import save_info_executions
from exec_functions.test_retrained_agent import test_retrained_agent_1

e = "roundaboutMA-v1"


best_adv = "out/RoundaboutEnvMA/DQNAgent/run_20260717-141656_2576212/checkpoint-best.tar"
repetitions = 5

ego_list = [
"out/RoundaboutEnvMA/DQNAgent/run_20260813-181221_1375555/checkpoint-best.tar",
"out/RoundaboutEnvMA/DQNAgent/run_20260813-182754_1375555/checkpoint-best.tar",
"out/RoundaboutEnvMA/DQNAgent/run_20260813-184314_1375555/checkpoint-best.tar",
"out/RoundaboutEnvMA/DQNAgent/run_20260813-185817_1375555/checkpoint-best.tar",
"out/RoundaboutEnvMA/DQNAgent/run_20260813-191319_1375555/checkpoint-best.tar",
"out/RoundaboutEnvMA/DQNAgent/run_20260813-181239_1375962/checkpoint-best.tar",
"out/RoundaboutEnvMA/DQNAgent/run_20260813-182752_1375962/checkpoint-best.tar",
"out/RoundaboutEnvMA/DQNAgent/run_20260813-184248_1375962/checkpoint-best.tar",
"out/RoundaboutEnvMA/DQNAgent/run_20260813-185740_1375962/checkpoint-best.tar",
"out/RoundaboutEnvMA/DQNAgent/run_20260813-191231_1375962/checkpoint-best.tar",
]

file_name = 'ra_test_ma_retrained_ego_advs' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv'

for ego in ego_list:
    for i in range(repetitions):

        result = test_retrained_agent_1(e,
                            ego,
                            best_adv,
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
                                best_adv,
                                2.0
                                )