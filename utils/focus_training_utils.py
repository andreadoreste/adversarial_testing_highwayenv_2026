import os
import datetime
import pandas as pd
import numpy as np

import time
import logging

from exec_functions.retrain_agent_1 import retrain_agent_1

def read_failures(path_to_csv):
    
    col_names = list(range(120))
    col_names.extend(['success','episode','seed'])
    df = pd.read_csv(path_to_csv, names=col_names)

    df['focus_training'] = None
    df['difficulty_level'] = None

    filtered_df = df[df['success'] == False]
    filtered_df = filtered_df[['success', 'episode','seed']]
    list_of_failures = filtered_df.to_dict('index')

    return list_of_failures, df

def focus_train_agent_1(env,
                        ego_trained_file,
                        adv_trained_file,
                        ego_env_config,
                        adv_env_config,
                        model_json_file,
                        budget,
                        list_of_failures,
                        factor):
    
    #run_folder = 'run_{}_{}'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), os.getpid())

    counter = 0
    success_counter = 0
    difficulty_level = None

    for failure_id in list_of_failures:

        iter_start = time.time()

        failure = list_of_failures[failure_id]
        seed = failure["seed"]
        replayed_episode = failure["episode"]

        run_folder = 'run_{}_{}_failure{}'.format(
        datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
        os.getpid(),
        failure_id
        )
        
        success = False
        
        result = retrain_agent_1(env,
                        ego_trained_file,
                        adv_trained_file,
                        ego_env_config,
                        adv_env_config,
                        model_json_file,
                        1,
                        factor,
                        "focus_train",
                        seed,
                        replayed_episode,
                        budget,
                        run_folder
                        )

        if result["ego_success"]:
            difficulty_level = (result["episode"] + 1)/budget
            success_counter += 1
            success = True
        else:
            difficulty_level = np.inf
        counter += 1
        
        #update dataframe
        failure["focus_training"] = success
        failure["difficulty_level"] = difficulty_level

    return list_of_failures, success_counter

def update_failures_file_after_focus_train(failures_dic, df,save_path):
    
    for id in failures_dic:
        failure = failures_dic[id]
        failure_ep = failure["episode"]
        df.loc[df["episode"] == failure_ep,"focus_training"] = failure["focus_training"]
        df.loc[df["episode"] == failure_ep, "difficulty_level"] = failure["difficulty_level"]

    df.to_csv(save_path)
    return df