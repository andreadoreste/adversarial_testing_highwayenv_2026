import gymnasium as gym
from rl_agents.agents.common.factory import agent_factory, load_agent_config
from utils.multi_agent_evaluation import MultiAgentEvaluation
from utils.trajectory_collection_callback import trajectory_collection_callback


def ma_retrain_agent_1(env,
                       ego_trained_file,
                       adv_trained_file_list,
                       ego_env_config,
                       adv_env_config,
                       model_json_file,
                       number_of_episodes,
                       factor,
                       mode,
                       seed,
                       replay_episode,
                       budget,
                       run_folder):

    #Make env for ego
    ego_env = gym.make(env, render_mode = "rgb_array")
    ego_env.configure(ego_env_config)

    ego_env.config.update({"seed" : None})

    ego_env.reset()

    #Make env for adv
    adv_env = gym.make(env, render_mode = "rgb_array")
    adv_env.configure(adv_env_config)
    adv_env.reset()

    #Make agents

    ##Ego
    ego_config = load_agent_config(model_json_file)
    ego_agent = agent_factory(ego_env, ego_config)
    ego_agent.load(ego_trained_file)


    ##ADVs

    adv_agent_list = []
    for adv_trained_file in adv_trained_file_list:
        print("adv_trained_file: ", adv_trained_file)
        adv_config = load_agent_config(model_json_file)
        adv_agent = agent_factory(adv_env, adv_config)
        adv_agent.load(adv_trained_file)
        adv_agent_list.append(adv_agent)

    print("env_name: ", ego_env.unwrapped.__class__.__name__)
    evaluation = MultiAgentEvaluation(ego_env,
                                      ego_agent,
                                      adv_agent,
                                      number_of_episodes,
                                      mode,
                                      step_callback_fn= trajectory_collection_callback,
                                      factor = factor,
                                      run_directory= run_folder,
                                      budget = budget
                                      )

    evaluation.multi_adversarial_retraining_mode(adv_agent_list, adv_trained_file_list)

    ego_adv_crashes = evaluation.ego_adv_crashes
    ego_crashes = evaluation.ego_crashes
    adv_crashes = evaluation.adv_crashes
    directory = evaluation.run_directory
    episode = evaluation.episode
    final_success_rate = evaluation.final_success_rate
    mean_success_rate = evaluation.mean_success_rate
    advs_file_name = evaluation.this_file_name

    result = {
        "ego_success" : evaluation.ego_success,
        "episode" : evaluation.episode,
        "ego_adv_crashes" : ego_adv_crashes,
        "ego_crashes" : ego_crashes,
        "adv_crashes" : adv_crashes,
        "directory": directory,
        "episode" : episode,
        "final_success_rate": final_success_rate,
        "mean_success_rate" : mean_success_rate,
        "advs_file_name" : advs_file_name,
        } 
    
    return result




    