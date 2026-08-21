import gymnasium as gym
from rl_agents.agents.common.factory import agent_factory, load_agent_config

from utils.multi_agent_evaluation import MultiAgentEvaluation
from utils.trajectory_collection_callback import trajectory_collection_callback

def test_agent_2(env,
                 ego_trained_file,
                 adv_trained_file,
                 ego_env_config,
                 adv_env_config,
                 model_json_file,
                 number_of_episodes,
                 factor,
                 mode):
    

    #Make env for ego

    ego_env = gym.make(env, render_mode = "rgb_array")
    ego_env.configure(ego_env_config)
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

    ##Adv
    adv_config = load_agent_config(model_json_file)
    adv_agent = agent_factory(adv_env, adv_config)
    adv_agent.load(adv_trained_file)

    evaluation = MultiAgentEvaluation(adv_env,
                                      adv_agent,
                                      ego_agent,
                                      number_of_episodes,
                                      mode,
                                      display_env = True,
                                      step_callback_fn = trajectory_collection_callback,
                                      factor = factor
                                      )
    evaluation.test_agent_2()

    ego_adv_crashes = evaluation.ego_adv_crashes
    ego_crashes = evaluation.ego_crashes
    adv_crashes = evaluation.adv_crashes
    directory = evaluation.run_directory
    episode = evaluation.episode
    final_success_rate = evaluation.final_success_rate
    mean_success_rate = evaluation.mean_success_rate

    return  ego_adv_crashes, ego_crashes, adv_crashes, episode, directory, final_success_rate, mean_success_rate

