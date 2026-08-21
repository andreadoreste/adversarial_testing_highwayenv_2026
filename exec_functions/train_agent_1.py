import gymnasium as gym


from rl_agents.agents.common.factory import agent_factory, load_agent_config
from utils.multi_agent_evaluation import MultiAgentEvaluation

def train_agent_1(env,
                  ego_env_config,
                  model_json_file,
                  number_of_episodes
                  ):
    
    ego_env = gym.make(env, render_mode = "rgb_array") 

    if ego_env_config:
        ego_env.configure(ego_env_config)
    ego_env.reset()

    #Make Agent

    ego_config = load_agent_config(model_json_file)
    ego_agent = agent_factory(ego_env, ego_config)

    evaluation = MultiAgentEvaluation(ego_env,
                            ego_agent,
                            num_episodes = number_of_episodes,
                            mode = "single_agent")   
    evaluation.train()

    ego_crashes = evaluation.ego_crashes
    directory = evaluation.run_directory
    episode = evaluation.episode
    final_success_rate = evaluation.final_success_rate
    mean_success_rate = evaluation.mean_success_rate

    return ego_crashes, episode, directory, final_success_rate, mean_success_rate

