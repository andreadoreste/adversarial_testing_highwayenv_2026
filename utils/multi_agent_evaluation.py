import os
import csv
import time
import json
import random
import datetime
import logging
import warnings
import numpy as np
from collections import deque


import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from rl_agents.trainer.evaluation import Evaluation
from rl_agents.configuration import serialize
import rl_agents.trainer.logger

logger = logging.getLogger(__name__)

class MultiAgentEvaluation(Evaluation):

    def __init__(self,
                 env, 
                 agent, 
                 agent_0 = None,
                 num_episodes = 2000,
                 mode = "adversarial",
                 display_env = True,
                 step_callback_fn = None,
                 factor = 0.1,
                 run_directory = None,
                 budget = None):
        

        super().__init__(env,
                     agent,
                     directory = None,
                     run_directory = run_directory, 
                     num_episodes = num_episodes,
                     training = True, 
                     sim_seed = env.seed, 
                     display_env = display_env,
                     display_agent = True,
                     display_rewards = True,
                     close_env = True,
                     step_callback_fn = step_callback_fn)
        
        self.agent_0 = agent_0
        self.factor = factor #todo
        self.ego_adv_crashes = 0
        self.final_success_rate = None
        self.mean_success_rate = None
        self.adv_crashes = 0
        self.ego_crashes = 0
        self.ego_success = None #todo
        self.success_counter = 0

        max_len = 100
        self.total_reward_queue = deque(maxlen=max_len)
        self.return_queue = deque(maxlen=max_len)
        self.length_queue = deque(maxlen=max_len)
        self.success_queue = deque(maxlen=max_len)
        self.ego_success_queue = deque(maxlen=max_len)
        self.adv_success_queue = deque(maxlen=max_len)

        self.info = None

        self.file_name = str(os.path.basename(self.run_directory))
        self.data_folder_name = os.path.join(self.run_directory,self.file_name)
        
        self.mode = mode

        self.budget = budget #for focus training

        if self.budget is not None:
            prefix = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') 
            last_episode_trigger = lambda ep: ep == self.budget - 1
            self.wrapped_env = RecordVideo(self.env,video_folder=self.run_directory, episode_trigger= last_episode_trigger, name_prefix=prefix)
            
            try:
                self.wrapped_env.unwrapped.set_record_video_wrapper(self.wrapped_env)
            except AttributeError:
                pass
        else:
            self.write_metadata()
            self.write_logging()
    
    def train_agent_2(self):
        self.training = True

        print('self.mode: ', self.mode)
        try:
            self.agent_0.eval()
        except AttributeError:
            pass

        self.run_multi_agent_episodes()
        self.close()
    
    def test_agent_2(self):
        self.training = False
        try:
			#changing the policy for both agents 
            self.agent_0.eval()
            self.agent.eval()
        except AttributeError:
            pass

        self.run_multi_agent_episodes()
        self.close()

    def retraining_mode(self):
        self.training = True
        try:
            self.agent_0.eval()
        except AttributeError:
            pass

        self.run_multi_agent_episodes()
        self.close()

    def multi_adversarial_retraining_mode(self, list_of_agents, adv_file_list):
        self.training = True
        #agent_0.eval will occur on run_multi_adversarial_retraining
        self.run_multi_adversarial_retrain_episodes(list_of_agents, adv_file_list)
        self.close()


    def test_retrained_agent(self):
        self.training = False
        try:
            self.agent_0.eval()
            self.agent.eval()
        except AttributeError:
            pass
        self.run_multi_agent_episodes()
        self.close()


    def focus_train_agent_1(self, replay_episode = 0):
        try:
            self.agent_0.eval()
        except AttributeError:
            pass

        try:
            self.run_focus_train(replay_episode)
        finally:
            self.close()

    def run_multi_agent_episodes(self):

        for self.episode in range(self.num_episodes):
            terminal = False
            self.reset(seed=self.episode)
            self.r_step_collection = []
            rewards = []
            start_time = time.time()

            while not terminal:
                if self.mode in ("adversarial", "purely_adversarial"):
                    reward, terminal = self.multi_agent_step()

                elif self.mode == "retraining":
                    reward, terminal = self.retraining_step()

                else:
                    warnings.warn(
                        f"Unknown mode '{self.mode}'. Expected one of: "
                        f"'adversarial_2', 'purely_adversarial_2', 'retraining'. "
                        f"Stopping execution."
                    )
                    terminal = True
                    break
                    
                #Catch interruptions
                try:
                    if self.env.unwrapped.done:
                        break
                except AttributeError:
                    pass
                
                rewards.append(reward)
            
            duration = time.time() - start_time
            self.after_all_episodes(self.episode, rewards, duration)
            self.after_some_episodes(self.episode, rewards)

    def run_focus_train(self, replay_episode):

        for self.episode in range(self.budget):
            terminal = False
            self.reset(seed = replay_episode)
            while not terminal:
                try:
                    _, terminal = self.retraining_step()
                except Exception:
                    logger.exception(f"focus_train step failed at episode {self.episode}")
                    raise
                try:
                    if self.env.unwrapped.done:
                        break
                except AttributeError:
                    pass
            self.ego_success = not(self.info["crashed_list"][0])
            if self.ego_success == True:
                break

    def run_multi_adversarial_retrain_episodes(self, list_of_agents, adv_file_list):
        list_of_adversarials = list_of_agents
        list_of_adversarials_file = adv_file_list

        for self.episode in range(self.num_episodes):
            start_time = time.time()

            #set the agent
            adv_choice = random.randint(0, len(list_of_adversarials)-1)

            adv_file = list_of_adversarials_file[adv_choice]
            self.agent_0 = list_of_adversarials[adv_choice]
            self.agent_0.eval() #changing the policy for adversarial agent
            print("agent_0: ", self.agent_0)

            terminal = False
            self.reset(seed=self.episode)

            rewards = []
            adv_rewards = []

            while not terminal:

                reward, terminal = self.retraining_step()
                adv_reward = self.info["agents_rewards"][1]

                rewards.append(reward)
                adv_rewards.append(adv_reward)

                #Catch interruptions
                try:
                    if self.env.unwrapped.done:
                        break
                except AttributeError:
                    pass

            #End of episode
            duration = time.time() - start_time
            self.after_all_episodes(self.episode, rewards, duration)
            self.after_some_episodes(self.episode, rewards)

            #save info into file
            episode_info = [self.episode, adv_file]
            self.this_file_name = str(self.data_folder_name) + "adv_per_episode.csv"

            with open(self.this_file_name, 'a') as f_object:
                writer_object = csv.writer(f_object)
                writer_object.writerow(episode_info)



    def multi_agent_step(self):

        ##Actions
        ego_obs = self.observation[0]
        ego_action = self.agent_0.plan(ego_obs)

        adv_obs = self.observation[1]
        adv_action = self.agent.plan(adv_obs)

        if not (ego_action and adv_action):
        #if not ([ego_action] and adv_action):
            raise Exception("The agents did not plan any action")
        
        ##Forward actions to the environment viewer
        try:
            self.env.unwrapped.viewer.set_agent_action_sequence((ego_action, adv_action))
        except AttributeError:
            pass
        
        ##Step Environment
        previous_observation, adv_action, ego_action = [ego_obs, adv_obs], adv_action[0], ego_action[0]
        transition = self.wrapped_env.step((ego_action, adv_action))
    
        self.observation, reward, done, truncated, info = transition
        ego_r_qod = info["agents_rewards"][0]
        adv_r_qod = info["agents_rewards"][1]

        ego_obs = self.observation[0]
        adv_obs = self.observation[1]

        self.info = info.copy()
        self.done = done   
        self.truncated = truncated

        terminal = done or truncated
        r_step = 0
        #Call callback function
        if self.step_callback_fn is not None:
            self.step_callback_fn(self.episode,
                                  self.wrapped_env,
                                  transition,
                                  self.data_folder_name,
                                  self.num_episodes)
        
        #Mode: adversarial, purely_adversarial

        if self.mode == "adversarial":
            #print('adversarial_2')
            r_diff = info["r_diff"]
            r_step = adv_r_qod + (r_diff * self.factor)

            distance = info["dist_ego_adv"]
            r = 1/(1 + distance)
            #self.r_step_collection.append(r_step)


            #if r <= 0:
                #print('r: ', r)
            self.r_step_collection.append(r)

            if terminal and info["crashed"]:
                #print('info: ', info['crashed_list'])
                k = 10 #todo
            
                #r_ep = max(k* (sum(self.r_step_collection)),0)
                r_ep = k * (sum(self.r_step_collection))
                #print('r_ep: ', r_ep)
                #print('epis: ', self.episode)
                #print('r_ep: ', r_ep)
                r_step += r_ep

                

        elif self.mode == "purely_adversarial":
            print('purely_adversarial_2')        
            r_step = info["r_diff"]
            
            distance = info["dist_ego_adv"]
            r = 1/(1 + distance)

            self.r_step_collection.append(r)

            if terminal and info["crashed"]:
                #print('info: ', info['crashed_list'])
                k = 10 #todo

                r_ep = k * (sum(self.r_step_collection))
                r_step += r_ep

        
        else:
            raise ValueError(f"Unknown mode '{self.mode}'. Must be one of: 'adversarial', 'purely_adversarial'")

        i = {}
        try:
            #adversarial as main agent
            self.agent.record(
                previous_observation[1],
                adv_action,
                r_step,
                adv_obs,
                done,
                i
            )
            
            #ego as agent_0
            self.agent_0.record(
                previous_observation[0],
                ego_action,
                ego_r_qod,
                ego_obs,
                done, 
                i
            )
            
        except NotImplementedError:
            pass
        
        return r_step, terminal

    def retraining_step(self):
        
        ego_obs = self.observation[0]
        ego_action = self.agent.plan(ego_obs)
        
        adv_obs = self.observation[1]
        adv_action = self.agent_0.plan(adv_obs)

        if not (ego_action and adv_action):
        #if not([ego_action] and adv_action):
            raise Exception("The agents did not plan any action")
        
        #Forward the actions to the environment viewer
        try:
            self.env.unwrapped.viewer.set_agent_action_sequence((ego_action, adv_action))
        except AttributeError:
            pass

        #step the environment
        previous_observation, adv_action, ego_action = [ego_obs, adv_obs], adv_action[0], ego_action[0]

        transition = self.wrapped_env.step((ego_action, adv_action))
        self.observation, reward, done, truncated, info = transition

        ego_reward = info["agents_rewards"][0]

        ego_obs = self.observation[0]
        adv_obs = self.observation[1]

        self.info = info.copy()
        self.done = done
        self.truncated = truncated

        terminal = done or truncated

        #Call callback function
        if (self.step_callback_fn is not None) and (self.budget is None):
            self.step_callback_fn(self.episode,
                                  self.wrapped_env, 
                                  transition, 
                                  self.data_folder_name,
                                  self.num_episodes)
 
        i = {}
        try:
            self.agent.record(previous_observation[0], ego_action, ego_reward, ego_obs, done, i)
        except NotImplementedError:
            pass
        return ego_reward, terminal

    def after_all_episodes(self, episode, rewards, duration):
        rewards = np.array(rewards)
        gamma = self.agent.config.get("gamma", 1)

        self.total_reward_queue.append(sum(rewards))
        self.return_queue.append(sum(r*gamma**t for t, r in enumerate(rewards)))
        self.length_queue.append(len(rewards))

        crashed = self.info["crashed"]
        if crashed == False:
            self.success_queue.append(1)
            self.success_counter += 1
        else:
            self.success_queue.append(0)
            self.ego_adv_crashes += 1
        
        ego_crashed = self.info["crashed_list"][0]
        if ego_crashed == False:
            self.ego_success_queue.append(1)
        else:
            self.ego_success_queue.append(0)
            self.ego_crashes += 1

        self.writer.add_scalar("episode/length", len(rewards), episode)
        self.writer.add_scalar("episode/total_reward", sum(rewards), episode)
        self.writer.add_scalar("episode/return", sum(r*gamma**t for t, r in enumerate(rewards)), episode)
        self.writer.add_scalar("episode/fps", len(rewards) / max(duration, 1e-6), episode)
        self.writer.add_histogram("episode/rewards", rewards, episode)

        #Plot mean of 100 in tensorboard    
        self.writer.add_scalar("mean/mean_total_reward", np.mean(self.total_reward_queue), episode)
        self.writer.add_scalar('mean/mean_return', np.mean(self.return_queue), episode)
        self.writer.add_scalar('mean/mean_length', np.mean(self.length_queue), episode)
        self.writer.add_scalar('mean/success_mean', np.mean(self.success_queue), episode)
        self.writer.add_scalar('mean/ego_success_mean', np.mean(self.ego_success_queue), episode)

        self.writer.add_scalar('success_rate/final_success_rate', self.success_counter/(episode+1), episode)
        self.writer.add_scalar('success_rate/ego_crashes', self.ego_crashes, episode)

        if not(self.mode == "single_agent"):

            adv_crashed = self.info["crashed_list"][1]
            if adv_crashed == False:
                self.adv_success_queue.append(1)
            else:
                self.adv_success_queue.append(0)
                self.adv_crashes += 1
            
            self.writer.add_scalar('mean/adv_success_mean', np.mean(self.adv_success_queue), episode)

            self.writer.add_scalar('success_rate/adv_crashes', self.adv_crashes, episode)
            self.writer.add_scalar('success_rate/ego_adv_crashes', self.ego_adv_crashes, episode)

        self.mean_success_rate = np.mean(self.success_queue)
        self.final_success_rate = self.success_counter/(episode+1)

        logger.info("Episode {} score: {:.1f}".format(episode, sum(rewards)))

    def write_metadata(self):
        if self.budget is None:
            if self.agent_0:
                metadata = dict(env=serialize(self.env), agent=serialize(self.agent),agent_0 = serialize(self.agent_0))
            else:
                metadata = dict(env=serialize(self.env), agent=serialize(self.agent))
            file_infix = '{}.{}'.format(id(self.wrapped_env), os.getpid())
            file = self.run_directory / self.METADATA_FILE.format(file_infix)
            with file.open('w') as f:
                json.dump(metadata, f, sort_keys=True, indent=4)
    
    def close(self):
        """
			Close the evaluation.
		"""
        if self.training and (self.budget is None):
            self.save_agent_model("final")
        self.wrapped_env.close()
        self.writer.close()
        if self.close_env:
            self.env.close()
    
    def reset(self, seed=0):
        seed = self.sim_seed + seed if self.sim_seed is not None else None
        self.observation, info = self.wrapped_env.reset()
        if not(self.mode == "focus_train"): #not including retraining
            self.agent.seed(seed)  # Seed the agent with the main environment seed
		
        self.agent.reset()
        if self.agent_0:
            self.agent_0.seed(seed)
            self.agent_0.reset()
