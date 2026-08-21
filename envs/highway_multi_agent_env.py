from typing import Tuple, TypeVar

from gymnasium.envs.registration import register

import copy
import os
from typing import List, Tuple, Optional, Callable, TypeVar, Generic, Union, Dict, Text
import gymnasium as gym
from gymnasium import Wrapper
from gymnasium.wrappers import RecordVideo
from gymnasium.utils import seeding
import numpy as np
import math
import random

from highway_env.envs.highway_env import HighwayEnvFast
from highway_env.envs.common.action import action_factory, Action, DiscreteMetaAction, ActionType
from highway_env.envs.common.observation import observation_factory, ObservationType
from highway_env.envs.common.finite_mdp import finite_mdp
from highway_env.envs.common.graphics import EnvViewer
from highway_env.vehicle.behavior import IDMVehicle, LinearVehicle
from highway_env.vehicle.controller import MDPVehicle, ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env import utils
from highway_env.utils import near_split

Observation = TypeVar("Observation")

class HighwayEnvMA(HighwayEnvFast):

    metadata = {
        'render_modes': ['human', 'rgb_array'],
    }
    
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({"reward_vehicle" : "ego"}) #ego is the first agent, 'adversarial' for choosing the one that should defy the ego
        config.update({"overtaken" : False })
        config.update({"delta" : 0})
        config.update({"adv_factor": 0.1})
        return config

    def step(self, action: Action) -> Tuple[Observation, float, float, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminal, info)
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")
        self.time += 1 / self.config["policy_frequency"]
        
        self._simulate(action)

        obs = self.observation_type.observe()
        reward = self._reward(action)
        info = self._info(obs, action, reward)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        if self.render_mode == 'human':
            self.render()
        
        #returning reward for the ego vehicle (the adv reward was added to the info dict)
        returned_reward = reward[0]
        
        if self.config['reward_vehicle'] == 'adversarial':
            
            returned_reward = reward[1]
        
        return obs, returned_reward, terminated, truncated, info


    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """

        rewards_2 = tuple(self._agent_rewards(action, vehicle) for vehicle in self.controlled_vehicles)
        
        r = []
        for rewards in rewards_2:
            
            reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
            
            if self.config["normalize_reward"]:
                
                reward = utils.lmap(reward,
                                    [self.config["collision_reward"],
                                     self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                                    [0, 1])
                
            reward *= rewards['on_road_reward']

            r.append(reward)            
        
        return r

    def _agent_rewards(self, action: Action, vehicle:Vehicle) -> Dict[Text, float]:
        
        neighbours = self.road.network.all_side_lanes(vehicle.lane_index)
        lane = vehicle.target_lane_index[2] if isinstance(vehicle, ControlledVehicle) \
            else vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = vehicle.speed * np.cos(vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        
        
        return {
            "collision_reward": float(vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(vehicle.on_road)
        }

    def _info(self, obs: Observation, action: Optional[Action] = None, reward: Optional[float]=None) -> dict:
        """
        Return a dictionary of additional information

        :param obs: current observation
        :param action: current action
        :return: info dict with crash status for all vehicles
        """
        crashed_list = []
        for vehicle in self.controlled_vehicles:
            crashed_list.append(vehicle.crashed)
        
        crashed = False
        crashed = all(crashed_list)
        
        self.overtaken = False
        r_diff = None
        dist_ego_adv = None    
        if len(obs) == 2:
            
            ego_x_pos = obs[0][0]
        
            adv_x_pos = obs[1][0]
        
            dist_between_agents = adv_x_pos[1] - ego_x_pos[1]
            delta = self.config["delta"]
            #delta = 0.0025
            if dist_between_agents <= delta:
                self.overtaken = True
            
            r_diff = self.distance_reward(obs[0], obs[1])

            
            dist_ego_adv = self.distance_calculation(obs[0], obs[1])
        
        info = {
            "speed": self.vehicle.speed,
            "crashed": crashed,
            "action": action,
            "crashed_list": crashed_list,
            "overtaken": self.overtaken,
            "r_diff": r_diff,
            "dist_ego_adv" : dist_ego_adv
        }
        
        try:
            rewards_2 = tuple(self._agent_rewards(action, vehicle) for vehicle in self.controlled_vehicles)
            info["rewards"] = rewards_2
            info['agents_rewards'] = reward

        except NotImplementedError:
            pass
        
        return info
        
    #change _is_terminated to include the crash of the second vehicle
    def _is_terminated(self) -> bool:
        """The episode is over if one controlled vehicle crashed."""
        stop = False
        for vehicle in self.controlled_vehicles:
            if vehicle.crashed:
                stop = True
        
        return (stop or self.overtaken or
                self.config["offroad_terminal"] and not self.vehicle.on_road)
    
    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        number_controlled_vehicles = self.config["controlled_vehicles"]
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])
        random.seed(self.seed)


        if type(self.config["ego_spacing"]) == int or float:
        
            ego_spacing = self.config["ego_spacing"]
        
        if type(self.config["ego_spacing"]) == list:
            
            start = min(self.config["ego_spacing"])
            end = max(self.config["ego_spacing"])

            ego_spacing = random.uniform(start,end)

        self.controlled_vehicles = []
            
        vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=ego_spacing
            )

        vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)

        ego_x_position = vehicle.position[0]

        self.controlled_vehicles.append(vehicle)
        self.road.vehicles.append(vehicle)
        
        if (int(self.config["controlled_vehicles"]) == 2):

        
            vehicle = Vehicle.create_random(
                    self.road,
                    speed=25,
                    lane_id=self.config["initial_lane_id"],
                    spacing=ego_spacing
                )
            
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            vehicle.color = (255,20,147) #rgb color for PINK
            self.controlled_vehicles.append(vehicle)
           
            self.road.vehicles.append(vehicle)
            
            adv_x_position = vehicle.position[0]

            assert adv_x_position > ego_x_position #adv should in front of ego

        for others in other_per_controlled:
        
            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)
               
        #Configuration from HighwayEnvFast
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False
        
    
    def distance_reward(self, ego_obs, adv_obs, a =1, b =1):
        
        adv_vehicle = adv_obs[0]
        adv_x = adv_vehicle[1]
        adv_y = adv_vehicle[2]
        adv_vx = adv_vehicle[3]
        adv_vy = adv_vehicle[4]
    
        
        ego_vehicle = ego_obs[0]
        ego_x = ego_vehicle[1]
        ego_y = ego_vehicle[2]
        ego_vx = ego_vehicle[3]
        ego_vy = ego_vehicle[4]

        r_x = adv_x - ego_x
        r_y = adv_y - ego_y
        r_vx = adv_vx - ego_vx
        r_vy = adv_vy - ego_vy
        
        reward = self.conditional_reward(r_x, r_y, r_vx, r_vy)
         
        return reward

    def conditional_reward(self, r_x, r_y, r_vx, r_vy, alpha = 1, beta = 1):
        
        b=0.01
        c=3
        if r_vx >= 0:
            r = -r_vx - b 
        elif r_vy != 0:
            r = -abs(r_vy)/c 
        else:
            distance = math.sqrt(r_x**2 + r_y**2)
            r = 1/(1 + distance)

        return r
    
    def distance_calculation(self, ego_obs, adv_obs):
        adv_vehicle = adv_obs[0]
        adv_x = adv_vehicle[1]
        adv_y = adv_vehicle[2]
        adv_vx = adv_vehicle[3]
        adv_vy = adv_vehicle[4]
    
        
        ego_vehicle = ego_obs[0]
        ego_x = ego_vehicle[1]
        ego_y = ego_vehicle[2]
        ego_vx = ego_vehicle[3]
        ego_vy = ego_vehicle[4]

        r_x = adv_x - ego_x
        r_y = adv_y - ego_y
        r_vx = adv_vx - ego_vx
        r_vy = adv_vy - ego_vy

        distance = math.sqrt(r_x**2 + r_y**2)

        return distance

