from typing import Dict, Text

import numpy as np
import random
import math

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle
from highway_env.vehicle.kinematics import Vehicle


class MergeEnvMA(AbstractEnv):

    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "collision_reward": -1,
            "right_lane_reward": 0.1,
            "high_speed_reward": 0.2,
            "merging_speed_reward": -0.5,
            "lane_change_reward": -0.05,
            "controlled_vehicles": 1,
            "reward_vehicle": "ego",
            "overtaken" : False,
            "delta" : 0,
            "train_phase": "merge",
            "adv_factor": 0.1
        })
        return cfg

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.

        :param action: the action performed
        :return: the reward of the state-action transition
        """
        
        rewards_2 = tuple(self._agent_rewards(action, vehicle) for vehicle in self.controlled_vehicles)

        r = []
        for rewards in rewards_2:
            reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
            reward = utils.lmap(reward,
                          [self.config["collision_reward"] + self.config["merging_speed_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1])
            r.append(reward)

        return r

    def _agent_rewards(self, action: int, vehicle: Vehicle) -> Dict[Text, float]:
        return {
            "collision_reward": vehicle.crashed,
            "right_lane_reward": vehicle.lane_index[2] / 1,
            "high_speed_reward": vehicle.speed_index / (vehicle.target_speeds.size - 1),
            "lane_change_reward": action in [0, 2],
            "merging_speed_reward": sum(  # Altruistic penalty
                (vehicle.target_speed - vehicle.speed) / vehicle.target_speed
                for vehicle in self.road.vehicles
                if vehicle.lane_index == ("b", "c", 2) and isinstance(vehicle, ControlledVehicle)
            )
        }

    def _is_terminated(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""

        return (
            any(vehicle.crashed for vehicle in self.controlled_vehicles) or
            any(bool(vehicle.position[0] > 400) for vehicle in self.controlled_vehicles)
        )
    def _is_truncated(self) -> bool:
        return False

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        ends = [150, 80, 80, 150]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]
        for i in range(2):
            net.add_lane("a", "b", StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]))
            net.add_lane("b", "c", StraightLane([sum(ends[:2]), y[i]], [sum(ends[:3]), y[i]], line_types=line_type_merge[i]))
            net.add_lane("c", "d", StraightLane([sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]))

        # Merging lane
        amplitude = 3.25
        ljk = StraightLane([0, 6.5 + 4 + 4], [ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True)
        lkb = SineLane(ljk.position(ends[0], -amplitude), ljk.position(sum(ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2*ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        lbc = StraightLane(lkb.position(ends[1], 0), lkb.position(ends[1], 0) + [ends[2], 0],
                           line_types=[n, c], forbidden=True)
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])

        self.obst = Obstacle(road, lbc.position(ends[2], 0))
        road.objects.append(self.obst)
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        road = self.road

        merge_lane, merge_speed = road.network.get_lane(("j", "k", 0)).position(110, 0), 20
        regular_lane, regular_speed = road.network.get_lane(("a","b",0)).position(90,0), 29

        pos, sp = 90, 29
        reg_lane = road.network.get_lane(("a", "b", self.np_random.integers(2)))
        reg_position = reg_lane.position(pos + self.np_random.uniform(-5, 5), 0)
        regular_speed += self.np_random.uniform(-1, 1)

        if self.config["train_phase"] == "merge":
            ego_lane, ego_speed = merge_lane, merge_speed
            challenger_lane, challenger_speed = reg_position, regular_speed       
        else:
            print('not_merge')
            challenger_lane, challenger_speed = merge_lane, merge_speed
            ego_lane, ego_speed = reg_position, regular_speed   
        
        ego_vehicle = self.action_type.vehicle_class(road, ego_lane, speed = ego_speed) 

        road.vehicles.append(ego_vehicle)

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        for position, speed in [(70, 31), (5, 31.5)]:
            lane = road.network.get_lane(("a", "b", self.np_random.integers(2)))
            position = lane.position(position + self.np_random.uniform(-5, 5), 0)
            speed += self.np_random.uniform(-1, 1)
            road.vehicles.append(other_vehicles_type(road, position, speed=speed))


        if self.config["controlled_vehicles"] == 2:

            adv_vehicle = self.action_type.vehicle_class(road, challenger_lane, speed=challenger_speed)
            adv_vehicle.color = (255,20,147) #rgb color for PINK
            road.vehicles.append(adv_vehicle)
        
            self.controlled_vehicles = [ego_vehicle, adv_vehicle]

        else:
            road.vehicles.append(other_vehicles_type(road, challenger_lane, speed= challenger_speed)) #to train the vehicle that will be the ego

            self.controlled_vehicles = [ego_vehicle]

        
    def _info(self, obs:np.ndarray, action: int) -> dict:

        info = super()._info(obs, action)

        info["agents_rewards"] = tuple(
            self._agent_rewards(action, vehicle) for vehicle in self.controlled_vehicles
        )
        
        crashed_list = []
        
        for vehicle in self.controlled_vehicles:
            crashed_list.append(vehicle.crashed)

        crashed = crashed_list[0]

        self.overtaken = False
        if len(obs) == 2:

            ego_x_pos = obs[0][0]
            adv_x_pos = obs[1][0]
                   

            dist_between_agents = adv_x_pos[1] - ego_x_pos[1]

            delta = self.config["delta"]

            if dist_between_agents <= delta:
                #self.overtaken = True
                adv_in_front = True
            info["overtaken"] = self.overtaken
            r_diff, dist_ego_adv = self.distance_reward(obs[0],obs[1])
            info["r_diff"] = r_diff
            info["dist_ego_adv"] = dist_ego_adv
            adv_crashed = crashed_list[1]

            if crashed_list[0] and (crashed_list[1] or self.obst.crashed):
                crashed = True
            else:
                crashed = False

        info["crashed"] = crashed
        info["crashed_list"] = crashed_list

        return info
    
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        info["agents_rewards"] = reward
        #returning reward for the ego vehicle (the adv reward was added to the info dict)
        returned_reward = reward[0]
        
        if self.config["reward_vehicle"] is "adversarial":
            returned_reward = reward[1]

        return obs, returned_reward, terminated, truncated, info
    

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
        
        
        distance = math.sqrt(math.pow(r_x,2) + math.pow(r_y,2)) 

        reward = 1/(1+distance)
        
        return reward, distance
