single_env = {"incoming_vehicle_destination": 2,
"observation": {
                    "type": "Kinematics",
                    "absolute": True,
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-15, 15],
                        "vy": [-15, 15],
                    },              
              "include_obstacles": False,
              "see_behind" : True,
              },
"action": {
                "type": "DiscreteMetaAction",
                "target_speeds": [2, 8, 16]
            },
"high_speed_reward": 1.0,
}

multi_env = {
            "controlled_vehicles":2,
             "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics",
                    "absolute": True,
                    "features_range": {"x": [-100, 100], "y": [-100, 100], "vx": [-15, 15], "vy": [-15, 15]},
                    "include_obstacles": False,
                    "see_behind" : True,
                },
            },
            "action": {
                    "type": "MultiAgentAction",
                    "action_config": {
                        "type": "DiscreteMetaAction",
                        "target_speeds": [2, 8, 16],
                    },
        },
            "collision_reward": 0,
            "reward_vehicle" : "adversarial",
            "high_speed_reward": 1.0,
            "screen_width": 1000,  # [px] width of the pygame window
        "screen_height": 1000,  # [px] height of the pygame window
        "show_trajectory": False,
            }


rt_env = {
            "controlled_vehicles":2,
             "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics",
                    "absolute": True,
                    "features_range": {"x": [-100, 100], "y": [-100, 100], "vx": [-15, 15], "vy": [-15, 15]},
                    "include_obstacles": False,
                    "see_behind" : True,
                },
            },
            "action": {
                    "type": "MultiAgentAction",
                    "action_config": {
                        "type": "DiscreteMetaAction",
                        "target_speeds": [2, 8, 16],
                    },
        },
            "collision_reward": -1,
            "high_speed_reward": 1.0,
            }
