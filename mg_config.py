single_env = {"observation": {
      "type": "Kinematics",
      "features": ["presence", "x", "y", "vx", "vy"],
      "features_range": {
            "x": [-1200, 1200],
      },
      "absolute": True,
      "normalize" : True,
      "see_behind": True,
      
  },
    "train_phase": "merge",
  
  }

multi_env = {
            "controlled_vehicles": 2,
             "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics",
                    "see_behind": True,
                    "absolute": True,
                    "features": ["presence", "x", "y", "vx", "vy"],
                    "features_range": {
                        "x": [-1200, 1200],
                    },
                },
            },
            "action": {
                    "type": "MultiAgentAction",
                    "action_config": {
                        "type": "DiscreteMetaAction",
                    },
        },
            "collision_reward": 0,
            "reward_vehicle": "adversarial"
            }

rt_env = {
            "controlled_vehicles": 2,
             "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics",
                    "see_behind": True,
                    "absolute": True,
                    "features": ["presence", "x", "y", "vx", "vy"],
                    "features_range": {
                        "x": [-1200, 1200],
                    },
                },
            },
            "action": {
                    "type": "MultiAgentAction",
                    "action_config": {
                        "type": "DiscreteMetaAction",
                    },
        },
            "collision_reward": -1,
            "reward_vehicle": "ego"
            }
