single_env= {
  "controlled_vehicles" : 1,
  "observation": {
      "type": "Kinematics",
      "features": ["presence", "x", "y", "vx", "vy"],
      "features_range": {
            "x": [-1200, 1200],
      },
      "absolute": True,
      "normalize" : True,
      "see_behind": True,
  },
      "lanes_count" : 3,
      "vehicles_count": 5,
      "duration": 30,
      "normalize_reward": True,
      "collision_reward": -1
}

multi_env = {
    "controlled_vehicles" : 2,
    "action": {
        "type": "MultiAgentAction",
        "action_config": {
            "type": "DiscreteMetaAction",
        },
    },
  "observation": {
        "type": 'MultiAgentObservation',
        "observation_config": {
            "type": "Kinematics",
            "features": ["presence", "x", "y", "vx", "vy"],
            "features_range": {
            "x": [-1200, 1200],
            },
            "absolute": True,
            "normalize" : True,
            "see_behind": True,
        }  
  },
  "lanes_count" : 3,
  "vehicles_count": 4,
  "reward_vehicle" : 'adversarial',
  "duration": 30,
  "collision_reward": 0,
"normalize_reward": True,
"ego_spacing" : [0.5,2.0],
"simulation_frequency" : 15,
"delta" : 0.0025,
}

n_car_c11 = single_env["controlled_vehicles"] + single_env["vehicles_count"]
n_car_cma = multi_env["controlled_vehicles"] + multi_env["vehicles_count"]

assert n_car_c11 == n_car_cma

assert n_car_c11 == 1 + 5

retrain_env = {
    "controlled_vehicles" : 2,
    "action": {
        "type": "MultiAgentAction",
        "action_config": {
            "type": "DiscreteMetaAction",
        },
    },
  "observation": {
        "type": 'MultiAgentObservation',
        "observation_config": {
            "type": "Kinematics",
            "features": ["presence", "x", "y", "vx", "vy"],
            "features_range": {
            "x": [-1200, 1200],
            },
            "absolute": True,
            "normalize" : True,
            "see_behind": True,
        }  
  },
  "lanes_count" : 3,
  "vehicles_count": 4,
  "reward_vehicle" : 'ego',
  "duration": 30,
  "collision_reward": -1,
"normalize_reward": True,
"ego_spacing" : [0.5,2.0],
"simulation_frequency" : 15,
"delta" : 0.0025,
}

n_car_crt = retrain_env["controlled_vehicles"] + retrain_env["vehicles_count"]

assert n_car_crt == n_car_c11

assert n_car_crt == 1 + 5