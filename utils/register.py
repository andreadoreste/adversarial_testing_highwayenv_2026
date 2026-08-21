import highway_env
from gymnasium.envs.registration import register

register(
    id='highwayMA-v1',
    entry_point = 'envs:HighwayEnvMA',
)

register(
    id='mergeMA-v1',
    entry_point = 'envs:MergeEnvMA',
)

register(
    id='roundaboutMA-v1',
    entry_point = 'envs:RoundaboutEnvMA',
)