from JSSEnv import envs

from gym.envs.registration import register

register(
    id="JSSEnv-v1",
    entry_point="JSSEnv.envs:JssEnv",
    kwargs={'env_config': None}
)
