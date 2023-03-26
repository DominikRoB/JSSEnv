import gym


class ObjectiveWrapper(gym.Wrapper):
    """ For an environment with a reward dictionary, this wrapper returns the specified reward from that dictionary and includes all rewards in the info dict"""

    def __init__(self, env, objective: str):
        super().__init__(env)
        self.objective = objective

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if type(reward) != dict:
            print("ObjectiveWrapper: Reward is not a Dict - returning original reward and info")
            return obs, reward, terminated, truncated, info

        assert self.objective in reward

        single_value_reward = reward[self.objective]
        info.update(reward)
        return obs, single_value_reward, terminated, truncated, info
