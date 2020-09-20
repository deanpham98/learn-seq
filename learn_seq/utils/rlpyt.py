import os
import gym
import torch
from rlpyt.samplers.collections import TrajInfo
from rlpyt.envs.gym import GymEnvWrapper

class CustomTrajInfo(TrajInfo):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Success = 0.
        self.InsertDepth = 0.

    def step(self, observation, action, reward, done, agent_info, env_info):
        super().step(observation, action, reward, done, agent_info, env_info)
        if done:
            self.Success = getattr(env_info, "success")
            self.InsertDepth = getattr(env_info, "insert_depth")

def gym_make(*args, info_example=None, wrapper=None, wrapper_kwargs=None, **kwargs):
    """adapted from rlpyt.envs.gym.gym_make
    add the option to append an additional wrapper
    """
    env = gym.make(*args, **kwargs)
    if info_example is None:
        if wrapper is None:
            return GymEnvWrapper(env)
        else:
            return GymEnvWrapper(wrapper(env, **wrapper_kwargs))

    else:
        if wrapper is None:
            return GymEnvWrapper(EnvInfoWrapper(env, info_example))
        else:
            return GymEnvWrapper(
                        EnvInfoWrapper(
                            wrapper(env, **wrapper_kwargs),
                            info_example
                        )
                    )

def load_agent_state_dict(data_path):
    model_path = os.path.join(data_path, "params.pkl")
    if torch.cuda.is_available():
        model_data = torch.load(model_path)
    else:
        model_data = torch.load(model_path, map_location=torch.device("cpu"))
    agent_state_dict = model_data["agent_state_dict"]
    return agent_state_dict
