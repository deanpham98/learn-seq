import gym
from gym.envs.mujoco import mujoco_env

# TODO: create a wrapper to adjust mujoco env
class MujocoEnvBase(mujoco_env.MujocoEnv):
    def __init__(self):
        pass
