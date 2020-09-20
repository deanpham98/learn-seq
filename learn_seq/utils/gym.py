import gym
import os
import numpy as np
import mujoco_py
from gym.envs.mujoco import mujoco_env
from gym.spaces import Space, Discrete

class MujocoEnvBase(mujoco_env.MujocoEnv):
    def __init__(self, model_path, frame_skip):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self._set_action_space()

        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done

        self._set_observation_space(observation)

        self.seed()

class DynamicDiscrete(Space):
    """
    Implement a discrete action spaces that generate sample
    based on a condition on the state.

    It is assumed that there is a finite number of subspaces A_i(s), and that
    n = |union(A_i(s))|

    The user provide n and the list of indices of the actions in a subspace A_i(s)
    E.g: A_1(s) = (1, 2, 3, 4), A_2(s) = (3, 5, 6, 7)
    then n = 7, sub_indices = [[1,2,3,4], [3,5,6,7]]
    """
    def __init__(self, n, sub_indices):
        assert n >= 0
        self.n = n
        self.sub_spaces = []
        self.sub_actions = sub_indices
        for s in sub_indices:
            sub_n = len(s)
            self.sub_spaces.append(Discrete(sub_n))
        super(DynamicDiscrete, self).__init__((), np.int64)

    def sample(self):
        return self.np_random.randint(self.n)

    def sample_subspace(self, idx):
        action_idx = self.np_random.randint(self.sub_spaces[idx].n)
        return self.sub_actions[idx][action_idx]

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (x.dtype.char in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return False
        return as_int >= 0 and as_int < self.n

    def __repr__(self):
        return "Discrete(%d)" % self.n

    def __eq__(self, other):
        return isinstance(other, Discrete) and self.n == other.n
