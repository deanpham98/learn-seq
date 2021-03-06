import os

import mujoco_py

import gym
from learn_seq.utils.mujoco import set_state


class MujocoEnv(gym.Env):
    """Base environment for mujoco environment, mainly to load the model,
    render and set the current simulation state.

    :param string model_path: path of the xml mujoco model.

    """
    def __init__(self, model_path):
        if model_path.startswith('/'):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(
                __file__), 'assets', model_path)
        if not os.path.exists(fullpath):
            raise IOError('File {} does not exist'.format(fullpath))

        model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(model)
        self.data = self.sim.data
        self.model = self.sim.model
        self.viewer = None

    def render(self):
        self._get_viewer().render()

    def close(self):
        self.viewer = None

    def _get_viewer(self):
        """Initialize the viewer if can't find one"""
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer_setup()
        return self.viewer

    def set_state(self, qpos, qvel):
        """Enforce a joint configuration and velocity"""
        set_state(self.sim, qpos, qvel)

    @property
    def dt(self):
        return self.model.opt.timestep

    def viewer_setup(self):
        """Use to change the camera view"""
        raise NotImplementedError
