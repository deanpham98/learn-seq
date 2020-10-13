import numpy as np
import gym
from learn_seq.utils.mujoco import integrate_quat

class InsertionBaseEnv(gym.Env):
    """Base environment for panda insertion task.

    :param type hole_pos: esimated hole position.
    :param type hole_quat: estimated hole orientation (in quaternion).
    :param type hole_depth: insertion depth.
    :param type peg_pos_range: range of peg position
    :param type peg_rot_range: range of peg orientation (in angle).Define relative
                               to the "goal" pose. Use to check limit the peg pose
                               not go to far from the hole
    :param type initial_pos_range: range of initial peg position.
    :param type initial_rot_range: range of initial peg orientation.

    """
    def __init__(self,
                 hole_pos,
                 hole_quat,
                 hole_depth,
                 peg_pos_range,
                 peg_rot_range,
                 initial_pos_range,
                 initial_rot_range):
        # set task frame to the estimated hole position
        self.set_task_frame(hole_pos, hole_quat)

        # target pose relative to the task frame
        self.target_pos = np.array([0., 0, -hole_depth])
        self.target_quat = np.array([0, 1., 0, 0])

        # initial pos and quat (relative to the task frame)
        self.initial_pos_mean = np.array([0, 0, 0.01])
        self.initial_pos_range = initial_pos_range
        self.initial_rot_mean = np.zeros(3)
        self.initial_rot_range = initial_rot_range

        # safety limit for the peg position
        self.peg_pos_range = peg_pos_range
        self.peg_rot_range = peg_rot_range

        self._set_observation_space()
        self._set_action_space()

        # episode info
        self.eps_info = {}

    def set_task_frame(self, hole_pos, hole_quat):
        self.tf_pos = hole_pos
        self.tf_quat = hole_quat

    def _set_observation_space(self):
        """Set observation space to be the 6D pose (rotation vector representation).
        The vector is normalized to lie in the range [-1, 1]

        :return: Description of returned object.
        :rtype: type
        """
        self.obs_low_limit = np.array(self.peg_pos_range[0] + self.peg_rot_range[0])
        self.obs_up_limit = np.array(self.peg_pos_range[1] + self.peg_rot_range[1])
        # TODO this is wrong, calculate based on self.target_pos and self.target quat
        self.observation_space = gym.spaces.Box(self.obs_low_limit, self.obs_up_limit)

    def reset(self):
        p0, q0 = self._sample_init_pose()
        obs = self.reset_to(p0, q0)
        return obs

    def _sample_init_pose(self):
        p0 = self.initial_pos_mean + np.random.uniform(self.initial_pos_range[0],\
                                        self.initial_pos_range[1])
        r = self.initial_rot_mean + np.random.uniform(self.initial_rot_range[0],\
                                        self.initial_rot_range[1])
        q0 = integrate_quat(self.target_quat, r, 1)
        self.eps_info["init_pos"] = p0
        self.eps_info["init_rot"] = r
        return p0, q0

    def step(self):
        raise NotImplementedError

    def _get_obs(self):
        """Return the observation.
        """
        raise NotImplementedError

    def _set_action_space(self):
        raise NotImplementedError

    def reset_to(self, p, q):
        raise NotImplementedError
