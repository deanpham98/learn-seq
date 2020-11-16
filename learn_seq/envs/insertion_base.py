import numpy as np

import gym
from learn_seq.utils.mujoco import integrate_quat, quat2vec


class InsertionBaseEnv(gym.Env):
    """Base environment for panda insertion task. The goal is to move a peg to
    a goal pose. The goal pose is calculated based on the hole frame composed
    of hole position and hole orientation (this hole frame is usually attached
    to one feature of the hole, e.g. the center of the circle for the round
    hole).

    :param np.array(3) hole_pos: esimated hole position.
    :param np.array(4) hole_quat: estimated hole orientation (in quaternion).
    :param float hole_depth: insertion depth.
    :param list peg_pos_range: list of lower limit and upper limit of the
                               peg position (m)
    :param list peg_rot_range: list of lower limit and upper limit of the
                               peg orientation (rad)
    :param list initial_pos_range: list of lower limit and upper limit of the
                                    initial peg position (m)
    :param list initial_rot_range: list of lower limit and upper limit of the
                                    initial peg orientation (rad)

    The goal pose is calculated from `hole_pos`, `hole_quat`, and `hole_depth`.

    The `peg_pos_range` amd `peg_rot_range` is to limit the peg position inside
    a box around the goal pose, and to normalize the observation to [-1, 1].

    When `reset()` is called, a random pose is chosen uniformly based on
    `initial_pos_range` and `initial_rot_range`, and the peg is moved to this
    pose.

    All position, orientation (except `hole_pos`, `hole_quat`) are defined
    relative to the hole frame.

    """
    def __init__(self, hole_pos, hole_quat, hole_depth, peg_pos_range,
                 peg_rot_range, initial_pos_range, initial_rot_range):
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
        self.traj_info = {
            "p0": np.zeros(3),  # init pos
            "r0": np.zeros(3)  # init quat
        }

    def set_task_frame(self, hole_pos, hole_quat):
        self.tf_pos = hole_pos
        self.tf_quat = hole_quat

    def _set_observation_space(self):
        """Set observation space to be the 6D pose (rotation vector representation).
        The vector is normalized to lie in the range [-1, 1]
        """
        peg_pos_low = self.target_pos + np.array(self.peg_pos_range[0])
        peg_pos_up = self.target_pos + np.array(self.peg_pos_range[1])
        r0 = quat2vec(self.target_quat)
        peg_rot_low = r0 + np.array(self.peg_rot_range[0])
        peg_rot_up = r0 + np.array(self.peg_rot_range[1])
        self.obs_low_limit = np.hstack((peg_pos_low, peg_rot_low))
        self.obs_up_limit = np.hstack((peg_pos_up, peg_rot_up))
        self.observation_space = gym.spaces.Box(self.obs_low_limit,
                                                self.obs_up_limit)

    def reset(self):
        p0, q0 = self._sample_init_pose()
        self.traj_info["p0"] = p0 - self.initial_pos_mean
        self.traj_info["r0"] = quat2vec(
            q0, self.initial_rot_mean) - self.initial_rot_mean
        obs = self.reset_to(p0, q0)
        return obs

    def _sample_init_pose(self):
        p0 = self.initial_pos_mean + np.random.uniform(
            self.initial_pos_range[0], self.initial_pos_range[1])
        r = self.initial_rot_mean + np.random.uniform(
            self.initial_rot_range[0], self.initial_rot_range[1])
        q0 = integrate_quat(self.target_quat, r, 1)
        return p0, q0

    def get_task_frame(self):
        return self.tf_pos.copy(), self.tf_quat.copy()

    def _get_obs(self):
        """Return the observation.
        """
        raise NotImplementedError

    def _set_action_space(self):
        raise NotImplementedError

    def reset_to(self, p, q):
        raise NotImplementedError
