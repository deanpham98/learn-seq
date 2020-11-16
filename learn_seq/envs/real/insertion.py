# TODO: move _reward_func, _is_success, _is_limit_reach to insertion_base to
# avoid duplication
import time

import numpy as np

import gym
from learn_seq.envs.insertion_base import InsertionBaseEnv
from learn_seq.primitive.real_container import RealPrimitiveContainer
from learn_seq.ros.ros_interface import (FRANKA_ERROR_MODE, KP_DEFAULT,
                                         FrankaRosInterface)
from learn_seq.utils.mujoco import quat2vec


class RealInsertionEnv(InsertionBaseEnv):
    """Franka insertion environment on real robot.

    :param np.array(3) hole_pos: see InsertionBaseEnv
    :param np.array(4) hole_quat: see InsertionBaseEnv
    :param float hole_depth: see InsertionBaseEnv
    :param type peg_pos_range: see InsertionBaseEnv
    :param type peg_rot_range: see InsertionBaseEnv
    :param type initial_pos_range: see InsertionBaseEnv
    :param type initial_rot_range: see InsertionBaseEnv
    :param float goal_thresh: the distance threshold to terminate an episode
        and detect success.
    :param list primitive_list: a list contains all primitives used. Each item
        in the list is a tuple (PrimitiveClass, param)

    The observation space is the 6-D relative pose between the peg and hole. The
    action space is a discrete space whose element is a primitive in the
    `primitive_list`.

    """
    def __init__(self,
                 hole_pos,
                 hole_quat,
                 hole_depth,
                 primitive_list,
                 peg_pos_range,
                 peg_rot_range,
                 initial_pos_range,
                 initial_rot_range,
                 goal_thresh=1e-3,
                 **controller_kwargs):
        # ros interface used to communicate with the ros controller
        self.ros_interface = FrankaRosInterface()
        # container used to execute MPs
        self.container = RealPrimitiveContainer(self.ros_interface, hole_pos,
                                                hole_quat)
        self.primitive_list = primitive_list
        super().__init__(hole_pos=hole_pos,
                         hole_quat=hole_quat,
                         hole_depth=hole_depth,
                         peg_pos_range=peg_pos_range,
                         peg_rot_range=peg_rot_range,
                         initial_pos_range=initial_pos_range,
                         initial_rot_range=initial_rot_range)
        self._eps_time = 0  # execution time
        self.goal_thresh = goal_thresh

        # initial gain
        self.kp_init = controller_kwargs.get("kp_init", KP_DEFAULT)
        self.kd_init = 2 * np.sqrt(self.kp_init)

    def step(self, action):
        t_exec = 0
        type, param = self.primitive_list[action]
        # execute the MP
        status, t_exec = self.container.run(type, param)
        info = {}
        self._eps_time += t_exec

        obs = self._get_obs()
        reward = self._reward_func(obs, t_exec)
        isLimitReach = self._is_limit_reach(obs[:3])
        isSuccess = self._is_success(obs[:3])
        isTimeout = self._eps_time > 20.
        isRobotError = self._is_robot_error()
        done = isTimeout or isLimitReach or isSuccess or isRobotError

        info.update({
            "success": isSuccess,
            "insert_depth": obs[2],
            "eps_time": self._eps_time,
            "mp_time": t_exec,
            "mp_status": status
        })

        return self._normalize_obs(obs), reward, done, info

    def _get_obs(self):
        p, q = self.ros_interface.get_ee_pose(self.tf_pos, self.tf_quat)
        # quat to angle axis
        r = quat2vec(q, ref=self.r_prev)
        f = self.ros_interface.get_ee_force()
        return np.hstack((p, r, f))

    def _set_action_space(self):
        no_primitives = len(self.primitive_list)
        self.action_space = gym.spaces.Discrete(no_primitives)

    def reset_to(self, p, q):
        self._eps_time = 0
        self.r_prev = quat2vec(self.target_quat)

        # if error is detected, clear it first
        if self.ros_interface.get_robot_mode() == FRANKA_ERROR_MODE:
            self.ros_interface.move_up(timeout=0.5)
            self.ros_interface.error_recovery()
        # set init gain
        self.ros_interface.set_gain(self.kp_init, self.kd_init)
        time.sleep(0.5)

        # move up if inside hole
        pc = self.ros_interface.get_ee_pos()
        if pc[2] < self.tf_pos[2]:
            self.ros_interface.move_up(timeout=2.)

        # calibrate force
        p0 = self.target_pos.copy()
        p0[2] = 0.01
        q0 = self.target_quat.copy()
        self.ros_interface.move_to_pose(p0, q0, 0.3, self.tf_pos, self.tf_quat,
                                        10)
        pa, qa = self.ros_interface.get_ee_pose(self.tf_pos, self.tf_quat)
        time.sleep(0.5)
        self.ros_interface.set_init_force()
        # move to initial pose
        self.ros_interface.move_to_pose(p, q, 0.1, self.tf_pos, self.tf_quat,
                                        10)
        pa, qa = self.ros_interface.get_ee_pose(self.tf_pos, self.tf_quat)
        obs = self._get_obs()
        return self._normalize_obs(obs)

    def _reward_func(self, obs, t_exec):
        pos = obs[:3]
        dist = obs[:3] - self.target_pos
        d = np.linalg.norm(dist)
        wi = 2
        rwd_goal = wi * np.exp(-d**2 / 0.02**2) - wi

        rwd_short_length = -t_exec / 4

        rwd_terminal = 0
        if self._is_success(pos):
            rwd_terminal = 5.

        r = rwd_goal + rwd_short_length + rwd_terminal
        return r

    def _is_limit_reach(self, p):
        return np.max(
            np.abs(p[:2] - self.target_pos[:2])) > self.obs_up_limit[0]

    def _is_success(self, p):
        return np.linalg.norm(p[:3] - self.target_pos[:3]) < self.goal_thresh

    def _is_robot_error(self):
        robot_mode = self.ros_interface.get_robot_mode()
        return robot_mode == FRANKA_ERROR_MODE

    def _normalize_obs(self, obs):
        # normalize
        obs[:6] = 2*(obs[:6] - self.obs_low_limit) \
                  / (self.obs_up_limit - self.obs_low_limit) - 1
        return obs

    def set_task_frame(self, tf_pos, tf_quat):
        super().set_task_frame(tf_pos, tf_quat)
        self.container.set_task_frame(tf_pos, tf_quat)
