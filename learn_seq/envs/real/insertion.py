# TODO: move _reward_func, _is_success, _is_limit_reach to insertion_base to
# avoid duplication
import time
import numpy as np
import gym
from learn_seq.envs.insertion_base import InsertionBaseEnv
from learn_seq.ros.ros_interface import FrankaRosInterface, FRANKA_ERROR_MODE, KP_DEFAULT
from learn_seq.primitive.real_container import RealPrimitiveContainer
from learn_seq.utils.mujoco import quat2vec

class RealInsertionEnv(InsertionBaseEnv):
    def __init__(self,
                 hole_pos,
                 hole_quat,
                 hole_depth,
                 primitive_list,
                 peg_pos_range,
                 peg_rot_range,
                 initial_pos_range,
                 initial_rot_range,
                 depth_thresh=0.95,
                 **controller_kwargs):
        self.ros_interface = FrankaRosInterface()
        self.container = RealPrimitiveContainer(self.ros_interface, hole_pos, hole_quat)
        self.primitive_list = primitive_list
        super().__init__(hole_pos=hole_pos,
                         hole_quat=hole_quat,
                         hole_depth=hole_depth,
                         peg_pos_range=peg_pos_range,
                         peg_rot_range=peg_rot_range,
                         initial_pos_range=initial_pos_range,
                         initial_rot_range=initial_rot_range)
        self._eps_time = 0
        self.depth_thresh = depth_thresh

        # kp_init
        self.kp_init = controller_kwargs.get("kp_init", KP_DEFAULT)
        self.kd_init = 2*np.sqrt(self.kp_init)

    def step(self, action):
        t_exec = 0
        type, param = self.primitive_list[action]
        status, t_exec = self.container.run(type, param)
        info = {}
        if self._eps_time == 0:
            p, q = self.ros_interface.get_ee_pose(self.tf_pos, self.tf_quat)
            info["init_pos"] = p
            info["init_quat"] = q
        self._eps_time += t_exec

        obs = self._get_obs()
        reward = self._reward_func(obs, t_exec)
        isLimitReach = self._is_limit_reach(obs[:3])
        isSuccess = self._is_success(obs[:3])
        isTimeout = self._eps_time > 20.
        isRobotError = self._is_robot_error()
        done = isTimeout or isLimitReach or isSuccess or isRobotError

        info.update({"success": isSuccess,
                "insert_depth": obs[2],
                "eps_time": self._eps_time})

        return self._normalize_obs(obs), reward, done, info

    def _get_obs(self):
        p, q= self.ros_interface.get_ee_pose(self.tf_pos, self.tf_quat)
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

        if self.ros_interface.get_robot_mode() ==FRANKA_ERROR_MODE:
            self.ros_interface.move_up(timeout=0.5)
            self.ros_interface.error_recovery()
        # set init gain
        # self.ros_interface.set_gain(self.kp_init, self.kd_init)
        # time.sleep(0.5)

        # move up if inside hole
        pc = self.ros_interface.get_ee_pos()
        if pc[2] < self.tf_pos[2]:
            self.ros_interface.move_up(timeout=2.)

        # calibrate force
        p0 = self.target_pos.copy()
        p0[2] = 0.01
        q0 = self.target_quat.copy()
        self.ros_interface.move_to_pose(p0, q0, 0.3, self.tf_pos, self.tf_quat, 10)
        pa, qa = self.ros_interface.get_ee_pose(self.tf_pos, self.tf_quat)
        time.sleep(0.5)
        self.ros_interface.set_init_force()
        # move to reset position
        self.ros_interface.move_to_pose(p, q, 0.1, self.tf_pos, self.tf_quat, 10)
        pa, qa = self.ros_interface.get_ee_pose(self.tf_pos, self.tf_quat)
        obs =  self._get_obs()
        return self._normalize_obs(obs)

    def _reward_func(self, obs, t_exec):
        pos = obs[:3]
        # assume the z axis align with insert direction
        dist = obs[:3] - self.target_pos
        rwd_near = np.min(np.exp(-dist[:2]**2 / 0.01**2)) - 1

        # rwd for insertion
        wi = 2
        rwd_insert = wi * np.exp(-dist[2]**2/0.05**2) - wi

        # limit the peg inside a box around the goal
        rwd_inside = 0.
        if self._is_limit_reach(pos):
            rwd_inside = -50.

        # rwd_short_length = -3.
        rwd_short_length = -t_exec/4

        # error (large force) is bad
        rwd_error = 0
        if self._is_robot_error():
            rwd_error = -20

        rwd_terminal = 0
        if self._is_success(pos):
            rwd_terminal = 5.

        r = rwd_near +rwd_inside +rwd_insert +rwd_short_length +rwd_terminal + rwd_error
        return r

    def _is_limit_reach(self, p):
        return np.max(np.abs(p[:2] - self.target_pos[:2])) > self.obs_up_limit[0]

    def _is_success(self, p):
        pos_thresh = 0.01
        isDepthReach = p[2] < self.target_pos[2]*self.depth_thresh
        isInHole = np.linalg.norm(p[:2] - self.target_pos[:2]) < pos_thresh
        return (isDepthReach and isInHole)

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
