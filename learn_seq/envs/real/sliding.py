import time
import rospy
import gym
import numpy as np
import transforms3d.quaternions as Q
from learn_seq.ros.ros_interface import (FRANKA_ERROR_MODE, KP_DEFAULT,
                                         FrankaRosInterface)

class RealSlidingEnv(gym.Env):
    """Base environment for panda sliding task. The goal is to slide the end-effector
    along pre-defined trajectory while maintaining constant normal contact force
    towards a surface.

    :param np.array(3) hole_pos: esimated hole position.
    :param np.array(4) hole_quat: estimated hole orientation (in quaternion).
    :param float hole_depth: insertion depth.
    :param list peg_pos_range: list of lower limit and upper limit of the
                               peg position (m)

    The goal pose is calculated from `hole_pos`, `hole_quat`, and `hole_depth`.

    The `peg_pos_range` amd `peg_rot_range` is to limit the peg position inside
    a box around the goal pose, and to normalize the observation to [-1, 1].

    When `reset()` is called, a random pose is chosen uniformly based on
    `initial_pos_range` and `initial_rot_range`, and the peg is moved to this
    pose.

    All position, orientation (except `hole_pos`, `hole_quat`) are defined
    relative to the hole frame.

    """
    def __init__(self,
                 fd_range=[3, 5],
                 speed_range=[0.03, 0.05]):

        # ros_interface to communicate with the ROS Controller
        self.ros_interface = FrankaRosInterface()

        # initial pos
        self.q0 = np.array([0, -np.pi / 4, 0, -3 * np.pi / 4,
                            0, np.pi / 2, np.pi / 4, 0.015, 0.015])

        # init pos
        self.init_pos = np.array([0.558861, 0.127538, 0.23938])
        self.init_quat = np.array([0., 1, 0, 0])

        # move to target
        self.tf_pos = np.array([0, 0, 0])
        self.tf_quat = np.array([1., 0, 0, 0])

        # sliding distance
        self.d_slide = 0.08

        # fd range (varied every time reset)
        self.fd_range = fd_range
        self.fd = fd_range[0]

        # sliding speed
        self.speed_range = speed_range

        # init gain
        self.kp_init = KP_DEFAULT
        self.kd_init = 2 * np.sqrt(self.kp_init)
        self.kp = self.kp_init.copy()
        self.kd = self.kd_init.copy()

        # gain limit
        range = [0.25, 2]
        self.gain_low = self.kp_init * range[0]
        self.gain_up = self.kp_init * range[1]

        # gain derivative limit (at each time step)
        self.dgain_low = np.array([-40]*3 + [-4]*3)
        self.dgain_up = np.array([40]*3 + [4]*3)

        self._set_observation_space()
        self._set_action_space()

    def _set_observation_space(self):
        """Set observation space to be the 6D pose (rotation vector representation).
        The vector is normalized to lie in the range [-1, 1]
        """
        # observation space: velocity, force, desired force
        vel_low = np.array([-0.5]*3 + [-1.]*3)
        vel_up = np.array([0.5]*3 + [1.]*3)
        force_low = np.array([-np.inf]*6)
        force_up = np.array([np.inf]*6)

        # normalized gain
        gain_low = np.array([-1]*6)
        gain_up = np.array([1]*6)

        self.obs_low_limit = np.hstack([vel_low, force_low, force_low, gain_low])
        self.obs_up_limit = np.hstack([vel_up, force_up, force_up, gain_up])
        self.observation_space = gym.spaces.Box(self.obs_low_limit, self.obs_up_limit)

    def reset(self):
        # if error is detected, clear it first
        if self.ros_interface.get_robot_mode() == FRANKA_ERROR_MODE:
            self.ros_interface.move_up(timeout=0.5)
            self.ros_interface.error_recovery()

        # # reset kp
        self.kp = self.kp_init
        self.kd = self.kd_init
        self.ros_interface.set_gain(self.kp_init, self.kd_init)
        time.sleep(0.5)
        import pdb
        pdb.set_trace()
        self.ros_interface.move_up()
        self.ros_interface.move_up()

        # move to init pose
        self.ros_interface.move_to_pose(self.init_pos, self.init_quat, 0.2, self.tf_pos, self.tf_quat, 10.)
        # reset controller
        self.ros_interface.reset_controller()

        cmd = self.ros_interface.get_constant_velocity_cmd()
        cmd.constant_velocity_param.speed_factor = 0.01
        cmd.constant_velocity_param.direction = np.array([0, 0, -1., 0, 0, 0])

        # move for 1 sec
        cmd.constant_velocity_param.timeout = 10.
        cmd.constant_velocity_param.f_thresh = 2.
        self.ros_interface.run_primitive(cmd)

        # set random fd
        # self.fd = self.fd_range[0]
        self.fd = np.random.uniform(low=self.fd_range[0], high=self.fd_range[1])
        self.s = np.random.uniform(low = self.speed_range[0], high=self.speed_range[1])

        # configure sliding motion
        u = np.array([0, -1, 0, 0, 0, 0])
        ft = np.array([0, 0, -self.fd, 0, 0, 0])
        self.timeout = self.d_slide / (self.s)
        self.p0_, self.q0_ = self.ros_interface.get_ee_pose()
        # print(self.p0_)
        time.sleep(1)
        self.start = rospy.Time.now().to_sec()

        return self._get_obs()

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
        v = self.ros_interface.get_ee_velocity()
        f = self.ros_interface.get_ee_force()
        fd = np.array([0, 0, -self.fd, 0, 0, 0])
        kp_norm = 2*(self.kp - self.gain_low) / (self.gain_up - self.gain_low) -1
        return np.hstack([v, f, fd, kp_norm])

    def _action_to_gain_map(self, a):
        return (a + 1)*(self.dgain_up - self.dgain_low)/2 + self.dgain_low

    def _set_action_space(self):
        # action space is normalized gain displacement delta_kp
        dgain_low = np.array([-1] * 6)
        dgain_up = np.array([1] * 6)

        self.action_space = gym.spaces.Box(dgain_low, dgain_up)

    def _reward_func(self, f):
        ef = -self.fd - f[2]
        rf = 1./8 - 2/(1 + np.exp(-np.abs(ef / 2) + 4))
        return rf

    def step(self, action):
        dkp = self._action_to_gain_map(action)
        # calculate new gain
        self.kp += dkp
        for i, kpi in enumerate(self.kp):
            if kpi > self.gain_up[i]:
                self.kp[i] = self.gain_up[i]
            elif kpi < self.gain_low[i]:
                self.kp[i] = self.gain_low[i]

        self.kd = 2 * np.sqrt(self.kp)

        # Update Gains
        self.ros_interface.set_gain(self.kp, self.kd)
        u = np.array([0, -self.s, 0, 0, 0, 0])
        ft = np.array([0, 0, -self.fd, 0, 0, 0])
        for i in range(40):
            cur_pos = u * (rospy.Time.now().to_sec() - self.start)
            cur_pos = cur_pos[:3] + self.p0_
            # print(cur_pos)
            self.ros_interface.set_cmd(ft, cur_pos, self.q0_, u)
            rospy.sleep(0.001)
            # Get next observation and reward received due to last action
        obs = self._get_obs()
        rew = self._reward_func(obs[6:12])
        rew += -0.01 * np.linalg.norm(action)

        # Terminates after 120s?
        done = rospy.Time.now().to_sec() - self.start > self.timeout or self._is_robot_error()

        return obs, rew, done, {}

    def _is_robot_error(self):
        return self.ros_interface.get_robot_mode() == FRANKA_ERROR_MODE
