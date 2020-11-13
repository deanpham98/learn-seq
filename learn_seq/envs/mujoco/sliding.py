import os
import numpy as np
import gym
import mujoco_py
from .mujoco_env import MujocoEnv
from learn_seq.utils.general import get_mujoco_model_path
from learn_seq.controller.robot_state import RobotState
from learn_seq.controller.impedance import StateRecordImpedanceController
from learn_seq.utils.mujoco import get_geom_pose, get_geom_size, quat2mat,\
    set_state, quat2vec, get_body_pose, get_mesh_vertex_pos
from learn_seq.primitive.impedance import Move2Target, Move2Contact
from learn_seq.primitive.base import PrimitiveStatus

class MujocoFrankaSlidingEnv(MujocoEnv):
    def __init__(self,
                 no_env_steps=20, #0.04s
                 fd_range=[5, 10],
                 speed_range=[0.01, 0.02],
                 xml_model_name="sliding.xml"):

        # mujoco model
        mujoco_path = get_mujoco_model_path()
        model_path = os.path.join(mujoco_path, xml_model_name)
        MujocoEnv.__init__(self, model_path)
        # robot state
        self.robot_state = RobotState(self.sim, "end_effector")
        # initial pos
        self.q0 = np.array([0, -np.pi / 4, 0, -3 * np.pi / 4,
                            0, np.pi / 2, np.pi / 4, 0.015, 0.015])
        # controller
        self.controller = StateRecordImpedanceController(self.robot_state)
        self._reset_sim()

        # init pos
        self.init_pos = np.array([0.53, 0.062, 0.1888])
        self.init_quat = np.array([0., 1, 0, 0])

        # move to target
        tf_pos = np.array([0, 0, 0])
        tf_quat = np.array([1., 0, 0, 0])
        self.mp1 = Move2Target(
            self.robot_state, self.controller, tf_pos, tf_quat)
        self.mp2 = Move2Contact(
            self.robot_state, self.controller, tf_pos, tf_quat)
        # sliding
        self.mp_sliding = Move2Contact(
            self.robot_state, self.controller, tf_pos, tf_quat)
        # sliding distance
        self.d_slide = 0.1

        # fd range (varied every time reset)
        self.fd_range = fd_range
        self.fd = fd_range[0]

        # sliding speed
        self.speed_range = speed_range

        # init gain
        self.kp_init = np.array([1000., 1000, 1000, 60, 60, 60])
        self.kd_init = 2*np.sqrt(self.kp_init)
        self.kp = self.kp_init.copy()
        self.kd = self.kd_init.copy()
        # gain limit
        range = [0.25, 2]
        self.gain_low = self.kp_init * range[0]
        self.gain_up = self.kp_init * range[1]
        # gain derivative limit (at each time step)
        self.dgain_low = np.array([-40]*3 + [-4]*3)
        self.dgain_up = np.array([40]*3 + [4]*3)

        # observation and action space
        self._set_observation_space()
        self._set_action_space()

        # mujoco sim steps per 1 env step
        self.no_env_steps = no_env_steps

    def _set_observation_space(self):
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

    def _set_action_space(self):
        # action space is normalized gain displacement delta_kp
        dgain_low = np.array([-1]*6)
        dgain_up = np.array([1]*6)

        self.action_space = gym.spaces.Box(dgain_low, dgain_up)

    def _action_to_gain_map(self, a):
        return (a + 1)*(self.dgain_up - self.dgain_low)/2 + self.dgain_low

    def _reset_sim(self):
        set_state(self.sim, self.q0, np.zeros(self.model.nv))
        # clear control
        self.sim.data.ctrl[:] = 0
        # reset controller cmd
        self.controller.reset_pose_cmd()
        self.controller.reset_tau_cmd()
        # reset filter
        self.robot_state.reset_filter_state()

    def _get_obs(self):
        v = self.robot_state.get_ee_velocity()
        f = self.robot_state.get_ee_force()
        fd = np.array([0, 0, -self.fd, 0, 0, 0])
        kp_norm = 2*(self.kp - self.gain_low) / (self.gain_up - self.gain_low) -1
        return np.hstack([v, f, fd, kp_norm])

    # def _normalize_obs(self)

    def _reward_func(self):
        f = self.robot_state.get_ee_force()
        ef = -self.fd - f[2]
        # rf = 1 - 1/4 * ef**2
        rf = 1./8 - 2/(1 + np.exp(-np.abs(ef/2) + 4))
        # rew = -1e-2 * np.linalg.norm(a) + rf
        return rf

    def viewer_setup(self):
        self.viewer.cam.distance = 0.43258
        self.viewer.cam.lookat[:] = [0.517255, 0.0089188, 0.25619]
        self.viewer.cam.elevation = -20.9
        self.viewer.cam.azimuth = 132.954

    def step(self, action, render=False):
        if render:
            viewer = self._get_viewer()
        else:
            viewer = None

        dkp = self._action_to_gain_map(action)
        # calculate new gain
        self.kp += dkp
        for i, kpi in enumerate(self.kp):
            if kpi > self.gain_up[i]:
                self.kp[i] = self.gain_up[i]
            elif kpi < self.gain_low[i]:
                self.kp[i] = self.gain_low[i]

        kd = 2*np.sqrt(self.kp)
        # set controller gain
        self.controller.set_gain(self.kp, kd)
        rew = 0
        for i in range(self.no_env_steps):
            # update robot state
            self.robot_state.update()
            # compute tau_cmd
            tau_cmd, status = self.mp_sliding.step()
            # set control torque
            self.robot_state.set_control_torque(tau_cmd)
            self.robot_state.update_dynamic()
            rew += self._reward_func()

        obs = self._get_obs()
        rew += -0.01 * np.linalg.norm(action)
        if status is PrimitiveStatus.EXECUTING:
            done = False
        else:
            done = True
        info = {}
        return obs, rew, done, info

    def reset(self):
        self._reset_sim()
        # reset kp
        self.kp = self.kp_init
        self.kd = self.kd_init

        # move to init pose
        self.mp1.configure(self.init_pos, self.init_quat, 0.5, self.kp_init, self.kd_init)
        # move down until contact
        u = np.array([0, 0, -1., 0, 0, 0])
        ft = np.zeros(6)
        self.mp2.configure(u, 0.01, 5., ft, self.kp_init, self.kd_init)

        # run
        self.mp1.run()
        self.mp2.run()

        # set random fd
        self.fd = np.random.uniform(low=self.fd_range[0], high=self.fd_range[1])
        s = np.random.uniform(low = self.speed_range[0], high=self.speed_range[1])

        # configure sliding motion
        u = np.array([0, -1, 0, 0, 0, 0])
        ft = np.array([0, 0, -self.fd, 0, 0, 0])
        timeout = self.d_slide / (s)
        self.mp_sliding.configure(u, s, 1000, ft, self.kp, self.kd, timeout=timeout)
        self.mp_sliding.plan()
        # update robot state
        self.robot_state.update()

        return self._get_obs()


class SimpleStateSlidingEnv(MujocoFrankaSlidingEnv):
    def __init__(self, *argv, **kwargs):
        super().__init__(*argv, **kwargs)

    def _set_observation_space(self):
        # observation space: vx, vy, fx, fy, fz, fdz, kpx, kpy, kpz
        vel_low = np.array([-0.5]*2)
        vel_up = np.array([0.5]*2)
        force_low = np.array([-np.inf]*4)
        force_up = np.array([np.inf]*4)
        # normalized gain
        norm_gain_low = np.array([-1]*3)
        norm_gain_up = np.array([1]*3)

        self.obs_low_limit = np.hstack([vel_low, force_low, norm_gain_low])
        self.obs_up_limit = np.hstack([vel_up, force_up, norm_gain_up])

        self.observation_space = gym.spaces.Box(self.obs_low_limit, self.obs_up_limit)

    def _set_action_space(self):
        # action space is normalized gain displacement delta_kp
        dgain_low = np.array([-1]*3)
        dgain_up = np.array([1]*3)
        self.action_space = gym.spaces.Box(dgain_low, dgain_up)

    def _get_obs(self):
        v = self.robot_state.get_ee_velocity()
        f = self.robot_state.get_ee_force()
        kp_norm = 2*(self.kp - self.gain_low) / (self.gain_up - self.gain_low) -1
        return np.hstack([v[:2], f[:3], self.fd, kp_norm[:3]])

    def _action_to_gain_map(self, a):
        return (a + 1)*(self.dgain_up[:3] - self.dgain_low[:3])/2 + self.dgain_low[:3]

    def step(self, action, render=False):
        if render:
            viewer = self._get_viewer()
        else:
            viewer = None

        dkp = self._action_to_gain_map(action)
        # calculate new gain
        self.kp[:3] += dkp
        for i, kpi in enumerate(self.kp):
            if kpi > self.gain_up[i]:
                self.kp[i] = self.gain_up[i]
            elif kpi < self.gain_low[i]:
                self.kp[i] = self.gain_low[i]

        kd = 2*np.sqrt(self.kp)
        # set controller gain
        self.controller.set_gain(self.kp, kd)
        rew = 0
        for i in range(self.no_env_steps):
            # update robot state
            self.robot_state.update()
            # compute tau_cmd
            tau_cmd, status = self.mp_sliding.step()
            # set control torque
            self.robot_state.set_control_torque(tau_cmd)
            self.robot_state.update_dynamic()
            rew += self._reward_func()

        obs = self._get_obs()
        rew += -0.01 * np.linalg.norm(action)
        if status is PrimitiveStatus.EXECUTING:
            done = False
        else:
            done = True
        info = {}
        return obs, rew, done, info
