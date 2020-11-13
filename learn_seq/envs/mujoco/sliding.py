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


class MujocoFrankaSlidingEnv(gym.Env, MujocoEnv):
    def __init__(self,
                 fd_range=[5, 10],
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
        self.init_pos = np.array([0.53, 0.052, 0.1788])
        self.init_quat = np.array([0, 1, 0, 0])

        # move to target
        tf_pos = np.array([0, 0, 0])
        tf_quat = np.array([1, 0, 0, 0])
        self.mp1 = Move2Target(
            self.robot_state, self.controller, tf_pos, tf_quat)
        self.mp2 = Move2Contact(
            self.robot_state, self.controller, tf_pos, tf_quat)

        # init gain
        self.kp_init = np.array([1000, 1000, 1000, 60, 60, 60])
        self.kd_init = 2*np.sqrt(self.kp_init)

        # observation space: velocity, force, desired force
        self.observation_space = self.velocity()

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
        return np.zeros(6)

    # def _normalize_obs(self)

    def _reward_func(self, obs):
        return 0

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

        obs = self._get_obs()
        rew = _reset_func(obs)
        done = False
        info = {}
        return obs, rew, done, info

    def reset(self):
        self._reset_sim()
        # move to init pose
        self.mp1.configure(self.init_pos, self.init_quat, 0.5, self.kp_init, self.kd_init)
        # move down until contact
        u = np.array([0, 0, -1, 0, 0, 0])
        ft = np.zeros(6)
        self.mp2.configure(u, 0.01, 5, ft, self.kp_init, self.kd_init)

        # run
        self.mp1.run()
        self.mp2.run()

        return self._get_obs()
