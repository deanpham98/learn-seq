import os
import numpy as np
import gym
import mujoco_py
from ..insertion_base import InsertionBaseEnv
from .mujoco_env import MujocoEnv
from learn_seq.utils.general import get_mujoco_model_path
from learn_seq.utils.mujoco import get_geom_pose, get_geom_size, quat2mat,\
            set_state, quat2vec
from learn_seq.primitive.container import PrimitiveContainer
from learn_seq.controller.hybrid import HybridController

KP_DEFAULT = np.array([1000]*3 + [60]*3)

class MujocoInsertionEnv(InsertionBaseEnv, MujocoEnv):
    def __init__(self,
                 xml_model_name,
                 robot_state,
                 primitive_list,
                 peg_pos_range,
                 peg_rot_range,
                 initial_pos_range,
                 initial_rot_range,
                 controller_class=HybridController,
                 **controller_kwargs
                 ):
        # mujoco model
        mujoco_path = get_mujoco_model_path()
        model_path = os.path.join(mujoco_path, xml_model_name)
        MujocoEnv.__init__(self, model_path)
        # init robot position for reset
        self.init_qpos = np.array([0, -np.pi/4, 0, -3 * np.pi/4, 0, np.pi/2, np.pi / 4, 0.015, 0.015])
        self._eps_time = 0
        self._reset_sim()

        self.robot_state = robot_state
        # hole base pose relative to world frame
        base_pos = self.data.get_body_xpos("hole").copy()
        base_quat = self.data.get_body_xquat("hole").copy()
        base_mat = quat2mat(base_quat)

        base_half_height = get_geom_size(self.model, "base")[2]
        base_origin = get_geom_pose(self.model, "base")[0]   # in "hole" body frame
        base_to_hole_pos = np.array([0, 0, base_half_height + base_origin[2]])
        # hole pos in world coordinate frame
        self.hole_pos = base_pos + base_mat.dot(base_to_hole_pos)
        self.hole_quat = base_quat

        # get hole_depth
        # TODO: change square_pih to this format
        hole_depth = get_geom_size(self.model, "hole1")
        self.primitive_list = primitive_list
        InsertionBaseEnv.__init__(
            self,
            hole_pos=self.hole_pos,
            hole_quat=self.hole_quat,
            hole_depth=hole_depth,
            peg_pos_range=peg_pos_range,
            peg_rot_range=peg_rot_range,
            initial_pos_range=initial_pos_range,
            initial_rot_range=initial_rot_range,)

        # controller and primitives
        self.controller = controller_class(robot_state, **controller_kwargs)
        self.container = PrimitiveContainer(robot_state, self.controller,
                            self.tf_pos, self.tf_quat)

        # kp_init
        self.kp_init = controller_kwargs.get("kp_init", KP_DEFAULT)
        self.kd_init = 2*np.sqrt(self.kp_init)

    def _reset_sim(self):
        init_qvel = np.zeros(self.model.nv)
        set_state(self.sim, self.init_qpos, init_qvel)
        # clear control
        self.sim.data.ctrl[:] = 0

    def _get_obs(self):
        if not self.robot_state.is_update():
            self.robot_state.update()
        p, q = self.robot_state.get_pose(self.tf_pos, self.tf_quat)
        # quat to angle axis
        r = quat2vec(q)
        if r[0] < 0:
            angle = np.linalg.norm(r)
            r = -r*(2*np.pi - angle)
        f = self.robot_state.get_ee_force(frame_quat=self.tf_quat)
        obs = np.hstack((p, r, f))
        # normalize
        obs[:6] = 2*(obs[:6] - self.obs_low_limit) \
                  / (self.obs_up_limit - self.obs_low_limit) - 1
        return obs

    def _set_action_space(self):
        no_primitives = len(self.primitive_list)
        self.action_space = gym.spaces.Discrete(no_primitives)

    def step(self):
        pass

    def reset_to(self, p, q):
        self._reset_sim()
        self._eps_time = 0
        param = dict(pt=p, qt=q, ft=np.zeros(6), s=0.5,
                     kp=self.kp_init, kd=self.kd_init)
        self.container.run("move2target", param)
        return self._get_obs()
