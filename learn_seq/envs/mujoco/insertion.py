import os
import numpy as np
import gym
import mujoco_py
from ..insertion_base import InsertionBaseEnv
from .mujoco_env import MujocoEnv
from learn_seq.utils.general import get_mujoco_model_path
from learn_seq.utils.mujoco import get_geom_pose, get_geom_size, quat2mat,\
            set_state, quat2vec, get_body_pose
from learn_seq.primitive.container import PrimitiveContainer
from learn_seq.controller.hybrid import HybridController
from learn_seq.controller.robot_state import RobotState

KP_DEFAULT = np.array([1000]*3 + [60]*3)

class MujocoInsertionEnv(InsertionBaseEnv, MujocoEnv):
    def __init__(self,
                 xml_model_name,
                 primitive_list,
                 peg_pos_range,
                 peg_rot_range,
                 initial_pos_range,
                 initial_rot_range,
                 depth_thresh=0.95,
                 controller_class=HybridController,
                 **controller_kwargs
                 ):
        # mujoco model
        mujoco_path = get_mujoco_model_path()
        model_path = os.path.join(mujoco_path, xml_model_name)
        MujocoEnv.__init__(self, model_path)
        self.robot_state = RobotState(self.sim, "end_effector")
        self.depth_thresh = depth_thresh
        # init robot position for reset
        self.init_qpos = np.array([0, -np.pi/4, 0, -3 * np.pi/4, 0, np.pi/2, np.pi / 4, 0.015, 0.015])
        self._eps_time = 0
        # controller
        self.controller = controller_class(self.robot_state, **controller_kwargs)
        self._reset_sim()

        hole_pos, hole_quat, hole_depth = self._hole_pose_from_model()

        # primitives
        self.container = PrimitiveContainer(self.robot_state, self.controller,
                            hole_pos, hole_quat)
        self.primitive_list = primitive_list

        InsertionBaseEnv.__init__(
            self,
            hole_pos=hole_pos,
            hole_quat=hole_quat,
            hole_depth=hole_depth,
            peg_pos_range=peg_pos_range,
            peg_rot_range=peg_rot_range,
            initial_pos_range=initial_pos_range,
            initial_rot_range=initial_rot_range,)

        # kp_init
        self.kp_init = controller_kwargs.get("kp_init", KP_DEFAULT)
        self.kd_init = 2*np.sqrt(self.kp_init)

    def _reset_sim(self):
        init_qvel = np.zeros(self.model.nv)
        set_state(self.sim, self.init_qpos, init_qvel)
        # clear control
        self.sim.data.ctrl[:] = 0
        # reset controller cmd
        self.controller.reset_pose_cmd()
        self.controller.reset_tau_cmd()

    def _hole_pose_from_model(self):
        # get hole_depth
        # TODO: change square_pih to this format
        hole_depth = get_geom_size(self.model, "hole1")[2]*2
        # hole base pose relative to world frame
        base_pos, base_quat = get_body_pose(self.model, "hole")
        base_mat = quat2mat(base_quat)

        base_half_height = get_geom_size(self.model, "base")[2]
        base_origin = get_geom_pose(self.model, "base")[0]   # in "hole" body frame
        base_to_hole_pos = np.array([0, 0, base_half_height + base_origin[2] + hole_depth])
        # hole pos in world coordinate frame
        hole_pos = base_pos + base_mat.dot(base_to_hole_pos)
        hole_quat = base_quat
        return hole_pos, hole_quat, hole_depth

    def _get_obs(self):
        if not self.robot_state.is_update():
            self.robot_state.update()
        p, q = self.robot_state.get_pose(self.tf_pos, self.tf_quat)
        # quat to angle axis
        r = quat2vec(q)
        f = self.robot_state.get_ee_force(frame_quat=self.tf_quat)
        obs = np.hstack((p, r, f))
        return obs

    def _normalize_obs(self, obs):
        # normalize
        obs[:6] = 2*(obs[:6] - self.obs_low_limit) \
                  / (self.obs_up_limit - self.obs_low_limit) - 1
        return obs

    def _set_action_space(self):
        no_primitives = len(self.primitive_list)
        self.action_space = gym.spaces.Discrete(no_primitives)

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

        rwd_terminal = 0
        if self._is_success(pos):
            rwd_terminal = 5.

        r = rwd_near +rwd_inside +rwd_insert +rwd_short_length +rwd_terminal
        return r

    # TODO check also for rotation
    def _is_limit_reach(self, p):
        return np.max(np.abs(p[:2] - self.target_pos[:2])) > self.obs_up_limit[0]

    def _is_success(self, p):
        pos_thresh = 0.01
        isDepthReach = p[2] < self.target_pos[2]*self.depth_thresh
        isInHole = np.linalg.norm(p[:2] - self.target_pos[:2]) < pos_thresh
        return (isDepthReach and isInHole)

    def viewer_setup(self):
        self.viewer.cam.distance=0.43258
        self.viewer.cam.lookat[:] = [0.517255, 0.0089188, 0.25619 ]
        self.viewer.cam.elevation = -20.9
        self.viewer.cam.azimuth = 132.954

    def step(self, action, render=False):
        if render:
            viewer = self._get_viewer()
        else:
            viewer = None
        type, param = self.primitive_list[action]
        t_exec = self.container.run(type, param, viewer=viewer)
        self._eps_time += t_exec

        #
        obs = self._get_obs()
        reward = self._reward_func(obs, t_exec)
        isLimitReach = self._is_limit_reach(obs[:3])
        isSuccess = self._is_success(obs[:3])
        isTimeout = self._eps_time > 20.
        done = isTimeout or isLimitReach or isSuccess

        info = {"success": isSuccess,
                "insert_depth": obs[2]}

        return self._normalize_obs(obs), reward, done, info

    def reset_to(self, p, q):
        self._reset_sim()
        self._eps_time = 0
        param = dict(pt=p, qt=q, ft=np.zeros(6), s=0.5,
                     kp=self.kp_init, kd=self.kd_init)
        self.container.run("move2target", param)
        obs = self._get_obs()
        return self._normalize_obs(obs)

    def set_task_frame(self, p, q):
        InsertionBaseEnv.set_task_frame(self, p, q)
        # set task frame to all primitives
        self.container.set_task_frame(p, q)
