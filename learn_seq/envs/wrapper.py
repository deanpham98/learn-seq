import numpy as np
import gym
from gym import Wrapper
from gym.core import ObservationWrapper
from learn_seq.utils.mujoco import integrate_quat, inverse_frame, pose_transform, quat2vec
from learn_seq.utils.gym import DynamicDiscrete
from learn_seq.mujoco_wrapper import MujocoModelWrapper

class BaseInsertionWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.robot_state = env.robot_state

    def get_task_frame(self):
        return self.env.get_task_frame()

    def set_task_frame(self, *argv, **kwargs):
        return self.env.set_task_frame(*argv, **kwargs)

    def step(self, *argv, **kwargs):
        return self.env.step(*argv, **kwargs)

class TrainInsertionWrapper(BaseInsertionWrapper):
    """Vary the hole position virtually, and assume the `hole_pos` and
    `hole_quat` attribute of environment is the true hole pose.
    Use for training with RL. At the beginning of each episode, a random error
    is generated and added to the real hole pose

    :param tuple hole_pos_error_range: lower bound and upper bound of hole pos error in m
    :param tuple hole_rot_error_range: lower bound and upper bound of hole orientation error in rad

    Example: hole_pos_error_range = ([-0.001, -0.001, -0.001], [0.001, 0.001, 0.001])
             -> random error in [-1, 1] mm
    """
    def __init__(self, env,
                 hole_pos_error_range,
                 hole_rot_error_range):
        super().__init__(env)
        self.pos_error_range = hole_pos_error_range
        self.rot_error_range = hole_rot_error_range
        # true hole pose
        self.hole_pos, self.hole_quat = self.get_task_frame()

    def reset(self):
        # add noise to create virtual estimated hole pose
        hole_pos = self.hole_pos + np.random.uniform(self.pos_error_range[0],\
                                        self.pos_error_range[1])
        hole_rot_rel = np.random.uniform(self.rot_error_range[0],\
                                self.rot_error_range[1])
        hole_quat = integrate_quat(self.hole_quat, hole_rot_rel, 1)
        self.set_task_frame(hole_pos, hole_quat)
        return self.env.reset()

# NOTE: the logic (observation -> action spaces) is not incorporated here. It is
# in the agent and NN model
class StructuredActionSpaceWrapper(TrainInsertionWrapper):
    """Divide action spaces into 2 subspaces, contact and non-contact subspaces

    :param list spaces_idx_list: the list of action indices corresponding to
                                 each subspaces. see utils/gym/DynamicDiscrete
                                 for more detail

    """
    def __init__(self, env,
                 hole_pos_error_range,
                 hole_rot_error_range,
                 spaces_idx_list):
        no_primitives = len(env.unwrapped.primitive_list)
        env.action_space = DynamicDiscrete(no_primitives, spaces_idx_list)
        super().__init__(env, hole_pos_error_range, hole_rot_error_range)

class SetMujocoModelWrapper(BaseInsertionWrapper):
    def __init__(self, env, config_dict):
        super().__init__(env)
        self.model_wrapper = MujocoModelWrapper(env.unwrapped.model)
        for key, val in config_dict.items():
            if val is not None:
                self.model_wrapper.set(key, val)
        print(self.model_wrapper.get_mass())
        print(self.model_wrapper.get_clearance())
        print(self.model_wrapper.get_friction())
        print(self.model_wrapper.get_joint_damping())

class InitialPoseWrapper(BaseInsertionWrapper):
    """Change the initial pose (reset pose) of the peg

    :param np.array(3) p0: Initial position
    :param np.array(3) r0: Initial rotation vector

    Position and rotaion vector is defined relative to the task frame
    """
    def __init__(self, env, p0, r0):
        super().__init__(env)
        env.unwrapped.initial_pos_mean = p0
        env.unwrapped.initial_rot_mean = r0

class HolePoseWrapper(BaseInsertionWrapper):
    """Change hole pose in the mujoco xml model.

    :param np.array(3) hole_body_pos: the position of the "hole" body object in the xml model
    :param np.array(4) hole_body_quat: orientation (in quaternion)

    """
    def __init__(self, env, hole_body_pos, hole_body_quat):
        super().__init__(env)
        self.model_wrapper = MujocoModelWrapper(env.model)
        self.model_wrapper.set_hole_pose(hole_body_pos, hole_body_quat)
        p, q, _ = env.unwrapped._hole_pose_from_model()
        env.set_task_frame(p, q)

class QuaternionObservationWrapper(BaseInsertionWrapper):
    def __init__(self, env):
        obs_high = env.unwrapped.observation_space.high
        obs_low = env.unwrapped.observation_space.low
        obs_high = np.hstack((obs_high[:3], np.array([1, 1, 1, 1.])))
        obs_low = np.hstack((obs_low[:3], np.array([-1, -1, -1, -1.])))
        env.unwrapped.observation_space = gym.spaces.Box(obs_low, obs_high)
        env.observation_space = gym.spaces.Box(obs_low, obs_high)
        super().__init__(env)

    def observation(self, obs):
        tf_pos, tf_quat = self.get_task_frame()
        _, q = self.robot_state.get_pose(tf_pos, tf_quat)
        if q.dot(self.q_prev) < 0:
            q = -q
        self.q_prev = q.copy()
        return np.hstack((obs[:3], q, obs[6:]))

    def reset(self, **kwargs):
        self.q_prev = self.env.unwrapped.target_quat
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def step(self, action, **kwargs):
        observation, reward, done, info = self.env.step(action, **kwargs)
        return self.observation(observation), reward, done, info

class FixedHolePoseErrorWrapper(StructuredActionSpaceWrapper):
    """Vary the hole position virtually, and assume the `hole_pos` and
    `hole_quat` attribute of environment is the true hole pose.
    Use for training with RL

    :param tuple hole_pos_error_range: lower bound and upper bound of hole pos error.
                                 Example: ([-0.001]*3, [0.001]*3)
    :param tuple hole_rot_error_range: lower bound and upper bound of hole orientation error.

    """
    def __init__(self, env,
                 hole_pos_error,
                 hole_rot_error,
                 spaces_idx_list):
        super().__init__(env, hole_pos_error, hole_rot_error, spaces_idx_list)
        self.pos_error = hole_pos_error
        self.rot_error = hole_rot_error
        # true hole pose
        self.hole_pos = env.tf_pos.copy()
        self.hole_quat = env.tf_quat.copy()

    def reset(self):
        # add noise to create virtual estimated hole pose
        pos_dir = np.zeros(3)
        pos_dir[:2] = (np.random.random(2) - 0.5) * 2
        pos_dir[:2] = pos_dir[:2] / np.max(np.abs(pos_dir))
        hole_pos = self.hole_pos + self.pos_error * pos_dir

        rot_dir = (np.random.random(3) - 0.5) * 2
        rot_dir = rot_dir / np.linalg.norm(rot_dir)
        hole_rot_rel = self.rot_error * rot_dir
        hole_quat = integrate_quat(self.hole_quat, hole_rot_rel, 1)
        self.env.set_task_frame(hole_pos, hole_quat)
        return self.env.reset()

class ControllerCommandWrapper(BaseInsertionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.obs_size = self.observation_space.shape[0]
        obs_low_lim = self.observation_space.low
        obs_high_lim = self.observation_space.high
        obs_low_lim = np.hstack([obs_low_lim, obs_low_lim[:6]])
        obs_high_lim = np.hstack([obs_high_lim, obs_high_lim[:6]])
        self.observation_space = gym.spaces.Box(obs_low_lim, obs_high_lim)

    def reset(self):
        obs = self.env.reset()
        tf_pos, tf_quat = self.get_task_frame()
        inv_tf_pos, inv_tf_quat = inverse_frame(tf_pos, tf_quat)
        p_cmd, q_cmd = self.env.controller.get_pose_cmd()
        p_cmd, q_cmd = pose_transform(p_cmd, q_cmd, inv_tf_pos, inv_tf_quat)
        r_cmd = quat2vec(q_cmd, self.env.unwrapped.r_prev)
        pose_norm = self.env.unwrapped._normalize_obs(np.hstack([p_cmd, r_cmd]))
        new_obs = np.hstack([obs[:self.obs_size], pose_norm, obs[self.obs_size:]])
        return new_obs

    def step(self, action, **kwargs):
        obs, reward, done, info = self.env.step(action, **kwargs)
        tf_pos, tf_quat = self.get_task_frame()
        inv_tf_pos, inv_tf_quat = inverse_frame(tf_pos, tf_quat)
        p_cmd, q_cmd = self.env.controller.get_pose_control_cmd()
        p_cmd, q_cmd = pose_transform(p_cmd, q_cmd, inv_tf_pos, inv_tf_quat)
        r_cmd = quat2vec(q_cmd, self.env.unwrapped.r_prev)
        pose_norm = self.env.unwrapped._normalize_obs(np.hstack([p_cmd, r_cmd]))
        new_obs = np.hstack([obs[:self.obs_size], pose_norm, obs[self.obs_size:]])
        return new_obs, reward, done, info

class TaskFrameWrapper(BaseInsertionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.obs_size = self.observation_space.shape[0]
        obs_low_lim = self.observation_space.low
        obs_high_lim = self.observation_space.high
        obs_low_lim = np.hstack([obs_low_lim, obs_low_lim[:6]])
        obs_high_lim = np.hstack([obs_high_lim, obs_high_lim[:6]])
        self.observation_space = gym.spaces.Box(obs_low_lim, obs_high_lim)

    def reset(self):
        obs = self.env.reset()
        tf_pos, tf_quat = self.get_task_frame()
        # tf_r = quat2vec(tf_quat, self.env.unwrapped.r_prev)
        tf_r = quat2vec(tf_quat)
        new_obs = np.hstack([obs[:self.obs_size], tf_pos, tf_r, obs[self.obs_size:]])
        return new_obs

    def step(self, action, **kwargs):
        obs, reward, done, info = self.env.step(action, **kwargs)
        tf_pos, tf_quat = self.get_task_frame()
        tf_r = quat2vec(tf_quat, self.env.unwrapped.r_prev)
        new_obs = np.hstack([obs[:self.obs_size], tf_pos, tf_r, obs[self.obs_size:]])
        return new_obs, reward, done, info


class PrimitiveStateWrapper(BaseInsertionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.obs_size = self.observation_space.shape[0]
        obs_low_lim = self.observation_space.low
        obs_high_lim = self.observation_space.high
        obs_low_lim = np.hstack([obs_low_lim,  [0, 0]])
        obs_high_lim = np.hstack([obs_high_lim, [1, 1]])
        self.observation_space = gym.spaces.Box(obs_low_lim, obs_high_lim)

    def reset(self):
        obs = self.env.reset()
        new_obs = np.hstack([obs[:self.obs_size], [1, 0], obs[self.obs_size:]])
        return new_obs

    def step(self, action, **kwargs):
        obs, reward, done, info = self.env.step(action, **kwargs)
        if info["mp_status"] == 1:
            new_obs = np.hstack([obs[:self.obs_size], [1, 0], obs[self.obs_size:]])
        else:
            new_obs = np.hstack([obs[:self.obs_size], [0, 1], obs[self.obs_size:]])
        return new_obs, reward, done, info
