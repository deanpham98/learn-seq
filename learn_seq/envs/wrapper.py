import numpy as np
from gym import Wrapper
from learn_seq.utils.mujoco import integrate_quat
from learn_seq.utils.gym import DynamicDiscrete
from learn_seq.mujoco_wrapper import MujocoModelWrapper


class TrainInsertionWrapper(Wrapper):
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
        self.hole_pos = env.tf_pos.copy()
        self.hole_quat = env.tf_quat.copy()

    def reset(self):
        # add noise to create virtual estimated hole pose
        hole_pos = self.hole_pos + np.random.uniform(self.pos_error_range[0],\
                                        self.pos_error_range[1])
        hole_rot_rel = np.random.uniform(self.rot_error_range[0],\
                                self.rot_error_range[1])
        hole_quat = integrate_quat(self.hole_quat, hole_rot_rel, 1)
        self.env.unwrapped.set_task_frame(hole_pos, hole_quat)
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

class SetMujocoModelWrapper(Wrapper):
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

class InitialPoseWrapper(Wrapper):
    """Change the initial pose (reset pose) of the peg

    :param np.array(3) p0: Initial position
    :param np.array(3) r0: Initial rotation vector

    Position and rotaion vector is defined relative to the task frame
    """
    def __init__(self, env, p0, r0):
        super().__init__(env)
        env.unwrapped.initial_pos_mean = p0
        env.unwrapped.initial_rot_mean = r0

class HolePoseWrapper(Wrapper):
    """Change hole pose in the mujoco xml model.

    :param np.array(3) hole_body_pos: the position of the "hole" body object in the xml model
    :param np.array(4) hole_body_quat: orientation (in quaternion)

    """
    def __init__(self, env, hole_body_pos, hole_body_quat):
        super().__init__(env)
        self.model_wrapper = MujocoModelWrapper(env.model)
        self.model_wrapper.set_hole_pose(hole_body_pos, hole_body_quat)
        p, q, _ = env.unwrapped._hole_pose_from_model()
        env.unwrapped.set_task_frame(p, q)

class FixedHolePoseErrorWrapper(Wrapper):
    """Vary the hole position virtually, and assume the `hole_pos` and
    `hole_quat` attribute of environment is the true hole pose.
    Use for training with RL

    :param tuple hole_pos_error_range: lower bound and upper bound of hole pos error.
                                 Example: ([-0.001]*3, [0.001]*3)
    :param tuple hole_rot_error_range: lower bound and upper bound of hole orientation error.

    """
    def __init__(self, env,
                 hole_pos_error,
                 hole_rot_error):
        super().__init__(env)
        self.pos_error = hole_pos_error
        self.rot_error = hole_rot_error
        # true hole pose
        self.hole_pos = env.tf_pos.copy()
        self.hole_quat = env.tf_quat.copy()

    def reset(self):
        # add noise to create virtual estimated hole pose
        pos_dir = (np.random.random(3) - 0.5) * 2
        pos_dir = pos_dir / np.linalg.norm(pos_dir)
        hole_pos = self.hole_pos + self.pos_error * pos_dir

        rot_dir = (np.random.random(3) - 0.5) * 2
        rot_dir = rot_dir / np.linalg.norm(rot_dir)
        hole_rot_rel = self.rot_error * rot_dir
        hole_quat = integrate_quat(self.hole_quat, hole_rot_rel, 1)
        self.env.set_task_frame(hole_pos, hole_quat)
        return self.env.reset()
