import numpy as np
from gym import Wrapper
from learn_seq.utils.mujoco import integrate_quat
from learn_seq.utils.gym import DynamicDiscrete
from learn_seq.mujoco_wrapper import MujocoModelWrapper


class TrainInsertionWrapper(Wrapper):
    """Vary the hole position virtually, and assume the `hole_pos` and
    `hole_quat` attribute of environment is the true hole pose.
    Use for training with RL

    :param tuple hole_pos_error_range: lower bound and upper bound of hole pos error.
                                 Example: ([-0.001]*3, [0.001]*3)
    :param tuple hole_rot_error_range: lower bound and upper bound of hole orientation error.

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
        self.env.set_task_frame(hole_pos, hole_quat)
        return self.env.reset()

# NOTE: the logic (observation -> action spaces) is not incorporated here. It is
# in the agent and NN model
class StructuredActionSpaceWrapper(TrainInsertionWrapper):
    """Divide action spaces into different subspaces, defined by the subset of
    available actions. To be used by agent/model

    :param type env: Description of parameter `env`.
    :param type hole_pos_error_range: Description of parameter `hole_pos_error_range`.
    :param type hole_rot_error_range: Description of parameter `hole_rot_error_range`.
    :param type spaces_idx_list: Description of parameter `spaces_idx_list`.
    :attr spaces_idx_list:

    """
    def __init__(self, env,
                 hole_pos_error_range,
                 hole_rot_error_range,
                 spaces_idx_list):
        no_primitives = len(env.primitive_list)
        env.action_space = DynamicDiscrete(no_primitives, spaces_idx_list)
        super().__init__(env, hole_pos_error_range, hole_rot_error_range)

class SetMujocoModelWrapper(Wrapper):
    def __init__(self, env, config_dict):
        super().__init__(env)
        self.model_wrapper = MujocoModelWrapper(env.model)
        for key, val in config_dict.items():
            if val is not None:
                self.model_wrapper.set(key, val)
        print(self.model_wrapper.get_mass())
        print(self.model_wrapper.get_clearance())
        print(self.model_wrapper.get_friction())
        print(self.model_wrapper.get_joint_damping())

class InitialPoseWrapper(Wrapper):
    def __init__(self, env, p0, r0):
        super().__init__(env)
        env.initial_pos_mean = p0
        env.initial_rot_mean = r0
