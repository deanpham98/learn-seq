from learn_seq.utils.mujoco import integrate_quat
from gym import Wrapper

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
        env.set_task_frame(hole_pos, hole_quat)
        return env.reset()
