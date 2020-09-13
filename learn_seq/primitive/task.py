import numpy as np
from learn_seq.primitive.base import FixedGainTaskPrimitive
from learn_seq.utils.mujoco import integrate_quat, pose_transform,\
        transform_spatial, similarity_transform, quat_error

class Move2Target(FixedGainTaskPrimitive):
    def __init__(self,
                 robot_state,
                 controller,
                 tf_pos,
                 tf_quat,
                 timeout=2.,
                 **kwargs):
        super().__init__(robot_state, controller,\
                         tf_pos, tf_quat, timeout,
                        **kwargs)

        # init
        self.pt = np.zeros(3)               # target pos
        self.qt = np.array([1., 0, 0, 0])   # target quat
        self.ft = np.zeros(6)               # target force
        self.s = 0.                         # speed factor
        # trajectory params
        self.p0 = np.zeros(3)               # init position
        self.q0 = np.array([1., 0, 0, 0])   # init orientation
        self.vt_synch = np.zeros(6)         # planned velocity
        self.t_plan = 0.                    # planned execution time
        self.S_mat = np.zeros((6, 6))           # fixed selection matrix
        # check whether the primitive is configured
        self.isConfigured = False

    def configure(self, pt, qt, ft, s):
        """Configure the primitive for the target pos, target quat, target force and
        a speed factor. the target velocity is computed by vd = s*v_max

        All the target is defined relative to the task frame and transformed to
        the base frame in this function"""
        # transform to base frame
        self.pt, self.qt = pose_transform(pt, qt, self.tf_pos, self.tf_quat)
        # transform force and velocity to base frame
        self.ft = transform_spatial(ft, self.tf_quat)
        vt = s*self.xdot_max
        self.vt = transform_spatial(vt, self.tf_quat)
        # selection matrix in task frame
        s_vector = (ft == 0.).astype(int)
        S_mat = np.diag(s_vector)
        # selection matrix in base frame
        self.S_mat = S_mat.copy()
        self.S_mat[:3, :3] = similarity_transform(S_mat[:3, :3], self.tf_quat)
        self.S_mat[3:, 3:] = similarity_transform(S_mat[3:, 3:], self.tf_quat)
        if not self.isConfigured:
            self.isConfigured = True

    def step(self):
        # update time
        self.t_exec += self.dt
        self.timeout_count -= self.dt

        # limit execution time
        if self.t_exec > self.t_plan:
            self.t_exec = self.t_plan

        # position/orientation command
        pd = self.p0 + self.vt_synch[:3]*self.t_exec
        rd = self.vt_synch[3:]*self.t_exec
        qd = integrate_quat(self.q0, rd, 1)

        # velocity command
        if self.t_exec < self.t_plan:
            vd = self.vt_synch.copy()
        else:
            vd = np.zeros(6)

        # desired force
        fd = self.ft

        # compute tau command
        tau_cmd = self.controller.forward_ctrl(pd, qd, vd, fd, self.S_mat)

        return tau_cmd, self.is_terminate()

    def is_terminate(self):
        # TODO better if detect steady state?
        isSettle = self.t_exec > self.t_plan*1.05
        isTimeout = self.timeout_count < 0
        if isTimeout or isSettle:
            return True
        return False

    def plan(self):
        """Plan a straight path in task space"""
        super().plan()
        # plan from the previous command pose
        self.p0, self.q0 = self.controller.get_pose_cmd()

        # positional and rotational error
        err = np.zeros(6)
        err[:3] = self.pt - self.p0
        err[3:] = quat_error(self.q0, self.qt)

        t_arr = err / self.vt
        self.t_plan=  np.max(np.abs(t_arr))
        self.vt_synch = err / self.t_plan

    def run(self, viewer=None):
        assert self.isConfigured
        return super().run(viewer=viewer)

class Move2Contact(FixedGainTaskPrimitive):
    def __init__(self,
                 robot_state,
                 controller,
                 tf_pos,
                 tf_quat,
                 timeout=2.,
                 **kwargs):
        super().__init__(robot_state, controller,
                         tf_pos, tf_quat, timeout,
                        **kwargs)

class Displacement(FixedGainTaskPrimitive):
    def __init__(self,
                 robot_state,
                 controller,
                 tf_pos,
                 tf_quat,
                 timeout=2.,
                 **kwargs):
        super().__init__(robot_state, controller,
                         tf_pos, tf_quat, timeout,
                        **kwargs)

class AdmittanceMotion(FixedGainTaskPrimitive):
    def __init__(self,
                 robot_state,
                 controller,
                 tf_pos,
                 tf_quat,
                 timeout=2.,
                 **kwargs):
        super().__init__(robot_state, controller,
                         tf_pos, tf_quat, timeout,
                        **kwargs)
