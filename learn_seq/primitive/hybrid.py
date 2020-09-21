""" Primitives for hybrid controller"""
import numpy as np
from learn_seq.primitive.base import Primitive
from learn_seq.utils.mujoco import integrate_quat, pose_transform,\
        transform_spatial, similarity_transform, quat_error

class FixedGainTaskPrimitive(Primitive):
    """Primitives where motion are defined with respect to a task frame, and
    has fixed gains during execution

    :param np.array(3) tf_pos: origin of the task frame (relative to the base frame)
    :param np.array(4) tf_quat : orientation of the axis of the task frame (relative
                            to the base frame)
    """
    def __init__(self,
                 robot_state,
                 controller,
                 tf_pos,
                 tf_quat,
                 timeout=2.,
                 **kwargs):
        super().__init__(robot_state, controller, timeout, **kwargs)
        self.set_task_frame(tf_pos, tf_quat)
        # init
        self.pt = np.zeros(3)               # target pos
        self.qt = np.array([1., 0, 0, 0])   # target quat
        self.ft = np.zeros(6)               # target force
        self.S_mat = np.zeros((6, 6))       # fixed selection matrix
        #
        self.p0 = np.zeros(3)               # init position
        self.q0 = np.array([1., 0, 0, 0])   # init orientation

    def configure(self, kp, kd, timeout=None):
        self.controller.set_gain(kp, kd)
        self.timeout = timeout or self.timeout

    def set_task_frame(self, tf_pos, tf_quat):
        self.tf_pos = tf_pos
        self.tf_quat = tf_quat

    def set_controller_gain(self, kp, kd):
        self.controller.set_gain(kp, kd)

    def _transform_selection_matrix(self, ft):
        s_vector = (ft == 0.).astype(float)
        S_mat = np.diag(s_vector)
        # selection matrix in base frame
        S_mat_base = S_mat.copy()
        S_mat_base[:3, :3] = similarity_transform(S_mat[:3, :3], self.tf_quat)
        S_mat_base[3:, 3:] = similarity_transform(S_mat[3:, 3:], self.tf_quat)
        return S_mat_base

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
        self.s = 0.                         # speed factor
        # trajectory params
        self.vt_synch = np.zeros(6)         # planned velocity
        self.t_plan = 0.                    # planned execution time
        # check whether the primitive is configured
        self.isConfigured = False

    def __repr__(self):
        return "move to pos {}, quat {} with desired force {}, speed {}"\
                .format(self.pt, self.qt, self.ft, self.vt)

    def configure(self, pt, qt, ft, s, kp, kd, timeout=None):
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
        self.S_mat = self._transform_selection_matrix(ft)
        if not self.isConfigured:
            self.isConfigured = True
        super().configure(kp, kd, timeout=timeout)

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
    """Move in a direction until f > f_thresh."""
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

        # init
        self.thresh = 0.                # force thresh - stop condition
        self.isContact = False          # contact detection
        self.n_step_settle = 10

    def __repr__(self):
        return "move in direction {}, with desired force {}, until f > {}"\
                .format(self.vt, self.ft, self.thresh)

    def configure(self, u, s, fs, ft, kp, kd, timeout=None):
        """ Configure the controller, same as Move2Target

        :param np.array(6) u: move direction
        :param float s: speed (m/s or rad/s)
        :param type fs: threshold force (N or Nm).
        """
        # normalize
        u = u/np.linalg.norm(u)
        # desired velocity in task frame
        vt = s * u
        assert (vt < self.xdot_max).all()
        self.move_dir = u
        self.thresh = fs
        # transform vt to base frame
        self.vt = transform_spatial(vt, self.tf_quat)
        # force in base frame
        self.ft = transform_spatial(ft, self.tf_quat)
        # selection matrix
        self.S_mat = self._transform_selection_matrix(ft)
        super().configure(kp, kd, timeout=timeout)

    def step(self):
        # check stop condition
        done = self.is_terminate()

        # update time
        self.timeout_count -= self.dt
        if not done and not self.isContact:
            self.t_exec += self.dt
            # velocity
            vd = self.vt.copy()
        else:
            vd = np.zeros(6)

        # position/orientation command
        pd = self.p0 + self.vt[:3]*self.t_exec
        rd = self.vt[3:]*self.t_exec
        qd = integrate_quat(self.q0, rd, 1)

        #
        fd = self.ft
        # compute tau command
        tau_cmd = self.controller.forward_ctrl(pd, qd, vd, fd, self.S_mat)

        return tau_cmd, done

    def is_terminate(self):
        f_proj = self._project_force()
        if f_proj < -self.thresh:
            self.isContact = True

        if self.isContact:
            self.n_step_settle_count -= 1

        isSettle = self.n_step_settle_count < 0
        isTimeout = self.timeout_count < 0
        if isSettle or isTimeout:
            return True
        else:
            return False

    def plan(self):
        super().plan()
        self.isContact = False
        self.n_step_settle_count = self.n_step_settle
        # plan from last controller pose command
        self.p0, self.q0 = self.controller.get_pose_cmd()

    def _project_force(self):
        # get force acting on the ee, relative to the base frame
        q0 = np.array([1., 0, 0, 0])
        f = self.robot_state.get_ee_force(frame_quat=q0)

        # project fe to the move direction
        f_proj = f.dot(self.move_dir)
        return f_proj


class Displacement(Move2Contact):
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
        self.delta_d = 0                # desired displacement
        self.pr0 = np.zeros(6)          # initial position
        self.qr0 = np.zeros(6)          # initial orientationt

    def __repr__(self):
        return "move in direction {} with desired force {}, until f > {} or dp > {}"\
                .format(self.vt, self.ft, self.thresh, self.delta_d)

    def configure(self, u, s, fs, ft, delta_d, kp, kd, timeout=None):
        """Same as Move2Contact

        :param type delta_d: Desired displacement in the move direction.
        """
        super().configure(u, s, fs, ft, kp, kd, timeout=timeout)
        self.delta_d = delta_d
        self.isDisplaceAchieved = False

    def is_terminate(self):
        f_proj = self._project_force()
        if f_proj < -self.thresh:
            self.isContact = True

        # calculate displacement
        dp = np.zeros(6)
        p, q = self.robot_state.get_pose()
        dp[:3] = p - self.pr0
        dp[3:] = quat_error(self.qr0, q)
        if np.abs(dp.dot(self.move_dir)) > self.delta_d:
            self.isDisplaceAchieved = True

        if self.isContact or self.isDisplaceAchieved:
            self.n_step_settle_count -= 1

        isSettle = self.n_step_settle_count < 0
        isTimeout = self.timeout_count < 0
        if isSettle or isTimeout:
            return True
        else:
            return False

    def plan(self):
        super().plan()
        self.isDisplaceAchieved = False
        p, q = self.robot_state.get_pose()
        self.pr0 = p
        self.qr0 = q

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
        self.kd_adt = np.zeros(6)
        self.depth_thresh = 0.

    def __repr__(self):
        return "control desired force {} in z direction, and 0-torque in other until depth < {}"\
                .format(self.ft, self.depth_thresh)

    def configure(self, kd_adt, ft, depth_thresh, kp, kd, timeout=None):
        self.kd_adt = kd_adt
        self.depth_thresh = depth_thresh
        # force in base frame
        self.ft = transform_spatial(ft, self.tf_quat)
        # selection matrix
        self.S_mat = self._transform_selection_matrix(ft)
        super().configure(kp, kd, timeout=timeout)

    def is_terminate(self):
        p_task, q_task = self.robot_state.get_pose(self.tf_pos, self.tf_quat)
        if p_task[2] < self.depth_thresh or self.timeout_count < 0:
            return True
        return False

    def step(self):
        # update time
        self.t_exec += self.dt
        self.timeout_count -= self.dt
        # ee force in ee frame
        f_e = self.robot_state.get_ee_force()
        # ee vel in ee frame
        vd_e = self.kd_adt * f_e
        # limit velocity
        vd_e = np.maximum(-self.xdot_max, np.minimum(self.xdot_max, vd_e))
        # transform to base frame
        p, q= self.robot_state.get_pose()
        vd = transform_spatial(vd_e, q)

        # position/orientation based on vd
        pc, qc = self.controller.get_pose_cmd()
        pd = pc + vd[:3]*self.dt
        rd = vd[3:]*self.dt
        qd = integrate_quat(qc, rd, 1)

        #
        fd = self.ft

        # compute tau command
        tau_cmd = self.controller.forward_ctrl(pd, qd, vd, fd, self.S_mat)

        return tau_cmd, self.is_terminate()

    def plan(self):
        super().plan()
        self.p0, self.q0 = self.controller.get_pose_cmd()
