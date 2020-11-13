import numpy as np
from learn_seq.primitive.base import Primitive, PrimitiveStatus
from learn_seq.utils.mujoco import integrate_quat, pose_transform,\
        transform_spatial, similarity_transform, quat_error, inverse_frame

class TaskPrimitive(Primitive):
    """Primitives where motion are defined with respect to a task frame

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
        #
        self.p0 = np.zeros(3)               # init position
        self.q0 = np.array([1., 0, 0, 0])   # init orientation

    def configure(self, timeout=None):
        self.timeout = timeout or self.timeout

    def set_task_frame(self, tf_pos, tf_quat):
        self.tf_pos = tf_pos
        self.tf_quat = tf_quat

    def set_controller_gain(self, kp, kd):
        self.controller.set_gain(kp, kd)


class Move2Target(TaskPrimitive):
    def __init__(self, *argv, **kwargs):
        super().__init__(*argv, **kwargs)
        # init
        self.s = 0.                         # speed factor
        # trajectory params
        self.vt_synch = np.zeros(6)         # planned velocity
        self.t_plan = 0.                    # planned execution time
        # check whether the primitive is configured
        self.isConfigured = False
        # outer pos control
        self.kp_c = np.array([0.1]*6)
        self.pos_thresh = 5e-4
        self.rot_thresh = 1e-2

    def configure(self, pt, qt, s, kp, kd, timeout=None):
        """Configure the primitive for the target pos, target quat, target force and
        a speed factor. the target velocity is computed by vd = s*v_max

        All the target is defined relative to the task frame and transformed to
        the base frame in this function"""
        # set gain
        self.set_controller_gain(kp, kd)
        # transform to base frame
        self.pt, self.qt = pose_transform(pt, qt, self.tf_pos, self.tf_quat)
        # transform force and velocity to base frame
        vt = s*self.xdot_max
        self.vt = transform_spatial(vt, self.tf_quat)
        # selection matrix in task frame
        if not self.isConfigured:
            self.isConfigured = True
        super().configure(timeout=timeout)

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

        # update compliant frame
        p_cmd, q_cmd = self.controller.get_pose_cmd()
        p, q  = self.robot_state.get_pose()
        ep = (pd - p)
        er = self.kp_c[3:] * quat_error(q, qd)
        pc = p_cmd + self.kp_c[:3]*ep
        qc = integrate_quat(q_cmd, er, 1)
        # compute tau command
        tau_cmd = self.controller.forward_ctrl(pd, qd, vd, np.zeros(6))

        return tau_cmd, self._status()

    def _status(self):
        p, q = self.robot_state.get_pose()
        isSuccess = np.linalg.norm(self.pt - p) < self.pos_thresh and \
                    np.linalg.norm(quat_error(self.qt, q)) < self.rot_thresh
        isTimeout = self.timeout_count < 0
        if isTimeout or isSuccess:
            return PrimitiveStatus.SUCCESS
        return PrimitiveStatus.EXECUTING

    def plan(self):
        """Plan a straight path in task space"""
        super().plan()
        # plan from the previous command pose
        self.p0, self.q0 = self.robot_state.get_pose()

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

class Move2Contact(TaskPrimitive):
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
        # set gain
        self.set_controller_gain(kp, kd)
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
        super().configure(timeout=timeout)

    def step(self):
        # check status
        # done = self.is_terminate()
        status = self._status()

        # update time
        self.timeout_count -= self.dt
        if status is PrimitiveStatus.EXECUTING and not self.isContact:
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
        tau_cmd = self.controller.forward_ctrl(pd, qd, vd, fd)

        return tau_cmd, status

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

    def _status(self):
        f_proj = self._project_force()
        if f_proj < -self.thresh:
            self.isContact = True

        if self.isContact:
            self.n_step_settle_count -= 1

        isSettle = self.n_step_settle_count < 0
        isTimeout = self.timeout_count < 0
        if isSettle:
            return PrimitiveStatus.SUCCESS
        elif isTimeout:
            return PrimitiveStatus.FAIL
        else:
            return PrimitiveStatus.EXECUTING

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
