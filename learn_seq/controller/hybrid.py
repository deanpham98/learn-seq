import numpy as np
from collections import namedtuple
from learn_seq.utils.mujoco import quat_error
from learn_seq.utils.general import StateRecorder, saturate_vector
from learn_seq.controller.base import TaskController

# using quaternion as the orientation representation
Pose = namedtuple("Pose", ["pos", "quat"])

class HybridController(TaskController):
    """Simulated hybrid force/position controller for the panda robot
    The force control law is $\tau_f = J^T * (I - S) * F_d$
    The position control law is $\tau_m = J^T * S * (K_p*e + K_d*\dot{e})

    :param type kp_init: Initial stiffness of the controller.
    :param type dtau_max: Upper limit of torque rate.
    """
    def __init__(self, robot_state, kp_init=None, dtau_max=2.):
        super().__init__(robot_state)
        self.prev_tau_cmd = np.zeros(7)
        self.dtau_max = dtau_max

        kp = kp_init or np.array([1000.]*3 + [60]*3)
        kd = 2*np.sqrt(kp)
        self.set_gain(kp, kd)

        # store the command pose for external computation
        self.pose_cmd = Pose(np.zeros(3), np.zeros(4))

        # controller_state
        # self.t = []
        self.controller_state = {
            "err": None,
            "p": None,
            "pd": None,
            "q": None,
            "qd": None,
            "fd": None,
        }

    def forward_ctrl(self, pd, qd, vd, fd, S):
        """See TaskController. All inputs are in base frame

        :param type pd: desired position.
        :param type qd: desired orientation.
        :param type vd: desired velocity.
        :param type fd: desired force.
        :param type S: desired selection matrix.
        """
        # get current position and orientation
        p, q = self.robot_state.get_pose()

        # compute position and orientation error
        ep = np.zeros(6)
        ep[:3] = pd - p
        ep[3:] = quat_error(q, qd)

        # jacobian
        jac = self.robot_state.get_jacobian()

        # joint velocity
        dq = self.robot_state.get_joint_velocity()

        # compute velocity error
        ep_dot = vd - jac.dot(dq[:7])

        # position control law
        f_pos = self.kp*ep + self.kd*ep_dot

        # force control law
        f_force = fd

        # gravity and coriolis torque
        tau_comp = self.robot_state.get_bias_torque()

        # null space torque
        tau_null = np.zeros(7)

        # general control law
        iS = np.identity(6) - S
        f_cmd = S.dot(f_pos) + iS.dot(f_force)
        tau_cmd = jac.T.dot(f_cmd) + tau_null
        # saturate torque rate
        tau_sat = saturate_vector(self.prev_tau_cmd, tau_cmd, self.dtau_max)
        self.prev_tau_cmd = tau_sat.copy()

        # update controller state
        self.controller_state["err"] = ep
        self.controller_state["p"] = p
        self.controller_state["q"] = q
        self.controller_state["pd"] = pd
        self.controller_state["qd"] = qd
        self.controller_state["fd"] = fd

        return tau_sat + tau_comp

    def set_gain(self, kp, kd):
        self.kp = kp
        self.kd = kd

    def get_controller_state(self):
        return self.controller_state
