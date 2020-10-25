import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from learn_seq.utils.mujoco import quat_error, quat2vec
from learn_seq.utils.general import saturate_vector
from learn_seq.controller.base import TaskController

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

        if kp_init is None:
            kp = np.array([1000.]*3 + [60]*3)
        else:
            kp = kp_init
        kd = 2*np.sqrt(kp)
        self.set_gain(kp, kd)

        # store the command pose for external computation
        self.reset_pose_cmd()
        self.u_pos = np.zeros(3)
        self.S = np.eye(6)

        p, q = self.robot_state.get_pose()
        # controller_state
        self.controller_state = {
            "t": self.robot_state.get_sim_time(),
            "err": np.zeros(6),
            "p": p,
            "pd": p,
            "q": q,
            "qd": q,
            "f": np.zeros(6),
            "fd": np.zeros(6),
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

        # update pose cmd
        self.p_cmd = iS[:3, :3].dot(p) + S[:3, :3].dot(pd)
        self.q_cmd = qd
        self.S = S

        # update controller state
        self.controller_state["t"] = self.robot_state.get_sim_time()
        self.controller_state["err"] = ep
        self.controller_state["p"] = p
        self.controller_state["q"] = q
        self.controller_state["pd"] = pd
        self.controller_state["qd"] = qd
        self.controller_state["fd"] = fd
        self.controller_state["f"] = self.robot_state.get_ee_force()

        return tau_sat + tau_comp

    def reset_pose_cmd(self):
        p, q = self.robot_state.get_pose()
        self.p_cmd = p
        self.q_cmd = q

    def reset_tau_cmd(self):
        self.prev_tau_cmd = np.zeros(7)

    def set_gain(self, kp, kd):
        self.kp = kp
        self.kd = kd

    def get_controller_state(self):
        return self.controller_state

    def get_pose_cmd(self):
        return self.p_cmd.copy(), self.q_cmd.copy()

    def get_pose_control_cmd(self):
        return self.S[:3, :3].dot(self.p_cmd), self.q_cmd.copy()

class StateRecordHybridController(HybridController):
    """Record state, useful to visualize response, trajectory."""
    def __init__(self, robot_state, kp_init=None, dtau_max=2.):
        super().__init__(robot_state, kp_init, dtau_max)
        self.record = False
        self._reset_state()

    def _reset_state(self):
        self.state_dict = {}
        for key in self.controller_state.keys():
            self.state_dict[key] = []

    def start_record(self):
        self._reset_state()
        self.record = True

    def stop_record(self):
        self.record = False

    def forward_ctrl(self, *argv, **kwargs):
        res =  super().forward_ctrl(*argv, **kwargs)
        if self.record:
            for key in self.state_dict.keys():
                self.state_dict[key].append(self.controller_state[key])
        return res

    def plot_key(self, key):
        """Plot data defined by a key list.

        :param type key: Description of parameter `key`.
        :return: Description of returned object.
        :rtype: type

        """
        N = len(self.state_dict[key[0]])    # no. samples
        t_record = len(self.state_dict[key[0]]) * self.dt
        fig, ax = plt.subplots(3, 2, sharex=True)
        for k in key:
            data = np.array(self.state_dict[k])
            for i in range(3):
                ax[i, 0].plot(np.linspace(0, t_record, N), data[:, i])
                if data.shape[1] == 6:
                    ax[i, 1].plot(np.linspace(0, t_record, N), data[:, i+3])

        for i in range(3):
            ax[i, 0].legend(key)
            ax[i, 0].set_xlabel("Simulation time (s)")
            ax[i, 1].legend(key)
            ax[i, 1].set_xlabel("Simulation time (s)")

        return fig, ax

    def plot_error(self):
        return self.plot_key(["err",])

    def plot_pos(self):
        return self.plot_key(["p", "pd"])

    def plot_orient(self):
        # quat to rotation vector
        for i in range(len(self.state_dict["q"])):
            self.state_dict["q"][i] = quat2vec(self.state_dict["q"][i])
            self.state_dict["qd"][i] = quat2vec(self.state_dict["qd"][i])
        return self.plot_key(["q", "qd"])
