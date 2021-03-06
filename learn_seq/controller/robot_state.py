import numpy as np
from mujoco_py import functions

from learn_seq.utils.filter import ButterLowPass
from learn_seq.utils.mujoco import (MJ_SITE_OBJ, get_contact_force,
                                    inverse_frame, pose_transform,
                                    transform_spatial)


class RobotState:
    """Wrapper to the mujoco sim to store robot state and perform
    simulation operations (step, forward dynamic, ...).

    :param mujoco_py.MjSim sim: mujoco sim
    :param str ee_site_name: name of the end-effector site in mujoco xml model.
    :attr mujoco_py.MjData data: Description of parameter `data`.
    :attr mujoco_py.MjModel model: Description of parameter `model`.
    """

    def __init__(self, sim, ee_site_name):
        self.data = sim.data
        self.model = sim.model
        self.ee_site_idx = functions.mj_name2id(
            self.model, MJ_SITE_OBJ, ee_site_name)
        self.isUpdated = False
        # low pass filter
        dt = sim.model.opt.timestep
        fs = 1 / dt
        cutoff = 30
        self.fe = np.zeros(6)
        self.lp_filter = ButterLowPass(cutoff, fs, order=5)

    def update(self):
        """Update the internal simulation state (kinematics, external force, ...).
        Should be called before perform any setters or getters"""
        # update position-dependent state (kinematics, jacobian, ...)
        functions.mj_step1(self.model, self.data)
        # udpate the external force internally
        functions.mj_rnePostConstraint(self.model, self.data)
        # filter ee force
        self.update_ee_force()
        self.isUpdated = True

    def update_dynamic(self):
        """Update dynamic state (forward dynamic). The control torque should be
        set between self.update() and self.update_dynamic()"""
        functions.mj_step2(self.model, self.data)
        self.isUpdated = False

    def is_update(self):
        return self.isUpdated

    def update_ee_force(self):
        """calculate and filter the external end-effector force
        """
        p, q = self.get_pose()
        fe = get_contact_force(self.model, self.data, "peg", p, q)
        self.fe = self.lp_filter(fe.reshape((-1, 6)))[0, :]

    def reset_filter_state(self):
        self.lp_filter.reset_state()

    def get_pose(self, frame_pos=None, frame_quat=None):
        """Get current pose of the end-effector with respect to a particular frame

        :param np.array(3) frame_pos: if None then frame origin is coincide with
                                      the base frame
        :param np.array(4) frame_quat: if None then frame axis coincide with the
                                      base frame
        :return: position, quaternion
        :rtype: tuple(np.array(3), np.array(4))

        """
        p = self.data.site_xpos[self.ee_site_idx].copy()    # pos
        R = self.data.site_xmat[self.ee_site_idx]           # rotation matrix

        q = np.zeros(4)
        functions.mju_mat2Quat(q, R)

        if frame_pos is None:
            frame_pos = np.zeros(3)
        if frame_quat is None:
            frame_quat = np.array([1., 0, 0, 0])
        # inverse frame T_0t -> T_t0
        inv_pos, inv_quat = inverse_frame(frame_pos, frame_quat)
        pf, qf = pose_transform(p, q, inv_pos, inv_quat)
        return pf, qf

    def get_ee_force(self, frame_quat=None):
        """Get current force torque acting on the end-effector,
        with respect to a particular frame

        :param np.array(3) frame_pos: if None then frame origin is coincide with
                                      the ee frame
        :param np.array(4) frame_quat: if None then frame axis coincide with the
                                      ee frame
        :return: force:torque format
        :rtype: np.array(6)

        """
        # force acting on the ee, relative to the ee frame
        if frame_quat is None:
            return self.fe
        p, q = self.get_pose()
        qf0 = np.zeros(4)
        functions.mju_negQuat(qf0, frame_quat)
        qfe = np.zeros(4)
        functions.mju_mulQuat(qfe, qf0, q)  # qfe = qf0 * q0e

        # transform to target frame
        ff = transform_spatial(self.fe, qfe)
        # lowpass filter
        return ff

    def get_jacobian(self):
        """Get 6x7 geometric jacobian matrix."""
        dtype = self.data.qpos.dtype
        jac = np.zeros((6, self.model.nq), dtype=dtype)
        jac_pos = np.zeros((3 * self.model.nq), dtype=dtype)
        jac_rot = np.zeros((3 * self.model.nq), dtype=dtype)
        functions.mj_jacSite(
            self.model, self.data,
            jac_pos, jac_rot, self.ee_site_idx)
        jac[3:] = jac_rot.reshape((3, self.model.nq))
        jac[:3] = jac_pos.reshape((3, self.model.nq))
        # only return first 7 dofs
        return jac[:, :7].copy()

    def get_ee_velocity(self):
        """Get ee velocity w.s.t base frame"""
        dq = self.get_joint_velocity()
        jac = self.get_jacobian()
        return jac.dot(dq[:7])

    def get_joint_velocity(self):
        return self.data.qvel.copy()

    def get_bias_torque(self):
        """Get the gravity and Coriolis, centrifugal torque """
        return self.data.qfrc_bias[:7].copy()

    def get_timestep(self):
        """Timestep of the simulator is timestep of controller."""
        return self.model.opt.timestep

    def get_sim_time(self):
        return self.data.time

    def set_control_torque(self, tau):
        """Set control torque to robot actuators."""
        assert tau.shape[0] == 7
        self.data.ctrl[:] = np.hstack((tau, [0, 0]))
