import numpy as np
from mujoco_py import functions
from learn_seq.utils.mujoco import MJ_SITE_OBJ, MJ_BODY_OBJ, MJ_GEOM_OBJ
from learn_seq.utils.mujoco import quat2mat, pose_transform

class RobotState:
    """Wrapper to the mujoco sim to store Franka state and perform
    simulation operations (step, forward dynamic, ...).

    :param mujoco_py.MjSim sim:
    :param str ee_site_name: name of the end-effector site in mujoco xml model.
    :attr mujoco_py.MjData data: Description of parameter `data`.
    :attr mujoco_py.MjModel model: Description of parameter `model`.
    """
    def __init__(self, sim, ee_site_name):
        self.data = sim.data
        self.model = sim.model
        self.ee_site_idx = functions.mj_name2id(self.model, MJ_SITE_OBJ, ee_site_name)

    def update(self):
        """Update the internal simulation state (kinematics, external force, ...).
        Should be called before perform any setters or getters"""
        # update position-dependent state (kinematics, jacobian, ...)
        functions.mj_step1(self.model, self.data)
        # udpate the external force internally
        functions.mj_rnePostConstraint(self.model, self.data)

    def update_dynamic(self):
        """Update dynamic state (forward dynamic). The control torque should be
        set between self.update() and self.update_dynamic()"""
        functions.mj_step2(self.model, self.data)

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

        if frame_pos is None and frame_quat is None:
            return p, q
        else:
            frame_pos = frame_pos or np.zeros(3)
            frame_quat = frame_quat or np.array([1., 0, 0,0 ])
            pf, qf = pose_transform(p, q, frame_pos, frame_quat)
            return pf, qf

    def get_jacobian(self):
        """Get 6x7 geometric jacobian matrix."""
        dtype = self.data.qpos.dtype
        jac = np.zeros((6, self.model.nq), dtype=dtype)
        jac_pos = np.zeros((3*self.model.nq), dtype=dtype)
        jac_rot = np.zeros((3*self.model.nq), dtype=dtype)
        functions.mj_jacSite(
          self.model, self.data,
          jac_pos, jac_rot, self.ee_site_idx)
        # print(jac_pos, jac_rot)
        jac[3:] = jac_rot.reshape((3,self.model.nq))
        jac[:3] = jac_pos.reshape((3,self.model.nq))
        # only return first 7 dofs
        return jac[:, :7].copy()

    def get_joint_velocity(self):
        return self.data.qvel.copy()

    def get_bias_torque(self):
        """Get the gravity and Coriolis, centrifugal torque """
        return self.data.qfrc_bias[:7].copy()

    def get_timestep(self):
        """Timestep of the simulator is timestep of controller."""
        return self.model.opt.timestep

    def set_control_torque(self, tau):
        """Set control torque to the mujoco simulator."""
        assert tau.shape[0]==7
        self.data.ctrl[:] = np.hstack((tau, [0, 0]))
