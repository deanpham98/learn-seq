import os
import numpy as np
import mujoco_py
from mujoco_py import functions
from learn_seq.utils.general import get_mujoco_model_path

# object indicator in mujoco
MJ_SITE_OBJ = 6     # `site` objec
MJ_BODY_OBJ = 1     # `body` object
MJ_GEOM_OBJ = 5     # `geom` object

def load_model(xml_name="round_pih.xml"):
    """Load a model from `mujoco/franka_pih`

    :param type xml_name: Description of parameter `xml_name`.
    :param type primitive: Description of parameter `primitive`.
    :return: Description of returned object.
    :rtype: type

    """
    model_path = get_mujoco_model_path()
    xml_path = os.path.join(model_path, xml_name)
    model = mujoco_py.load_model_from_path(xml_path)
    sim = mujoco_py.MjSim(model)

    return sim

def attach_viewer(sim):
    return mujoco_py.MjViewer(sim)

def set_state(sim, qpos, qvel):
    assert qpos.shape == (sim.model.nq,) and qvel.shape == (sim.model.nv,)
    old_state = sim.get_state()
    new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                     old_state.act, old_state.udd_state)
    sim.set_state(new_state)
    sim.forward()


def get_contact_force(mj_model, mj_data, body_name, frame_pos, frame_quat):
    """Get the force acting on a body, with respect to a frame.
    Note that mj_rnePostConstraint should be called before this function
    to update the simulator state.

    :param str body_name: Body name in mujoco xml model.
    :return: force:torque format.
    :rtype: np.array(6)

    """
    bodyId = mujoco_py.functions.mj_name2id(mj_model, MJ_BODY_OBJ, body_name)
    force_com = mj_data.cfrc_ext[bodyId, :]
    # contact force frame
    # orientation is aligned with world frame
    qf = np.array([1, 0, 0, 0.])
    # position of origin in the world frame
    body_rootid = mj_model.body_rootid[bodyId]
    pf = mj_data.subtree_com[body_rootid, :]

    # inverse com frame
    pf_inv, qf_inv = np.zeros(3), np.zeros(4)
    functions.mju_negPose(pf_inv, qf_inv, pf, qf)
    # T^com_target
    p_ct, q_ct=  np.zeros(3), np.zeros(4)
    functions.mju_mulPose(p_ct, q_ct, pf_inv, qf_inv, frame_pos, frame_quat)
    # q_ct -> mat
    mat_ct = np.zeros(9)
    functions.mju_quat2Mat(mat_ct, q_ct)

    # transform to desired frame
    trn_force = force_com.copy()
    functions.mju_transformSpatial(trn_force, force_com, 1,
                        p_ct, np.zeros(3), mat_ct)

    # reverse order to get force:torque format
    return np.concatenate((trn_force[3:], trn_force[:3]))

def get_geom_pose(model, geom_name):
    """Return the geom pose (relative to parent body).

    :param mujoco_py.MjModel model:
    :param str geom_name:
    :return: position, quaternion
    :rtype: tuple(np.array(3), np.array(4))
    """
    geom_id = functions.mj_name2id(model, MJ_GEOM_OBJ, geom_name)
    pos = model.geom_pos[geom_id, :]
    quat = model.geom_quat[geom_id, :]
    return pos, quat

def get_geom_size(model, geom_name):
    """Return the geom size.

    :param mujoco_py.MjModel model:
    :param str geom_name:
    :return: (radius, half-length, _) for cylinder geom, and
             (X half-size; Y half-size; Z half-size) for box geom
    :rtype: np.array(3)
    """
    geom_id = functions.mj_name2id(model, MJ_GEOM_OBJ, geom_name)
    return model.geom_size[geom_id, :]

# -------- GEOMETRY TOOLs
def quat_error(q1, q2):
    """Compute the rotation vector (expressed in the base frame), that if follow
        in a unit time, will transform a body with orientation `q1` to
        orientation `q2`

    :param list/np.ndarray q1: Description of parameter `q1`.
    :param list/np.ndarray q2: Description of parameter `q2`.
    :return: a 3D rotation vector
    :rtype: np.ndarray

    """
    if isinstance(q1, list):
        q1 = np.array(q1)

    if isinstance(q2, list):
        q2 = np.array(q2)

    dtype=q1.dtype
    neg_q1 = np.zeros(4, dtype=dtype)
    err_rot_quat = np.zeros(4, dtype=dtype)
    err_rot = np.zeros(3, dtype=dtype)

    if q1.dot(q2) < 0:
        q1 = -q1

    functions.mju_negQuat(neg_q1, q1)
    functions.mju_mulQuat(err_rot_quat, q2, neg_q1)
    functions.mju_quat2Vel(err_rot, err_rot_quat, 1)
    return err_rot

def quat2mat(q):
    """Tranform a quaternion to rotation amtrix.

    :param type q: Description of parameter `q`.
    :return: 3x3 rotation matrix
    :rtype: np.array
    """
    mat = np.zeros(9)
    functions.mju_quat2Mat(mat, q)
    return mat.reshape((3, 3))

def pose_transform(p1, q1, p21, q21):
    """Coordinate transformation between 2 frames

    :param np.ndarray p1: position in frame 1
    :param np.ndarray q1: orientation (quaternion) in frame 1
    :param np.ndarray p21: relative position between frame 1 and 2
    :param np.ndarray q21: relative orientation between frame 1 and 2
    :return: position and orientation in frame 2
    :rtype: type

    """
    # quat to rotation matrix
    R21 = quat2mat(q21)

    p2 = p21 + R21.dot(p1)
    q2 = np.zeros_like(q1)
    functions.mju_mulQuat(q2, q21, q1) #q2 = q21*q1
    return p2, q2

def integrate_quat(q, r, dt):
    """Integrate quaternion by a fixed angular velocity over the duration dt.

    :param np.array(4) q: quaternion.
    :param np.array(3) r: angular velocity.
    :param float dt: duration.
    :return: result quaternion.
    :rtype: np.array(4)
    """
    qres = np.zeros(4)
    qe = np.zeros(4)
    r = r*dt
    angle = np.linalg.norm(r)
    if angle < 1e-9:
        # if angle too small then return current q
        return q.copy()
    axis = r/angle
    functions.mju_axisAngle2Quat(qe, axis, angle)
    functions.mju_mulQuat(qres, qe, q)
    return qres

def transform_spatial(v1, q21):
    """Coordinate transformation of a spatial vector. The spatial vector can be either
    twist (linear + angular velocity) or wrench (force + torque)

    :param type v1: Spatial vector in frame 1
    :param type q21: transformation matrix (in terms of quaternion)
    :return: Description of returned object.
    :rtype: type
    """
    R21 = quat2mat(q21)
    R = np.block([[R21, np.zeros((3, 3))], [np.zeros((3, 3)), R21]])
    return R.dot(v1)

def similarity_transform(A1, q21):
    """Similarity transformation of a matrix from frame 1 to frame 2
            A2 = R21 * A1 * R12

    :param np.array((3, 3)) A1: 3x3 matrix.
    :param np.array(4) q21: quaternion representation.
    :return: 3x3 matrix
    :rtype: np.array

    """
    R21 = quat2mat(q21)
    return R21.dot(A1.dot(R21.T))

def quat2vec(q):
    """Transform quaternion representation to rotation vector representation"""
    r = np.zeros(3)
    scale = 1
    mujoco_py.functions.mju_quat2Vel(r, q, scale)
    if r[0] < 0:
        angle = np.linalg.norm(r)
        r = r / angle
        if angle < 0:
            angle = -angle
        else:
            angle = 2*np.pi - angle
        r = -r*angle
    return r

def inverse_frame(p, q):
    pi, qi = np.zeros(3), np.zeros(4)
    functions.mju_negPose(pi, qi, p, q)
    return pi, qi
