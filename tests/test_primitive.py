import matplotlib.pyplot as plt
import numpy as np
import pytest

from learn_seq.primitive.hybrid import (AdmittanceMotion, Displacement,
                                        Move2Contact, Move2Target)
from learn_seq.utils.mujoco import attach_viewer, integrate_quat


@pytest.fixture
def move2target(state, record):
    tf_pos = np.zeros(3)
    tf_quat = np.array([1., 0, 0, 0])
    return Move2Target(state, record, tf_pos, tf_quat, timeout=4.)


@pytest.fixture
def move2contact(state, record):
    tf_pos = np.zeros(3)
    tf_quat = np.array([1., 0, 0, 0])
    return Move2Contact(state, record, tf_pos, tf_quat, timeout=4.)


@pytest.fixture
def displacement(state, record):
    tf_pos = np.zeros(3)
    tf_quat = np.array([1., 0, 0, 0])
    return Displacement(state, record, tf_pos, tf_quat, timeout=4.)


@pytest.fixture
def admittance(state, record):
    tf_pos = np.zeros(3)
    tf_quat = np.array([1., 0, 0, 0])
    return AdmittanceMotion(state, record, tf_pos, tf_quat, timeout=4.)


# move to a random pose and plot response
def test_move_to_target(sim, state, record, move2target):
    viewer = attach_viewer(sim)

    p, q = state.get_pose()
    pt = p + np.random.random(3) * 0.05
    r = np.random.random(3)
    r = r / np.linalg.norm(r)
    kp = np.array([1000] * 3 + [60] * 3)
    kd = 2 * np.sqrt(kp)
    qt = integrate_quat(q, r, 0.05)  # 0.1 rad
    qt = q

    s = 0.5
    ft = np.zeros(6)
    move2target.configure(pt, qt, ft, s, kp, kd)

    record.start_record()
    move2target.run(viewer=viewer)
    record.stop_record()
    record.plot_error()
    record.plot_pos()
    record.plot_orient()
    plt.show()


# test wiht different task frame
def test_task_frame(sim, state, record, move2target):
    viewer = attach_viewer(sim)

    p, q = state.get_pose()
    move2target.set_task_frame(p, q)
    pt = np.random.random(3) * 0.05
    r = np.random.random(3)
    r = r / np.linalg.norm(r)
    kp = np.array([1000] * 3 + [60] * 3)
    kd = 2 * np.sqrt(kp)
    qt = integrate_quat(np.array([1., 0., 0, 0]), r, 0.05)  # 0.1 rad

    s = 0.2
    ft = np.zeros(6)
    move2target.configure(pt, qt, ft, s, kp, kd)

    record.start_record()
    move2target.run(viewer=viewer)
    record.stop_record()
    record.plot_error()
    record.plot_pos()
    record.plot_orient()
    plt.show()


def test_move_to_contact(sim, state, record, move2target, move2contact):
    viewer = attach_viewer(sim)
    # move  to above the hole
    pt = np.array([0.53, 0.012, 0.2088])
    qt = state.get_pose()[1]
    ft = np.zeros(6)
    s = 0.5
    kp = np.array([1000] * 3 + [60] * 3)
    kd = 2 * np.sqrt(kp)
    move2target.configure(pt, qt, ft, s, kp, kd)

    # move down
    u = np.array([0, 0, -0, 0, 1, 0.])
    s = 0.1
    fs = 10.
    ft = np.zeros(6)
    move2contact.configure(u, s, fs, ft, kp, kd)

    #
    move2target.run(viewer=viewer)
    #
    record.start_record()
    move2contact.run(viewer=viewer)
    record.stop_record()
    record.plot_error()
    record.plot_pos()
    record.plot_orient()
    plt.show()
    assert False


def test_displacement(sim, state, record, displacement):
    viewer = attach_viewer(sim)
    # move down
    u = np.array([0, 0, 0, 0, 1, 0.])
    s = 0.1
    fs = 10.
    ft = np.zeros(6)
    dp = 0.1
    displacement.configure(u, s, fs, ft, dp)

    record.start_record()
    displacement.run(viewer=viewer)
    record.stop_record()
    record.plot_error()
    record.plot_pos()
    record.plot_orient()
    plt.show()


def test_admittance_motion(sim, state, record, admittance):
    viewer = attach_viewer(sim)
    p, q = state.get_pose()
    # move down
    kd = np.array([0.] * 3 + [0.12] * 3)
    ft = np.array([0, 0, -5, 0, 0, 0.])

    depth_thresh = p[2] + 0.1
    admittance.configure(kd, ft, depth_thresh)

    record.start_record()
    admittance.run(viewer=viewer)
    record.stop_record()
    record.plot_error()
    record.plot_pos()
    record.plot_orient()
    plt.show()
