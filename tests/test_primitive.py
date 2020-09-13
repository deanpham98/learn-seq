import pytest
import numpy as np
import matplotlib.pyplot as plt
from learn_seq.primitive.task import Move2Target
from learn_seq.utils.mujoco import integrate_quat, attach_viewer

@pytest.fixture
def move2target(state, record):
    tf_pos = np.zeros(3)
    tf_quat = np.array([1., 0, 0,0 ])
    return Move2Target(state, record, tf_pos, tf_quat, timeout=4.)

# move to a random pose and plot response
def test_move_to_target(sim, state, record, move2target):
    viewer = attach_viewer(sim)

    p, q = state.get_pose()
    pt = p + np.random.random(3) * 0.05
    r = np.random.random(3)
    r = r / np.linalg.norm(r)
    qt = integrate_quat(q, r, 0.05)  # 0.1 rad
    qt = q

    s = 0.5
    ft = np.zeros(6)
    move2target.configure(pt, qt, ft, s)

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
    qt = integrate_quat(np.array([1., 0., 0, 0]), r, 0.05)  # 0.1 rad

    s = 0.2
    ft = np.zeros(6)
    move2target.configure(pt, qt, ft, s)

    record.start_record()
    move2target.run(viewer=viewer)
    record.stop_record()
    record.plot_error()
    record.plot_pos()
    record.plot_orient()
    plt.show()
