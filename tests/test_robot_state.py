import pytest
from numpy import pi
import numpy as np
from learn_seq.controller.robot_state import RobotState
from learn_seq.utils.mujoco import load_model, attach_viewer, set_state

@pytest.fixture
def sim():
    sim = load_model("round_pih.xml")
    qpos = np.array([0, -pi/4, 0, -3 * pi/4, 0, pi/2, pi / 4, 0.015, 0.015])
    set_state(sim, qpos, np.zeros(9))
    return sim

@pytest.fixture
def state(sim):
    return RobotState(sim, "peg")

def test_mj_step(state):
    state.update()
    pose1 = state.get_pose()
    state.update_external_force()
    pose2 = state.get_pose()
    state.update_dynamic()
    pose3 = state.get_pose()
    print("pose after mj_step1: {}".format(pose1))
    print("pose after mj_rnePostConstraint: {}".format(pose2))
    print("pose after mj_step2: {}".format(pose3))

    assert False


def test_jac(state):
    viewer = attach_viewer(sim)
    state.update()
    jac1 = state.get_jacobian()
    state.update_external_force()
    jac2 = state.get_jacobian()
    state.update_dynamic()
    jac3 = state.get_jacobian()

    while sim.data.time < 1:
        viewer.render()
        sim.step()

    print("pose after mj_step1: {}".format(jac1))
    print("pose after mj_rnePostConstraint: {}".format(jac2))
    print("pose after mj_step2: {}".format(jac3))

    assert False
