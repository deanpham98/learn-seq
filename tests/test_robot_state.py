import pytest
from numpy import pi
import numpy as np
from learn_seq.controller.robot_state import RobotState
from learn_seq.utils.mujoco import load_model, attach_viewer, set_state

def test_mj_step(state):
    state.update()
    pose1 = state.get_pose()
    state.update_dynamic()
    pose2 = state.get_pose()
    print("pose after mj_step1: {}".format(pose1))
    print("pose after mj_step2: {}".format(pose2))

    assert False


def test_jac(sim, state):
    viewer = attach_viewer(sim)
    state.update()
    jac1 = state.get_jacobian()
    state.update_dynamic()
    jac2 = state.get_jacobian()

    while sim.data.time < 1:
        viewer.render()
        sim.step()

    print("pose after mj_step1: {}".format(jac1))
    print("pose after mj_step2: {}".format(jac2))

    assert False
