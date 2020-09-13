import pytest
from numpy import pi
import numpy as np
from learn_seq.utils.mujoco import load_model, attach_viewer, set_state
from learn_seq.controller.hybrid import HybridController, StateRecordHybridController
from learn_seq.controller.robot_state import RobotState

@pytest.fixture(scope="module")
def sim():
    sim = load_model("round_pih.xml")
    qpos = np.array([0, -pi/4, 0, -3 * pi/4, 0, pi/2, pi / 4, 0.015, 0.015])
    set_state(sim, qpos, np.zeros(9))
    return sim

@pytest.fixture(scope="module")
def state(sim):
    return RobotState(sim, "end_effector")

@pytest.fixture(scope="module")
def controller(state):
    controller = HybridController(state)
    return controller

@pytest.fixture(scope="module")
def record(state):
    controller = StateRecordHybridController(state)
    return controller
