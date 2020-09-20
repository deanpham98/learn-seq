import pytest
from numpy import pi
import numpy as np
import gym
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

@pytest.fixture(scope="module")
def env(state):
    peg_pos_range = [[-0.05]*3, [0.05]*3]
    peg_rot_range = [[np.pi - 0.2] + [-0.2]*2, [np.pi + 0.2] + [0.2]*2]
    initial_pos_range = [[-0.005]*3, [0.005]*3]
    initial_rot_range = [[-np.pi/180]*3, [np.pi/180]*3]
    primitive_list = []
    type = "move2contact"
    param = dict(u=np.array([0., 0, -1, 0, 0, 0]),
                 s=0.1, fs=5,
                 ft=np.array([0.]*6),
                 kp=np.array([1000]*3+[60.]*3),
                 kd=2*np.sqrt(np.array([1000]*3+[60.]*3)),
                 timeout=3.)
    primitive_list.append((type, param))
    controller_kwargs=dict(kp_init=np.array([1500]*3+[60.]*3))
    return gym.make(id="learn_seq:MujocoInsertionEnv-v0",
                    xml_model_name="round_pih.xml",
                    robot_state=state,
                    primitive_list=primitive_list,
                    peg_pos_range=peg_pos_range,
                    peg_rot_range=peg_rot_range,
                    initial_pos_range=initial_pos_range,
                    initial_rot_range=initial_rot_range,
                    controller_class=StateRecordHybridController,
                    **controller_kwargs
                    )
