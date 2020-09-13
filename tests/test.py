
import pytest
from numpy import pi
import numpy as np
from learn_seq.utils.mujoco import load_model, attach_viewer, set_state
from learn_seq.controller.hybrid import HybridController, StateRecordHybridController
from learn_seq.controller.robot_state import RobotState


sim = load_model("round_pih.xml")
qpos = np.array([0, -pi/4, 0, -3 * pi/4, 0, pi/2, pi / 4, 0.015, 0.015])
set_state(sim, qpos, np.zeros(9))
sim.step()

state = RobotState(sim, "peg")
controller = HybridController(state)
p, q = state.get_pose()

while sim.data.time < 1:
    # state.update()
    # tau_cmd = controller.forward_ctrl(p, q, np.zeros(6), np.zeros(6), np.ones((6, 6)))
    # state.set_control_torque(tau_cmd)
    # state.update_dynamic()
    controller.step(p, q, np.zeros(6), np.zeros(6), np.ones((6, 6)))
    print(state.get_jacobian())
    # viewer.render()
