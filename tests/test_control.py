import numpy as np

from learn_seq.utils.mujoco import attach_viewer


def test_hold_position(sim, state, controller):
    viewer = attach_viewer(sim)
    p, q = state.get_pose()

    while sim.data.time < 1:
        # state.update()
        # tau_cmd = controller.forward_ctrl(p, q, np.zeros(6), np.zeros(6),
        #                                   np.ones((6, 6)))
        # state.set_control_torque(tau_cmd)
        # state.update_dynamic()
        controller.step(p, q, np.zeros(6), np.zeros(6), np.ones((6, 6)))
        print(state.get_jacobian())
        viewer.render()
