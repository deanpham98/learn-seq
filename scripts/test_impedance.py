import os
import mujoco_py
import numpy as np
import matplotlib.pyplot as plt
from learn_seq.controller.impedance import StateRecordImpedanceController
from learn_seq.controller.robot_state import RobotState
from learn_seq.utils.general import get_mujoco_model_path
from learn_seq.primitive.impedance import Move2Target, Move2Contact
from learn_seq.utils.mujoco import set_state

def test_impedance():
    xml_model_name = "sliding.xml"
    mujoco_path = get_mujoco_model_path()
    model_path = os.path.join(mujoco_path, xml_model_name)
    model = mujoco_py.load_model_from_path(model_path)
    sim = mujoco_py.MjSim(model)
    robot_state = RobotState(sim, "end_effector")

    # controller
    controller = StateRecordImpedanceController(robot_state)
    q0 = np.array([0, -np.pi/4, 0, -3 * np.pi/4, 0, np.pi/2, np.pi / 4, 0.015, 0.015])
    set_state(sim, q0, np.zeros(sim.model.nv))

    # move to target primitive
    tf_pos = np.zeros(3)
    tf_quat = np.array([1, 0, 0, 0.])
    pr = Move2Target(robot_state, controller, tf_pos, tf_quat)

    pt = np.array([0.53, 0.062, 0.1888])
    qt = np.array([0, 1., 0, 0])
    s = 0.5
    kp = np.array([1000]*3 + [60.]*3)
    kd = 2*np.sqrt(kp)
    pr.configure(pt, qt, s, kp, kd, timeout=5)


    # move down to contact
    pr2 = Move2Contact(robot_state, controller, tf_pos, tf_quat)
    u = np.array([0, 0, -1, 0, 0, 0])
    s = 0.01
    fs = 5
    ft = np.array([0, 0, -5, 0, 0, 0])
    pr2.configure(u, s, fs, np.zeros(6), kp, kd, 5)

    # sliding
    pr3 = Move2Contact(robot_state, controller, tf_pos, tf_quat)
    u = np.array([0, -1, 0, 0, 0, 0])
    s = 0.01
    fs = 100
    ft = np.array([0, 0, -5, 0, 0, 0])
    pr3.configure(u, s, fs, ft, kp, kd, 10)

    # run
    viewer = mujoco_py.MjViewer(sim)
    pr.run(viewer=None)
    pr2.run(viewer=None)

    controller.start_record()
    pr3.run(viewer=viewer)
    controller.stop_record()
    controller.plot_key(["f", "fd"])
    controller.plot_key(["kp"])

    plt.show()

if __name__ == '__main__':
    test_impedance()
