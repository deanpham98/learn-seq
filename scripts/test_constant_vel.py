import numpy as np
from learn_seq.ros.ros_interface import FrankaRosInterface
import transforms3d.quaternions as Q
import matplotlib.pyplot as plt

class TestConstantVelocity:
    def __init__(self):
        self.ros_interface = FrankaRosInterface()

    def test_null_cmd(self):
        cmd = self.ros_interface.get_constant_velocity_cmd()
        self.ros_interface.run_primitive(cmd)

    def test_translation(self):
        cmd = self.ros_interface.get_constant_velocity_cmd()
        cmd.constant_velocity_param.speed_factor = 0.02
        cmd.constant_velocity_param.direction = np.array([0, -1, 0, 0, 0, 0])

        # move for 1 sec
        cmd.constant_velocity_param.timeout = 4.
        cmd.constant_velocity_param.f_thresh = 4.
        self.ros_interface.run_primitive(cmd)

    def test_rotation(self):
        cmd = self.ros_interface.get_constant_velocity_cmd()
        cmd.constant_velocity_param.speed_factor = 0.05
        cmd.constant_velocity_param.direction = np.array([0, 0, 0., 1., -1., 0])

        # move for 1 sec
        cmd.constant_velocity_param.timeout = 3
        cmd.constant_velocity_param.f_thresh = 100.
        self.ros_interface.run_primitive(cmd)

    def test_move_to_contact(self):
        target_pose = np.array([[ 1,  0,  0,  0.558861 ],
                                [ 0, -1,  0,  0.127538  ],
                                [ 0, 0, -1 ,  0.23938 ],
                                [ 0, 0.,  0., 1.       ]])

        # NOTE in base frame
        pos = target_pose[:3, 3]
        pos[2]+=0.01
        # pos[1]+=0.01
        # pos[0]+=0.01
        quat = Q.mat2quat(target_pose[:3, :3])
        # move to appropriate pose
        self.ros_interface.move_to_pose(pos, quat, 0.2)

        cmd = self.ros_interface.get_constant_velocity_cmd()
        cmd.constant_velocity_param.speed_factor = 0.02
        cmd.constant_velocity_param.direction = np.array([0, 0, -1., 0, 0, 0])

        # move for 1 sec
        cmd.constant_velocity_param.timeout = 15.
        cmd.constant_velocity_param.f_thresh = 4.
        self.ros_interface.run_primitive(cmd)

    # to be call after test_move_to_contact
    def test_sliding(self):
        self.ros_interface.start_record()
        cmd = self.ros_interface.get_constant_velocity_cmd()
        cmd.constant_velocity_param.speed_factor = 0.02
        cmd.constant_velocity_param.direction = np.array([0., -1, 0, 0 ,0 ,0])

        cmd.constant_velocity_param.f_thresh = 5.
        cmd.constant_velocity_param.fd = np.array([0, 0, -5, 0., 0, 0])
        cmd.constant_velocity_param.timeout = 20.
        self.ros_interface.run_primitive(cmd)
        self.ros_interface.stop_record("test.npy")
        self.ros_interface.plot_pose()
        self.ros_interface.plot_force()
        plt.show()

    def test_force_control(self):
        cmd = self.ros_interface.get_constant_velocity_cmd()
        cmd.constant_velocity_param.speed_factor = 0.
        cmd.constant_velocity_param.direction = np.array([0., 1, 0, 0 ,0 ,0])

        cmd.constant_velocity_param.f_thresh = 5.
        cmd.constant_velocity_param.fd = np.array([0, 0, -8., 0., 0, 0])
        cmd.constant_velocity_param.timeout = 5.
        self.ros_interface.run_primitive(cmd)


    def test_task_frame(self):
        cmd = self.ros_interface.get_constant_velocity_cmd()
        cmd.constant_velocity_param.task_frame.pos =np.array([0.5, 0.5, 0.3])
        cmd.constant_velocity_param.task_frame.quat =np.array([0.0, 1., 0, 0])

        cmd.constant_velocity_param.speed_factor = 0.05
        cmd.constant_velocity_param.direction = np.array([1, 0, 1., 0, 0, 0])

        # move for 1 sec
        cmd.constant_velocity_param.timeout = 1.
        cmd.constant_velocity_param.f_thresh = 5.
        self.ros_interface.run_primitive(cmd)

if __name__ == '__main__':
    test = TestConstantVelocity()
    # test.test_null_cmd()
    # test.test_translation()
    # test.test_rotation()
    test.test_move_to_contact()
    test.test_sliding()
    # test.test_task_frame()
    # test.test_force_control()
