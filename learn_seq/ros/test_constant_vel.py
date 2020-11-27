import numpy as np
from ros_interface import RosInterface
import transforms3d.quaternions as Q

class TestConstantVelocity:
    def __init__(self):
        self.ros_interface =RosInterface()

    def test_null_cmd(self):
        cmd = self.ros_interface.get_constant_velocity_cmd()
        self.ros_interface.run_primitive(cmd)

    def test_translation(self):
        cmd = self.ros_interface.get_constant_velocity_cmd()
        cmd.constant_velocity_param.speed_factor = 0.03
        cmd.constant_velocity_param.direction = np.array([0, 0., 1., 0, 0, 0])

        # move for 1 sec
        cmd.constant_velocity_param.timeout = 2.5
        cmd.constant_velocity_param.f_thresh = 100.
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
        target_pose = np.array([[0.980334, 0.197291, -0.00146722, 0.530611],
                    [0.19729, -0.980335, -0.000872652, -0.0848445],
                    [-0.00161053, 0.000566023, -0.999999, 0.15113],
                    [0 , 0, 0, 1]])

        # NOTE in base frame
        pos = target_pose[:3, 3]
        pos[2]+=0.01
        pos[1]+=0.015
        pos[0]+=0.015
        quat = Q.mat2quat(target_pose[:3, :3])
        # move to appropriate pose
        self.ros_interface.move_to_pose(pos, quat, 0.1)

        cmd = self.ros_interface.get_constant_velocity_cmd()
        cmd.constant_velocity_param.speed_factor = 0.03
        cmd.constant_velocity_param.direction = np.array([0, 0, 0., 0, -1, 0])

        # move for 1 sec
        cmd.constant_velocity_param.timeout = 10.
        cmd.constant_velocity_param.f_thresh = 5.
        self.ros_interface.run_primitive(cmd)

    # to be call after test_move_to_contact
    def test_sliding(self):
        cmd = self.ros_interface.get_constant_velocity_cmd()
        cmd.constant_velocity_param.speed_factor = 0.01
        cmd.constant_velocity_param.direction = np.array([0., 1, 0,0 ,0 ,0])

        cmd.constant_velocity_param.f_thresh = 5.
        cmd.constant_velocity_param.fd = np.array([0, 0, -10, 0., 0, 0])
        cmd.constant_velocity_param.timeout = 2.
        self.ros_interface.run_primitive(cmd)

    def test_force_control(self):
        cmd = self.ros_interface.get_constant_velocity_cmd()
        cmd.constant_velocity_param.speed_factor = 0.0
        cmd.constant_velocity_param.direction = np.array([0., 1, 0,0 ,0 ,0])

        cmd.constant_velocity_param.f_thresh = 10.
        cmd.constant_velocity_param.fd = np.array([0, 0, -10, 0., 0, 0])
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
    test.test_translation()
    # test.test_rotation()
    # test.test_move_to_contact()
    # test.test_sliding()
    # test.test_task_frame()
    # test.test_force_control()
