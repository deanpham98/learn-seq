import transforms3d.quaternions as Q
import numpy as np
from learn_seq.ros.ros_interface import FrankaRosInterface

class TestMoveToPose:
    def __init__(self):
        self.ros_interface = FrankaRosInterface()

    def test_null_cmd(self):
        cmd = self.ros_interface.get_move_to_pose_cmd()
        result, time_run = self.ros_interface.run_primitive(cmd)
        print("result: {}, time run {}".format(result, time_run))

    def test_position_control(self):
        cmd = self.ros_interface.get_move_to_pose_cmd()
        cmd.move_to_pose_param.target_pose.pos[2] += 0.03
        cmd.move_to_pose_param.speed_factor = 0.03
        result, time_run = self.ros_interface.run_primitive(cmd)
        print("result: {}, time run {}".format(result, time_run))

    def test_orientation_control(self):
        from transforms3d import quaternions as Q

        cmd = self.ros_interface.get_move_to_pose_cmd()
        axis = np.array([1., 0, 0])
        angle = -np.pi/2
        qe = Q.axangle2quat(axis, angle, True)
        q = cmd.move_to_pose_param.target_pose.quat.copy()
        cmd.move_to_pose_param.target_pose.quat = Q.qmult(qe, q)
        print(Q.qmult(qe, q))
        cmd.move_to_pose_param.speed_factor = 0.3
        result, time_run = self.ros_interface.run_primitive(cmd)
        print("result: {}, time run {}".format(result, time_run))

    def test_task_frame(self):
        cmd = self.ros_interface.get_move_to_pose_cmd()

        cmd.move_to_pose_param.task_frame.pos = self.ros_interface.get_pos()
        q = self.ros_interface.get_quat()
        axis = np.array([1., 0, 0])
        angle = -np.pi/2
        qe = Q.axangle2quat(axis, angle, True)
        cmd.move_to_pose_param.task_frame.quat = Q.qmult(qe, q)

        # not moving
        cmd.move_to_pose_param.target_pose.pos = np.zeros(3) + 0.1
        cmd.move_to_pose_param.target_pose.quat = np.array([1., 0., 0., 0.])
        cmd.move_to_pose_param.speed_factor = 0.1
        kp = np.array([1500,]*3 + [80]*2 + [40])
        kd = 2*np.sqrt(kp)

        cmd.move_to_pose_param.controller_gain.kp = kp
        cmd.move_to_pose_param.controller_gain.kd = kd
        cmd.move_to_pose_param.controller_gain.kDefineDamping = 1

        # +0.1
        # cmd.move_to_pose_param.target_pose.pos = np.zeros(3) +0.1
        # cmd.move_to_pose_param.target_pose.quat = np.array([1., 0., 0., 0.])

        result, time_run = self.ros_interface.run_primitive(cmd)
        print("result: {}, time run {}".format(result, time_run))

    def test_task_frame2(self):
        from transforms3d import quaternions as Q
        cmd = self.ros_interface.get_move_to_pose_cmd()

        cmd.move_to_pose_param.task_frame.pos = self.ros_interface.get_pos()
        cmd.move_to_pose_param.task_frame.quat = self.ros_interface.get_quat()

        # not moving
        cmd.move_to_pose_param.target_pose.pos = np.zeros(3) + 0.1
        cmd.move_to_pose_param.target_pose.quat = np.array([1., 0., 0., 0.])
        axis = np.array([1., 0, 0])
        angle = -np.pi/2
        qe = Q.axangle2quat(axis, angle, True)
        q = cmd.move_to_pose_param.target_pose.quat.copy()
        cmd.move_to_pose_param.target_pose.quat = Q.qmult(qe, q)

        cmd.move_to_pose_param.speed_factor = 0.3

        # +0.1
        # cmd.move_to_pose_param.target_pose.pos = np.zeros(3) +0.1
        # cmd.move_to_pose_param.target_pose.quat = np.array([1., 0., 0., 0.])

        result, time_run = self.ros_interface.run_primitive(cmd)
        print("result: {}, time run {}".format(result, time_run))

    def test_speed(self):
        cmd = self.ros_interface.get_move_to_pose_cmd()
        cmd.move_to_pose_param.target_pose.pos += 0.1
        cmd.move_to_pose_param.speed_factor = 0.3
        result, time_run = self.ros_interface.run_primitive(cmd)
        print("result: {}, time run {}".format(result, time_run))

    def test_sequence(self):
        test_position_control()
        test_orientation_control()

if __name__ == '__main__':
    test = TestMoveToPose()
    # test.test_null_cmd()    # pass
    test.test_position_control()
    # test.test_orientation_control()
    # test.test_task_frame2()
    # test.test_task_frame()
    # test.test_speed()
    # test.test_sequence()
    # NOTE: pass all

    # [0.999109,0.00339341,0.041838,0,0.004087,-0.999846,-0.0165033,0,0.0417763,0.0166599,-0.998988,0,0.513504,0.0901372,0.216613,1]