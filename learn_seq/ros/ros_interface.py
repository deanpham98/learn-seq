import math
import sys
from copy import deepcopy

import actionlib
import matplotlib.pyplot as plt
import numpy as np
import rospy
from franka_example_controllers.msg import Gain, VariableImpedanceControllerState, \
                                            VariableImpedanceControllerCommand
from franka_motion_primitive.msg import (ConstantVelocityParam,
                                         DisplacementParam,
                                         MotionGeneratorState, MoveToPoseParam,
                                         PrimitiveType)
from franka_motion_primitive.srv import (RunPrimitive, RunPrimitiveRequest,
                                         SetInitialForce)
from franka_msgs.msg import (ErrorRecoveryAction, ErrorRecoveryActionGoal,
                             FrankaState)

from learn_seq.utils.mujoco import inverse_frame, pose_transform, quat2vec

# constants
FRANKA_ERROR_MODE = 4
KP_DEFAULT = np.array([1000., 1000, 1000, 60, 60, 60])
KD_DEFAULT = 2 * np.sqrt(KP_DEFAULT)
TIMEOUT_DEFAULT = 5


class FrankaRosInterface:
    """communicate with C++ hybrid controller"""
    def __init__(self,
                 sub_state_topic="/variable_impedance_controller/state",
                 pub_command_topic="/variable_impedance_controller/command",
                 pub_gain_topic="/variable_impedance_controller/gain",
                 sub_motion_gen_topic="/motion_generator/state",
                 sub_franka_state_topic="/franka_state_controller/franka_states",
                 set_init_force_serv_name="/motion_generator/set_initial_force",
                 run_primitive_serv_name="/motion_generator/run_primitive",
                 recovery_action_name="/franka_control/error_recovery"):

        self.sub_state_topic = sub_state_topic
        self.sub_motion_gen_topic = sub_motion_gen_topic
        self.sub_franka_state_topic = sub_franka_state_topic
        self.set_init_force_serv_name = set_init_force_serv_name
        self.run_primitive_serv_name = run_primitive_serv_name
        self.recovery_action_name = recovery_action_name
        self._record = False
        rospy.init_node("interface")
        # state
        self._state = {
            "t": None,
            "p": None,
            "pd": None,
            "q": None,
            "qd": None,
            "f": None,
            "fd": None,
            # "pcmd": None,
            "mode": None,
            "ee_vel": None
        }

        # state subscriber (read pose)
        self.sub_state = rospy.Subscriber(sub_state_topic,
                                          VariableImpedanceControllerState,
                                          self._sub_state_callback)

        # Publishers
        self.pub_command = rospy.Publisher(pub_command_topic,
                                           VariableImpedanceControllerCommand)
        self.pub_gain = rospy.Publisher(pub_gain_topic, Gain)

        # motion generator (read force)
        self.sub_motion_gen_state = rospy.Subscriber(
            sub_motion_gen_topic, MotionGeneratorState,
            self._sub_motion_gen_callback)

        # read robot status
        self.sub_franka_state = rospy.Subscriber(
            sub_franka_state_topic, FrankaState,
            self._sub_franka_state_callback)
        # set init force service
        self.serv_set_init_force = rospy.ServiceProxy(
            set_init_force_serv_name, SetInitialForce)
        # run a single primitive service
        self.serv_run_primitive = rospy.ServiceProxy(
            run_primitive_serv_name, RunPrimitive)
        # error recovery action server
        self._recovery_action_client = actionlib.SimpleActionClient(
            recovery_action_name, ErrorRecoveryAction)
        self.reset_record_data()
        # wait until subscriber is on
        timeout = 0.
        while self._state["t"] is None:
            rospy.sleep(0.01)
            timeout += 0.01
            if timeout > 5.:
                sys.exit("Fail to connect to ROS publisher")
                break
        rospy.sleep(1.)

    def _sub_state_callback(self, msg):
        self._state["t"] = msg.time
        self._state["p"] = np.array(msg.p)
        self._state["pd"] = np.array(msg.pd)
        self._state["q"] = np.array(msg.q)
        self._state["qd"] = np.array(msg.qd)
        self._state["jacobian"] = np.reshape(np.array(msg.jacobian), (6, 7))
        self._state["ee_vel"] = np.array(msg.ee_vel)

        # record state for saving
        if self._record:
            self._record_data["p"].append([msg.time, msg.p])
            self._record_data["pd"].append([msg.time, msg.pd])
            self._record_data["q"].append([msg.time, msg.q])
            self._record_data["qd"].append([msg.time, msg.qd])
            self._record_data["f"].append([msg.time, msg.fe])
            self._record_data["fd"].append([msg.time, msg.fd])

    def _sub_motion_gen_callback(self, msg):
        # self._state["f"] = np.array(msg.f_ee)
        # ft sensor signal
        self._state["f"] = np.array(msg.f_s)
        self._state["fd"] = np.array(msg.fd)
        # if self._record:
        #     self._record_data["f"].append([msg.stamp, msg.f_s])
            # self._record_data["fd"].append([msg.stamp, msg.fd])

    def _sub_franka_state_callback(self, msg):
        self._state["mode"] = msg.robot_mode

    def start_record(self):
        self._record = True
        self.reset_record_data()

    def reset_record_data(self):
        self._record_data = {
            "p": [],
            "pd": [],
            "q": [],
            "qd": [],
            "f": [],
            "fd": []
        }

    def stop_record(self, save_path=None):
        self._record = False
        if save_path is not None:
            np.save(save_path, self._record_data)

    def plot_pose(self):
        fig, ax = plt.subplots(3, 2)
        data = deepcopy(self._record_data)
        for k in self._record_data.keys():
            t = np.array([i[0] for i in data[k]])
            if k == "q" or k == "qd":
                d = np.array([quat2vec(np.array(i[1])) for i in data[k]])
            else:
                d = np.array([i[1] for i in data[k]])
            t = t - t[0]
            data[k] = (t, d)
        for i in range(3):
            ax[i, 0].plot(data["p"][0], data["p"][1][:, i])
            ax[i, 1].plot(data["q"][0], data["q"][1][:, i])
            ax[i, 0].plot(data["pd"][0], data["pd"][1][:, i])
            ax[i, 1].plot(data["qd"][0], data["qd"][1][:, i])
            ax[i, 0].legend(["cur", "des"])
            ax[i, 1].legend(["cur", "des"])
        fig.suptitle("position and orientation")
        return ax

    def plot_force(self):
        fig, ax = plt.subplots(3, 2)
        data = deepcopy(self._record_data)
        for k in self._record_data.keys():
            t = np.array([i[0] for i in data[k]])
            if k == "q" or k == "qd":
                d = np.array([quat2vec(np.array(i[1])) for i in data[k]])
            else:
                d = np.array([i[1] for i in data[k]])
            t = t - t[0]
            data[k] = (t, d)
        for i in range(3):
            for j in range(2):
                ax[i, j].plot(data["f"][0], data["f"][1][:, i + 3 * j])
                ax[i, j].plot(data["fd"][0], data["fd"][1][:, i + 3 * j])
                ax[i, j].legend(["cur", "des"])
        fig.suptitle("force")
        return ax

    def run_primitive(self, cmd):
        """Send Primitive run service to ROS controller"""
        rospy.wait_for_service(self.run_primitive_serv_name)
        try:
            res = self.serv_run_primitive(cmd)
            return res.status, res.time
        except rospy.ServiceException as e:
            print("Service RunPrimitive call failed: %s" % e)
            return None, None

    def set_init_force(self):
        """Send service request to set init force"""
        rospy.wait_for_service(self.set_init_force_serv_name)
        try:
            res = self.serv_set_init_force()
            if res.success == 1:
                print("Set init force success")
        except rospy.ServiceException as e:
            print("Service SetInitialForce call failed: %s" % e)

    def move_to_pose(self,
                     p,     # target position
                     q,     # target orientation (quaternion)
                     s,     # speed factor
                     tf_pos=np.zeros(3),    # task frame pos
                     tf_quat=np.array([1., 0, 0, 0]),   # task frame orientation
                     timeout=5.):
        """Move to a target pose"""
        cmd = self.get_move_to_pose_cmd(p, q, np.zeros(6), s, KP_DEFAULT,
                                        KD_DEFAULT, tf_pos, tf_quat, timeout)
        status, t_exec = self.run_primitive(cmd)
        return status, t_exec

    def move_up(self, s=0.03, timeout=2.):
        """Move up"""
        u = np.array([0, 0, 1., 0, 0, 0])
        cmd = self.get_constant_velocity_cmd(u,
                                             s=s,
                                             fs=100,
                                             ft=np.zeros(6),
                                             kp=KP_DEFAULT,
                                             kd=KD_DEFAULT,
                                             timeout=timeout)

        status, t_exec = self.run_primitive(cmd)
        return status, t_exec

    def hold_pose(self):
        """Hold current pose"""
        p = self._state["p"].copy()
        q = self._state["q"].copy()
        status, t_exec = self.move_to_pose(p, q, 0.01)
        return status, t_exec

    def error_recovery(self):
        """Recovery Franka error"""
        goal = ErrorRecoveryActionGoal()
        self._recovery_action_client.send_goal(goal)
        self._recovery_action_client.wait_for_result(
            rospy.Duration.from_sec(2.0))

    def get_ee_pose(self, frame_pos=None, frame_quat=None):
        """Get current ee pose w.r.t a specific frame"""
        p = self._state["p"].copy()
        q = self._state["q"].copy()
        if frame_pos is None:
            frame_pos = np.zeros(3)
        if frame_quat is None:
            frame_quat = np.array([1., 0, 0, 0])
        inv_pos, inv_quat = inverse_frame(frame_pos, frame_quat)
        pf, qf = pose_transform(p, q, inv_pos, inv_quat)
        return pf, qf

    def get_ee_velocity(self):
        """Get current ee velocity w.r.t base frame"""
        return self._state["ee_vel"]

    def get_zero_jacobian(self):
        """Get Zero Jacobian"""
        return self._state["jacobian"]

    def get_ee_pos(self):
        """Get current position w.r.t the base frame"""
        return self._state["p"].copy()

    def get_ee_force(self, frame_quat=None):
        """Get current force w.r.t a particular frame orientation"""
        return self._state["f"]

    def get_robot_mode(self):
        """Get current robot mode"""
        return self._state["mode"]

    def get_move_to_pose_cmd(self,
                             pt=None,    # target position
                             qt=None,    # target orientation
                             ft=np.zeros(6),    # target force
                             s=0.1,     # speed factor
                             kp=KP_DEFAULT,    # stiffness
                             kd=KD_DEFAULT,    # damping
                             tf_pos=np.zeros(3),    # task frame position
                             tf_quat=np.array([1., 0, 0, 0]),   # task frame
                                                                # orientation
                             timeout=None):
        """Convert param to ROS msg"""
        cmd = RunPrimitiveRequest()
        # cmd.type = PrimitiveType.MoveToPose
        cmd.type = PrimitiveType.MoveToPoseFeedback
        p = MoveToPoseParam()
        p.task_frame.pos = tf_pos
        p.task_frame.quat = tf_quat

        p.target_pose.pos = self.get_pos() if pt is None else pt
        p.target_pose.quat = self.get_quat() if qt is None else qt

        p.speed_factor = s
        p.timeout = timeout or TIMEOUT_DEFAULT
        p.fd = ft
        p.controller_gain = self.get_gain(kp, kd)

        cmd.time = rospy.Time.now().to_sec()
        cmd.move_to_pose_param = p
        # gain
        return cmd

    def get_constant_velocity_cmd(self,
                                  u=[0,1.0,0,0,0],    # move direction
                                  s=0.01,    # speed factor
                                  fs=100.,   # threshold force
                                  ft=np.zeros(6),   # target force
                                  kp=KP_DEFAULT,   # stiffness
                                  kd=KD_DEFAULT,   # damping
                                  tf_pos=np.zeros(3),
                                  tf_quat=np.array([1., 0, 0, 0]),
                                  timeout=None):
        """Convert param to ROS msg"""
        cmd = RunPrimitiveRequest()
        cmd.type = PrimitiveType.ConstantVelocity

        p = ConstantVelocityParam()
        p.task_frame.pos = tf_pos
        p.task_frame.quat = tf_quat
        p.speed_factor = s
        p.direction = u
        p.timeout = timeout or TIMEOUT_DEFAULT
        p.f_thresh = fs
        p.fd = ft
        p.controller_gain = self.get_gain(kp, kd)

        cmd.time = rospy.Time.now().to_sec()
        cmd.constant_velocity_param = p
        return cmd

    def get_displacement_cmd(self,
                             u,     # move direction
                             s,     # speed factor
                             fs,    # force threshold
                             ft,    # target force
                             delta_d,   # distance threshold
                             kp,    # stiffness
                             kd,    # damping
                             tf_pos=np.zeros(3),
                             tf_quat=np.array([1., 0, 0, 0]),
                             timeout=None):
        """Convert param to ROS msg"""
        cmd = RunPrimitiveRequest()
        cmd.type = PrimitiveType.Displacement

        p = DisplacementParam()
        p.task_frame.pos = tf_pos
        p.task_frame.quat = tf_quat
        p.speed_factor = s
        p.direction = u
        p.timeout = timeout or TIMEOUT_DEFAULT
        p.f_thresh = fs
        p.fd = ft
        p.displacement = delta_d
        p.controller_gain = self.get_gain(kp, kd)

        cmd.time = rospy.Time.now().to_sec()
        cmd.displacement_param = p

        return cmd

    def get_admittance_cmd(self,
                           kd_adt,  # admittance matrix
                           ft,      # target force
                           pt,      # goal position
                           goal_thresh,  # goal thresh
                           kp,      # stiffness
                           kd,      # damping
                           tf_pos=np.zeros(3),
                           tf_quat=np.array([1., 0, 0, 0]),
                           timeout=None):
        """Convert param to ROS msg"""
        cmd = RunPrimitiveRequest()
        cmd.type = PrimitiveType.AdmittanceMotion

        p = AdmittanceMotionParam()
        p.task_frame.pos = tf_pos
        p.task_frame.quat = tf_quat
        p.kd = kd_adt
        p.fd = ft
        p.goal_thresh = goal_thresh
        p.pt = pt
        p.timeout = timeout or TIMEOUT_DEFAULT
        p.controller_gain = self.get_gain(kp, kd)
        cmd.time = rospy.Time.now().to_sec()
        cmd.admittance_motion_param = p

        return cmd

    def get_gain(self, kp, kd=None):
        """Convert numpy array to Gain Msg"""
        gain = Gain()
        gain.kp = kp
        gain.kd = kd
        return gain

    def set_gain(self, kp, kd=None):
        if kd is None:
            kd = 2 * math.sqrt(kp)
        gain = self.get_gain(kp, kd)
        self.pub_gain.publish(gain)
        
    def get_cmd(self, f, p, q, v):
        cmd = VariableImpedanceControllerCommand()
        cmd.f = f
        cmd.p = p
        cmd.q = q
        cmd.v = v

        return cmd

    def set_cmd(self, f, p, q, v):
        cmd = self.get_cmd(f, p, q, v)
        self.pub_command.publish(cmd)

    def get_ros_time(self):
        return self._state["t"]

    def get_pose_control_cmd(self):
        return self._state["pcmd"], self._state["qd"]

    def get_pos(self):
        return self._state["p"].copy()

    def get_quat(self):
        return self._state["q"].copy()