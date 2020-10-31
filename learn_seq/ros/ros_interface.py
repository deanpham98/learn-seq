import sys
from copy import deepcopy
import json
import numpy as np
import matplotlib.pyplot as plt
import rospy
import actionlib
from franka_controllers.msg import HybridControllerState, Gain
from franka_motion_primitive.srv import RunPrimitive, SetInitialForce,\
                RunPrimitiveRequest, SetInitialForceRequest
from franka_motion_primitive.msg import MotionGeneratorState, PrimitiveType,\
                MoveToPoseParam, ConstantVelocityParam, DisplacementParam, AdmittanceMotionParam
from franka_msgs.msg import FrankaState, ErrorRecoveryAction, ErrorRecoveryActionGoal
from learn_seq.utils.mujoco import pose_transform, inverse_frame, quat2vec

FRANKA_ERROR_MODE = 4
KP_DEFAULT = np.array([2000.] + [1500.]*2 + [60, 60, 30])
KD_DEFAULT = 2*np.sqrt(KP_DEFAULT)
TIMEOUT_DEFAULT = 5

class FrankaRosInterface:
    def __init__(self,):
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
            "pcmd": None,
            "mode": None}

        # state subscriber (read pose)
        sub_state_topic = "/hybrid_controller/state"
        self.sub_state = rospy.Subscriber(sub_state_topic, HybridControllerState,
                            self._sub_state_callback)

        # motion generator (read force)
        sub_motion_gen_topic = "/motion_generator/state"
        self.sub_motion_gen_state = rospy.Subscriber(sub_motion_gen_topic,
                                        MotionGeneratorState, self._sub_motion_gen_callback)

        # read robot status
        sub_franka_state_topic = "/franka_state_controller/franka_states"
        self.sub_franka_state = rospy.Subscriber(sub_franka_state_topic,
                                        FrankaState, self._sub_franka_state_callback)
        # set init force service
        self.serv_set_init_force = rospy.ServiceProxy("/motion_generator/set_initial_force", SetInitialForce)
        # run a single primitive service
        self.serv_run_primitive = rospy.ServiceProxy("/motion_generator/run_primitive", RunPrimitive)
        # error recovery action server
        self._recovery_action_client = actionlib.SimpleActionClient('/franka_control/error_recovery', ErrorRecoveryAction)
        self.reset_record_data()
        # wait until subscriber is on
        timeout = 0.
        while self._state["t"] is None:
            rospy.sleep(0.01)
            timeout += 0.01
            if timeout>5.:
                sys.exit("Fail to connect to ROS publisher")
                break
        rospy.sleep(1.)

    def _sub_state_callback(self, msg):
        self._state["t"] = msg.time
        self._state["p"] = np.array(msg.p)
        self._state["pd"] = np.array(msg.pd)
        self._state["q"] = np.array(msg.q)
        self._state["qd"] = np.array(msg.qd)
        self._state["pcmd"] = np.array(msg.p_cmd)
        if self._record:
            self._record_data["p"].append([msg.time, msg.p])
            self._record_data["pd"].append([msg.time, msg.pd])
            self._record_data["q"].append([msg.time, msg.q])
            self._record_data["qd"].append([msg.time, msg.qd])

    def _sub_motion_gen_callback(self, msg):
        # self._state["f"] = np.array(msg.f_ee)
        self._state["f"] = np.array(msg.f_s)
        self._state["fd"] = np.array(msg.fd)
        if self._record:
            self._record_data["f"].append([msg.stamp, msg.f_s])
            self._record_data["fd"].append([msg.stamp, msg.fd])

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
        data =deepcopy(self._record_data)
        for k in self._record_data.keys():
            t = np.array([i[0] for i in data[k]])
            if k is "q" or k is "qd":
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
            ax[i, 0].legend(["c", "d"])
            ax[i, 1].legend(["c", "d"])
        fig.suptitle("position and orientation")
        return ax

    def plot_force(self):
        fig, ax = plt.subplots(3, 2)
        data =deepcopy(self._record_data)
        for k in self._record_data.keys():
            t = np.array([i[0] for i in data[k]])
            if k is "q" or k is "qd":
                d = np.array([quat2vec(np.array(i[1])) for i in data[k]])
            else:
                d = np.array([i[1] for i in data[k]])
            t = t - t[0]
            data[k] = (t, d)
        for i in range(3):
            for j in range(2):
                ax[i, j].plot(data["f"][0], data["f"][1][:, i+3*j])
                ax[i, j].plot(data["fd"][0], data["fd"][1][:, i+3*j])
                ax[i, j].legend(["c", "d"])
        fig.suptitle("force")
        return ax

    def run_primitive(self, cmd):
        # print("run primitive {}", action)
        rospy.wait_for_service("/motion_generator/run_primitive")
        try:
            res = self.serv_run_primitive(cmd)
            return res.status, res.time
        except rospy.ServiceException as e:
            print("Service RunPrimitive call failed: %s"%e)
            return None, None

    def set_init_force(self):
        # print("start calling service SetInitialForce")
        rospy.wait_for_service("/motion_generator/set_initial_force")
        try:
            res = self.serv_set_init_force()
            if res.success == 1:
                print("Set init force success")
        except rospy.ServiceException as e:
            print("Service SetInitialForce call failed: %s"%e)
        # print("end calling service SetInitialForce")

    def error_recovery(self):
        goal = ErrorRecoveryActionGoal()
        self._recovery_action_client.send_goal(goal)
        self._recovery_action_client.wait_for_result(rospy.Duration.from_sec(2.0))

    def move_to_pose(self, p, q, s,
                     tf_pos=np.zeros(3),
                     tf_quat=np.array([1., 0, 0 ,0]),
                     timeout=5.):
        cmd = self.get_move_to_pose_cmd(p, q, np.zeros(6), s,
                                        KP_DEFAULT, KD_DEFAULT,
                                        tf_pos, tf_quat, timeout)
        status, t_exec =  self.run_primitive(cmd)
        return status, t_exec

    def move_up(self, s=0.05, timeout=2.):
        u = np.array([0, 0, 1., 0, 0, 0])
        cmd = self.get_constant_velocity_cmd(u, s=s, fs=100, ft=np.zeros(6),
                    kp=KP_DEFAULT, kd=KD_DEFAULT, timeout=timeout)

        status, t_exec =  self.run_primitive(cmd)
        return status, t_exec

    def hold_pose(self):
        p = self._state["p"].copy()
        q = self._state["q"].copy()
        status, t_exec = self.move_to_pose(p, q, 0.01)
        return status, t_exec

    def error_recovery(self):
        goal = ErrorRecoveryActionGoal()
        self._recovery_action_client.send_goal(goal)
        self._recovery_action_client.wait_for_result(rospy.Duration.from_sec(2.0))

    def get_ee_pose(self, frame_pos=None, frame_quat=None):
        p = self._state["p"].copy()
        q = self._state["q"].copy()
        if frame_pos is None:
            frame_pos = np.zeros(3)
        if frame_quat is None:
            frame_quat = np.array([1., 0, 0, 0])
        inv_pos, inv_quat = inverse_frame(frame_pos, frame_quat)
        pf, qf = pose_transform(p, q, inv_pos, inv_quat)
        return pf, qf

    def get_ee_pos(self):
        return self._state["p"].copy()

    def get_ee_force(self, frame_quat=None):
        return self._state["f"]

    def get_robot_mode(self):
        return self._state["mode"]

    def get_move_to_pose_cmd(self, pt, qt, ft, s, kp, kd,
                             tf_pos=np.zeros(3),
                             tf_quat=np.array([1., 0,0 ,0]),
                             timeout=None):
        cmd = RunPrimitiveRequest()
        # cmd.type = PrimitiveType.MoveToPose
        cmd.type = PrimitiveType.MoveToPoseFeedback
        p = MoveToPoseParam()
        p.task_frame.pos = tf_pos
        p.task_frame.quat = tf_quat

        p.target_pose.pos = pt
        p.target_pose.quat = qt

        p.speed_factor = s
        p.timeout = timeout or TIMEOUT_DEFAULT
        p.fd = ft
        p.controller_gain = self.get_gain_cmd(kp, kd)

        cmd.time = rospy.Time.now().to_sec()
        cmd.move_to_pose_param = p
        # gain
        return cmd

    def get_constant_velocity_cmd(self, u, s, fs, ft, kp, kd,
                                  tf_pos=np.zeros(3),
                                  tf_quat=np.array([1., 0,0 ,0]),
                                  timeout=None):
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
        p.controller_gain = self.get_gain_cmd(kp, kd)

        cmd.time = rospy.Time.now().to_sec()
        cmd.constant_velocity_param = p
        return cmd

    def get_displacement_cmd(self, u, s, fs, ft, delta_d, kp, kd,
                             tf_pos=np.zeros(3),
                             tf_quat=np.array([1., 0,0 ,0]),
                             timeout=None):
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
        p.controller_gain = self.get_gain_cmd(kp, kd)

        cmd.time = rospy.Time.now().to_sec()
        cmd.displacement_param = p

        return cmd

    def get_admittance_cmd(self, kd_adt, ft, pt, goal_thresh, kp, kd,
                            tf_pos=np.zeros(3),
                            tf_quat=np.array([1., 0,0 ,0]),
                            timeout=None):
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
        p.controller_gain = self.get_gain_cmd(kp, kd)
        cmd.time = rospy.Time.now().to_sec()
        cmd.admittance_motion_param = p

        return cmd

    def get_gain_cmd(self, kp, kd=None):
        cmd = Gain()
        if kd is None:
            cmd.kDefineDamping = 0
            cmd.kp = kp
        else:
            cmd.kDefineDamping = 1
            cmd.kp = kp
            cmd.kd = kd
        return cmd

    def get_ros_time(self):
        return self._state["t"]

    def get_pose_control_cmd(self):
        return self._state["pcmd"], self._state["qd"]
