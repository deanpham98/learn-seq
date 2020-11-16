import numpy as np
import rosbag

from learn_seq.utils.mujoco import quat_error


def read_bag(bag_file):
    """read a bag file.

    :param string bag_file: path lead to bag file
    :return: msg time SENT, msg, event time, event label
    :rtype: tuple(list(float), list(rospy.Msg), list(float), list(string))

    """
    bag = rosbag.Bag(bag_file)
    msgs = []
    ts = []
    event_time = []
    event_label = []
    for topic, msg, t in bag.read_messages():
        if topic == "metadata":
            event_time = msg.data
            event_label = [d.label for d in msg.layout.dim]
        else:
            msgs.append(msg)
            ts.append(t.to_sec())

    event_time = [te - ts[0] for te in event_time]
    print(event_time, event_label)
    return ts, msgs, event_time, event_label


def extract_pos_error(msg_list):
    """Extract position error from a list of HybridControllerState msg.

    :param type msg_list: Description of parameter `msg_list`.
    :return: Description of returned object.
    :rtype: type

    """
    N = len(msg_list)
    # position
    p = [m.p for m in msg_list]
    p = np.array(p)

    # desired position
    pd = [m.pd for m in msg_list]
    pd = np.array(pd)

    # position error
    ep = pd[:N - 1] - p[1:]

    # quaternion
    q = [m.q for m in msg_list]
    q = np.array(q)

    # desired quaternion
    qd = [m.qd for m in msg_list]
    qd = np.array(qd)

    eq = np.zeros((N - 1, 3))
    # orientation error
    for i in range(N - 1):
        # q1_inv = Q.qinverse(q[i+1, :])
        # qe = Q.qmult(qd[i, :], q[i+1,:])
        # aae = Q.quat2axangle(qe)
        # # print(q1_inv, q[i+1, :], qe, aae)
        # eq[i, :] = aae[0]*aae[1]
        eq[i, :] = quat_error(qd[i, :], q[i + 1, :])

    return ep, eq


def extract_pose(msg_list):
    # position
    p = [m.p for m in msg_list]
    p = np.array(p)

    # desired position
    pd = [m.pd for m in msg_list]
    pd = np.array(pd)
    # quaternion
    q = [m.q for m in msg_list]
    q = np.array(q)

    # desired quaternion
    qd = [m.qd for m in msg_list]
    qd = np.array(qd)

    return p, q, pd, qd


def extract_error(msg_list):
    er = np.array([m.data for m in msg_list])
    return er
