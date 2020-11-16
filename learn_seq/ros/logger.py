import os

import rosbag
import rospy
from franka_controllers.msg import HybridControllerState
from franka_motion_primitive.msg import MotionGeneratorState
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

from learn_seq.utils.general import create_file


class Logger:
    """Log data from a ROS topic to a file using rosbag

    :param string sub_topic: topic want to record
    :param Msg sub_msg: message correspond to the topic
    """
    def __init__(self, sub_topic, sub_msg, save_path="data.bag"):
        self._sub = rospy.Subscriber(sub_topic, sub_msg, self.callback)
        create_file(save_path)
        self.save_path = save_path

        self._record = False
        self.bag = rosbag.Bag(self.save_path, "w")
        self.topic = sub_topic
        # record specific event
        self.metadata = Float64MultiArray()
        self.metadata.data = []
        self.metadata.layout.dim = []

    def callback(self, msg):
        if self._record:
            self.bag.write(self.topic, msg)

    def start_record(self):
        self._record = True
        # clear bag file
        self.bag.close()
        self.bag = rosbag.Bag(self.save_path, "w")

    def stop_and_close(self, save_metadata=False):
        self._record = False
        if save_metadata:
            self.bag.write('metadata', self.metadata,
                           rospy.Time(self.bag.get_end_time()))
        self.bag.close()

    def record_event(self, label, time):
        self.metadata.data.append(time)
        dim = MultiArrayDimension()
        dim.label = label
        self.metadata.layout.dim.append(dim)


class LoggerMulti:
    """Log data from multiple publishers

    :param list sub_topics: List of topics want to record.
    :param list sub_msgs: list of messages corresponding to th etopic
    :param list save_dirs: list of save_dirs

    """
    def __init__(self, sub_topics, sub_msgs, save_paths):
        self.logger = []
        for topic, msg, save_dir in zip(sub_topics, sub_msgs, save_paths):
            self.logger.append(Logger(topic, msg, save_path=save_dir))

    def start_record(self):
        for logger in self.logger:
            logger.start_record()

    def stop(self, save_metadata=False):
        for logger in self.logger:
            logger.stop_and_close(save_metadata=save_metadata)

    def record_event(self, label, time):
        for logger in self.logger:
            logger.record_event(label, time)


def basic_logger(save_path, suffix="data/"):
    save_path = os.path.join(save_path, suffix)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    sub_topics = ["/hybrid_controller/state", "/motion_generator/state"]
    sub_msgs = [HybridControllerState, MotionGeneratorState]
    save_files = ["controller.bag", "motion_gen.bag"]
    save_paths = [os.path.join(save_path, f) for f in save_files]
    logger = LoggerMulti(sub_topics, sub_msgs, save_paths)
    return logger
