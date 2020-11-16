class RealPrimitiveContainer:
    """Store all primitive classes and execute a specific primitives given its
    parameters on real robot."""
    def __init__(self, ros_interface, tf_pos, tf_quat, timeout=2):
        self.set_task_frame(tf_pos, tf_quat)
        self.ros_interface = ros_interface

    def run(self, type, param):
        """Run the MP and return its status (0 for FAIL and 1 for SUCCESS),
        and execution time"""
        if not self._is_feasible(param):
            return self.ros_interface.hold_pose()
        else:
            param["tf_pos"] = self.tf_pos
            param["tf_quat"] = self.tf_quat
            cmd = self.get_primitive_req(type, param)
            status, t_exec = self.ros_interface.run_primitive(cmd)
            return status, t_exec

    def get_primitive_req(self, type, param):
        """Get the service request for running a specific MP given its type
        and param"""
        if type == "move2target":
            cmd = self.ros_interface.get_move_to_pose_cmd(**param)
        elif type == "move2contact":
            cmd = self.ros_interface.get_constant_velocity_cmd(**param)
        elif type == "displacement":
            cmd = self.ros_interface.get_displacement_cmd(**param)
        elif type == "admittance":
            cmd = self.ros_interface.get_admittance_cmd(**param)
        return cmd

    def set_task_frame(self, tf_pos, tf_quat):
        self.tf_pos = tf_pos
        self.tf_quat = tf_quat

    def _is_feasible(self, param):
        """Check whether a MP is feasible in current state given its param"""
        if param["ft"][2] != 0:
            p, q = self.ros_interface.get_ee_pose(self.tf_pos, self.tf_quat)
            if p[2] > 0.005:
                return False
        return True
