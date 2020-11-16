from .hybrid import AdmittanceMotion, Displacement, Move2Contact, Move2Target2


class PrimitiveContainer:
    """Store all primitive classes and execute a specific primitives given its
    parameters in Mujoco."""
    def __init__(self, robot_state, controller, tf_pos, tf_quat, timeout=2):
        input_dict = dict(robot_state=robot_state,
                          controller=controller,
                          tf_pos=tf_pos,    # task frame position
                          tf_quat=tf_quat,  # task frame orientation
                          timeout=timeout)
        self.move2target = Move2Target2(**input_dict)
        self.move2contact = Move2Contact(**input_dict)
        self.displacement = Displacement(**input_dict)
        self.admittance = AdmittanceMotion(**input_dict)

    def run(self, type, param, viewer=None):
        primitive = self.get_primitive(type)
        primitive.configure(**param)
        return primitive.run(viewer=viewer)

    def get_primitive(self, type):
        """Get primitive class based on its type"""
        if type == "move2target":
            return self.move2target
        elif type == "move2contact":
            return self.move2contact
        elif type == "displacement":
            return self.displacement
        elif type == "admittance":
            return self.admittance

    def set_task_frame(self, tf_pos, tf_quat):
        self.move2target.set_task_frame(tf_pos, tf_quat)
        self.move2contact.set_task_frame(tf_pos, tf_quat)
        self.displacement.set_task_frame(tf_pos, tf_quat)
        self.admittance.set_task_frame(tf_pos, tf_quat)
