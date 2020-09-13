from learn_seq.controller.robot_state import RobotState

class TaskController(object):
    """Base class for Panda task space controller in Mujoco. The controller
    receives the desired position, velocity or force of an object in the
    base frame and returns the command torque"""
    def __init__(self, robot_state):
        self.robot_state = robot_state

        # controller timestep
        self.dt = self.robot_state.get_timestep()

    def forward_ctrl(self, *argv, **kwargs):
        """Implement the control law. The input can be depends on teh specific
        controller

        :return: joint torque
        :rtype: np.array(7)
        """
        raise NotImplementedError

    def step(self, *argv, **kwargs):
        """Compute the command torque and forward physics simulator one timestep.
        The input is the same as forward_ctrl"""
        # update physics state
        self.robot_state.update()
        tau_cmd = self.forward_ctrl(*argv, **kwargs)
        self.robot_state.set_control_torque(tau_cmd)
        self.robot_state.update_dynamic()

    def get_controller_rate(self):
        return 1/self.dt
