
class TaskController(object):
    """Base class for Panda task space controller in Mujoco. The controller
    receives the desired position, velocity and/or force of the end-effector in
    the base frame and returns the command torque.

    The end-effector definition is in the `RobotState` robot_state.

    :param learn_seq.controller.RobotState robot_state: access simulation data
    """
    def __init__(self, robot_state):
        self.robot_state = robot_state

        # controller timestep
        self.dt = self.robot_state.get_timestep()

    def forward_ctrl(self, *argv, **kwargs):
        """Implement the control law. The inputs may depends on the specific
        controller, which are usually composed of one or multiple, but
        not limited to the following components:

            - end-effector position
            - end-effector velocity
            - external force

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
        """Timestep of the controller"""
        return 1/self.dt
