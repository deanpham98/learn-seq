import numpy as np

class Primitive(object):
    """Base class for primitives, whose purpose is to generate the input to the
    controller to execute a certain motion.

    :param learn_seq.controller.RobotState robot_state: access simulation data
    :param learn_seq.controller.TaskController controller :
    :param float timeout : maximum execution time
    """
    def __init__(self,
                 robot_state,
                 controller,
                 timeout=2.,
                 **kwargs):
        self.robot_state = robot_state
        self.controller = controller

        # upper velocity limit
        self.xdot_max = np.array([0.5]*3+[1.]*3)

        # common parameters
        self.timeout = timeout
        self._reset_time()

    def _reset_time(self):
        self.dt = self.robot_state.get_timestep()
        self.t_exec = 0.
        self.timeout_count = self.timeout

    def plan(self):
        """Plan motion parameters before execution."""
        self._reset_time()

    def run(self, viewer=None):
        """Execution until the stop condition is achieved. Return the execution
        time, which is equal to the time if the primitive is executed
        in the real world"""
        self.plan()
        done = False
        t_start = self.robot_state.get_sim_time()
        while not done:
            # update state
            self.robot_state.update()
            # compute tau_cmd and set
            tau_cmd, done = self.step()
            self.robot_state.set_control_torque(tau_cmd)
            # forward dynamic
            self.robot_state.update_dynamic()

            if viewer is not None:
                viewer.render()

        return self.robot_state.get_sim_time() - t_start

    def is_terminate(self):
        """Return whether the primitive is done."""
        raise NotImplementedError

    def step(self):
        """Short summary.

        :return: The command torque and status of the primitive (done or not)
        :rtype: tuple(np.array(7), bool)

        """
        raise NotImplementedError
