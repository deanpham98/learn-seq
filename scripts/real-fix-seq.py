import time
import numpy as np
from copy import deepcopy
from learn_seq.ros.ros_interface import FrankaRosInterface, FRANKA_ERROR_MODE
from learn_seq.primitive.real_container import RealPrimitiveContainer
from learn_seq.utils.mujoco import mat2quat, mul_quat, integrate_quat

# no runs
N = 10

# peg transformation matrix
# T_HOLE = np.array([0.998207,0.0595135,-0.00458621,0,0.0594896,-0.998206,-0.00517501,0,-0.00488606,0.00489299,-0.999976, 0,\
#                    0.530483,0.0755677,0.152598,1]).reshape((4, 4)).T
# square hole
# T_HOLE = np.array([0.997874,-0.0643417,-0.00936179,0,-0.0643266,-0.997917,0.0019045,0,-0.00946502,-0.00129826,-0.999954, 0,\
#                    0.528331,-0.122101,0.142952,1]).reshape((4, 4)).T
# triangle hole
T_HOLE = np.array([0.999711,-0.0170507,0.0163608,0,-0.0167906,-0.999723,-0.0159042,0,0.0166277,0.0156252,-0.99974, 0,\
                   0.534439,-0.0618864,0.143541,1]).reshape((4, 4)).T

SHAPE="round"

HOLE_DEPTH = 0.02
GOAL_THRESH = 2e-3
# stiffness
KP = np.array([2500, 1500, 1500] + [60, 60, 30])
KD = 2*np.sqrt(KP)

def round_mp_list():
    # params

    mp_list = []
    # move x direction 5mm
    # dp = 5./1000
    # param = dict(
    #     u=np.array([1, 0, 0, 0 ,0 ,0]),
    #     s=0.05,
    #     ft=np.zeros(6),
    #     delta_d=dp,
    #     fs=10.,
    #     kp=KP,
    #     kd=KD,
    #     timeout=3.
    # )
    # mp_list.append(("displacement", deepcopy(param)))

    # rotate y direction
    dp = 5*np.pi/180
    param = dict(
        u=np.array([0, 0, 0, 0 ,-1 ,0]),
        s=0.1,
        ft=np.zeros(6),
        delta_d=dp,
        fs=1.,
        kp=KP,
        kd=KD,
        timeout=3.
    )
    mp_list.append(("displacement", deepcopy(param)))

    # move z until contact
    param = dict(
        u=np.array([0, 0, -1, 0 ,0 ,0]),
        s=0.01,
        ft=np.zeros(6),
        fs=3.,
        kp=KP,
        kd=KD,
        timeout=3.
    )
    mp_list.append(("move2contact", deepcopy(param)))

    # move -x until contact
    param = dict(
        u=np.array([-1, 0, 0, 0 ,0 ,0]),
        s=0.005,
        ft=np.array([0, 0, -3, 0, 0, 0]),
        fs=5.,
        kp=KP,
        kd=KD,
        timeout=5.
    )
    mp_list.append(("move2contact", deepcopy(param)))

    # rotate y until contact
    param = dict(
        u=np.array([0, 0, 0, 0 ,1 ,0]),
        s=0.04,
        ft=np.array([0, 0, -3, 0, 0, 0]),
        fs=0.5,
        kp=KP,
        kd=KD,
        timeout=5.
    )
    mp_list.append(("move2contact", deepcopy(param)))

    # insert
    kd_adt = np.array([0.01]*3 + [0.15]*3)
    f = 8.
    kp_insert =[500, 500, 500, 50, 50, 50.]
    kd_insert = [10.]*6
    param = dict(
        kd_adt=kd_adt,
         ft=np.array([0, 0, -f, 0, 0, 0]),
         pt=np.array([0, 0, -HOLE_DEPTH]),
         goal_thresh=GOAL_THRESH,
         kp=kp_insert,
         kd=kd_insert,
         timeout=5.
    )
    mp_list.append(("admittance", deepcopy(param)))
    return mp_list

def square_mp_list():
    mp_list = []
    # move x direction 5mm
    # dp = 5./1000
    # param = dict(
    #     u=np.array([1, 0, 0, 0 ,0 ,0]),
    #     s=0.05,
    #     ft=np.zeros(6),
    #     delta_d=dp,
    #     fs=10.,
    #     kp=KP,
    #     kd=KD,
    #     timeout=3.
    # )
    # mp_list.append(("displacement", deepcopy(param)))

    # rotate y direction
    dp = 5*np.pi/180
    param = dict(
        u=np.array([0, 0, 0, 0 ,-1 ,0]),
        s=0.1,
        ft=np.zeros(6),
        delta_d=dp,
        fs=1.,
        kp=KP,
        kd=KD,
        timeout=3.
    )
    mp_list.append(("displacement", deepcopy(param)))

    # rotate x direction
    dp = 5*np.pi/180
    param = dict(
        u=np.array([0, 0, 0, -1 , 0 ,0]),
        s=0.1,
        ft=np.zeros(6),
        delta_d=dp,
        fs=1.,
        kp=KP,
        kd=KD,
        timeout=3.
    )
    mp_list.append(("displacement", deepcopy(param)))


    # move z until contact
    param = dict(
        u=np.array([0, 0, -1, 0 ,0 ,0]),
        s=0.01,
        ft=np.zeros(6),
        fs=3.,
        kp=KP,
        kd=KD,
        timeout=3.
    )
    mp_list.append(("move2contact", deepcopy(param)))

    # move -x until contact
    param = dict(
        u=np.array([-1, 0, 0, 0 ,0 ,0]),
        s=0.005,
        ft=np.array([0, 0, -3, 0, 0, 0]),
        fs=5.,
        kp=KP,
        kd=KD,
        timeout=5.
    )
    mp_list.append(("move2contact", deepcopy(param)))

    # rotate y until contact
    param = dict(
        u=np.array([0, 0, 0, 0 ,1 ,0]),
        s=0.04,
        ft=np.array([0, 0, -3, 0, 0, 0]),
        fs=0.5,
        kp=KP,
        kd=KD,
        timeout=5.
    )
    mp_list.append(("move2contact", deepcopy(param)))

    # rotate x until contact
    param = dict(
        u=np.array([0, 0, 0, 1 ,0 ,0]),
        s=0.04,
        ft=np.array([0, 0, -3, 0, 0, 0]),
        fs=0.5,
        kp=KP,
        kd=KD,
        timeout=5.
    )
    mp_list.append(("move2contact", deepcopy(param)))

    # insert
    kd_adt = np.array([0.01]*3 + [0.15]*3)
    f = 8.
    kp_insert =[500, 500, 500, 50, 50, 50.]
    kd_insert = [10.]*6
    param = dict(
        kd_adt=kd_adt,
         ft=np.array([0, 0, -f, 0, 0, 0]),
         pt=np.array([0, 0, -HOLE_DEPTH]),
         goal_thresh=GOAL_THRESH,
         kp=kp_insert,
         kd=kd_insert,
         timeout=5.
    )
    mp_list.append(("admittance", deepcopy(param)))
    return mp_list

class FixSequence:
    def __init__(self, T_hole, mp_list):
        hole_pos = T_HOLE[:3, 3]
        hole_rot = T_HOLE[:3, :3]
        hole_quat = mat2quat(hole_rot)
        # make the z axis point out of the hole
        qx = np.array([np.cos(np.pi/2), np.sin(np.pi/2), 0, 0])
        hole_quat = mul_quat(hole_quat, qx)

        self.ros_interface = FrankaRosInterface()
        self.container = RealPrimitiveContainer(self.ros_interface, hole_pos, hole_quat)
        # task frame
        self.hole_pos = hole_pos.copy()
        self.hole_quat = hole_quat.copy()
        self.tf_pos = hole_pos.copy()
        self.tf_quat = hole_quat.copy()

        # target pose relative to the task frame
        self.target_pos = np.array([0., 0, -HOLE_DEPTH])
        self.target_quat = np.array([0, 1., 0, 0])

        # init pos
        self.init_pos = np.array([0, 0, 0.01])
        self.init_quat = self.target_quat.copy()

        # goal thresh
        self.goal_thresh = GOAL_THRESH
        # mp
        self.mp_list = mp_list


    def reset(self, p, q):
        #
        self.traj_time = 0
        self._auto_reset()
        time.sleep(0.5)

        # move ip if inside hole
        pc = self.ros_interface.get_ee_pos()
        if pc[2] < self.tf_pos[2]:
            self.ros_interface.move_up(timeout=2.)


        # calibrate force
        p0 = self.target_pos.copy()
        p0[2] = 0.01
        q0 = self.target_quat.copy()
        self.ros_interface.move_to_pose(p0, q0, 0.3, self.tf_pos, self.tf_quat, 10)


        time.sleep(0.5)
        self.ros_interface.set_init_force()


        # move to reset position
        self.ros_interface.move_to_pose(p, q, 0.1, self.tf_pos, self.tf_quat, 10)


    def set_task_frame(self, tf_pos, tf_quat):
        self.container.set_task_frame(tf_pos, tf_quat)
        self.tf_pos = tf_pos
        self.tf_quat = tf_quat

    def _auto_reset(self):
        if self.ros_interface.get_robot_mode() == FRANKA_ERROR_MODE:
            self.ros_interface.move_up(timeout=0.1)
            self.ros_interface.error_recovery()

    def is_success(self):
        p, q = self.ros_interface.get_ee_pose(self.tf_pos, self.tf_quat)
        return np.linalg.norm(p[:3] - self.target_pos[:3]) < self.goal_thresh

    # run once (don't know estimation error)
    def run_single(self, p, q, record=False):
        # reset
        self.reset(p, q)
        if record:
            self.ros_interface.start_record()
        for type, param in self.mp_list:
            status, t_exec = self.container.run(type, param)
            self.traj_time += t_exec
            error = self._auto_reset()
            if error:
                break
            # input("test")
        if record:
            self.ros_interface.stop_record("fix-seq-traj.npy")
        success = self.is_success()
        return success, self.traj_time

    def run(self, no_run=1,
            hole_pos_error=0.,
            hole_rot_error=0.,
            delta_init_pos=0.,
            delta_init_rot=0.):
        no_success = 0
        t_exec = []
        for i in range(no_run):
            # add hole pos error
            pos_dir = np.zeros(3)
            pos_dir[:3] = (np.random.random(3) - 0.5) * 2
            # pos_dir[:3] = pos_dir[:3] / np.linalg.norm(pos_dir)
            pos_dir[:3] = pos_dir[:3] / np.max(np.abs(pos_dir))
            hole_pos = self.hole_pos + hole_pos_error * pos_dir

            rot_dir = (np.random.random(3) - 0.5) * 2
            # rot_dir = rot_dir / np.linalg.norm(rot_dir)
            rot_dir = rot_dir / np.max(np.abs(rot_dir))
            hole_rot_rel = hole_rot_error * rot_dir
            hole_quat = integrate_quat(self.hole_quat, hole_rot_rel, 1)
            self.set_task_frame(hole_pos, hole_quat)
            print(hole_pos_error*pos_dir*1000, hole_rot_rel*180 / np.pi)
            # init pos
            pos_dir = np.zeros(3)
            pos_dir[:3] = (np.random.random(3) - 0.5) * 2
            # pos_dir[:3] = pos_dir[:3] / np.linalg.norm(pos_dir)
            pos_dir[:3] = pos_dir[:3] / np.max(np.abs(pos_dir))
            p0 = self.init_pos + delta_init_pos * pos_dir

            rot_dir = (np.random.random(3) - 0.5) * 2
            # rot_dir = rot_dir / np.linalg.norm(rot_dir)
            rot_dir = rot_dir / np.max(np.abs(rot_dir))
            hole_rot_rel = delta_init_rot * rot_dir
            q0 = integrate_quat(self.init_quat, hole_rot_rel, 1)
            print(delta_init_pos*pos_dir*1000, hole_rot_rel*180 / np.pi)

            success, traj_time = self.run_single(p0, q0)
            if success:
                no_success += 1
                t_exec.append(traj_time)

        # no success
        print("success rate {}".format(float(no_success)/no_run))
        print("mean success time {}".format(np.mean(t_exec)))
        print("std success time {}".format(np.std(t_exec)))

def exp_config():
    config = []
    c1 = dict(
        no_run=10,
        hole_pos_error=0.5/1000,
        hole_rot_error=0.5*np.pi/180,
        delta_init_pos=0.,
        delta_init_rot=0.
    )
    config.append(c1)

    c2 = dict(
        no_run=10,
        hole_pos_error=1.5/1000,
        hole_rot_error=1.5*np.pi/180,
        delta_init_pos=0.,
        delta_init_rot=0.
    )
    config.append(c2)

    c3 = dict(
        no_run=10,
        hole_pos_error=0.5/1000,
        hole_rot_error=0.5*np.pi/180,
        delta_init_pos=1./1000,
        delta_init_rot=1.*np.pi/180    )
    config.append(c3)
    return config

def eval_fix_seq(shape="round"):
    if shape == "round":
        mp_list = round_mp_list()
    elif shape=="square":
        mp_list = square_mp_list()
    fix_seq = FixSequence(T_hole=T_HOLE, mp_list=mp_list)
    # evaluate for different estimation error and init pos
    config = exp_config()
    for c in config:
        fix_seq.run(**c)

def record_single_run():
    fix_seq = FixSequence(T_hole=T_HOLE)
    fix_seq.run_single(fix_seq.init_pos, fix_seq.init_quat, record=True)

if __name__ == '__main__':
    eval_fix_seq(SHAPE)
    # record_single_run()
