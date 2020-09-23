from copy import deepcopy
import numpy as np
from torch.nn import ReLU
from rlpyt.algos.pg.ppo import PPO
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from learn_seq.envs.wrapper import StructuredActionSpaceWrapper
from learn_seq.rlpyt.ppo_agent import PPOStructuredRealAgent
from learn_seq.utils.mujoco import mat2quat, mul_quat

# peg transformation matrix
T_HOLE = np.array([0.995478,0.0930097,-0.018803,0,\
                   0.0930499,-0.995651,0.00127024,0,\
                   -0.0186034,-0.00301417,-0.999822,0,\
                   0.529588,-0.0857657,0.150818,1]).reshape((4, 4)).T
hole_pos = T_HOLE[:3, 3]
hole_rot = T_HOLE[:3, :3]
hole_quat = mat2quat(hole_rot)
# make the z axis point out of the hole
qx = np.array([np.cos(np.pi/2), np.sin(np.pi/2), 0, 0])
hole_quat = mul_quat(hole_quat, qx)

#
SPEED_FACTOR_RANGE = [0.01, 0.05]
SLIDING_SPEED_FACTOR_RANGE = [0.004, 0.01]
FORCE_THRESH_RANGE = [5, 10]
TORQUE_THRESH_RANGE = [0.1, 1]
TRANSLATION_DISPLACEMENT_RANGE = [0.001, 0.005]
ROTATION_DISPLACEMENT_RANGE = [np.pi/180, 5*np.pi/180]
INSERTION_FORCE_RANGE = [5., 15]
KD_ADMITTANCE_ROT_RANGE = [0.05, 0.5]
FORCE_THRESH_MOVE_DOWN = 5.
SAFETY_FORCE = 20.
SAFETY_TORQUE = 2.

KP_DEFAULT = [1000.]*3 + [60.]*3
KD_DEFAULT = [2*np.sqrt(i) for i in KP_DEFAULT]
TIMEOUT = 2.

# no discretization
NO_QUANTIZATION = 4
PRIMITIVES_PER_TYPE = 16

# TODO change in environment
HOLE_DEPTH = 0.02
DEPTH_THRESH = 0.95
TRAINING_STEP = 1000000
SEED = 18

# ----- Primitive config
primitive_list = []
# move down until contact
for i in range(NO_QUANTIZATION):
    dv = (SPEED_FACTOR_RANGE[1] - SPEED_FACTOR_RANGE[0])/NO_QUANTIZATION
    v = SPEED_FACTOR_RANGE[0] + dv/2 + i*dv

    param = dict(u=np.array([0, 0, -1, 0, 0, 0]),
                 s=v, fs=FORCE_THRESH_MOVE_DOWN,
                 ft=np.array([0.]*6),
                 kp=KP_DEFAULT,
                 kd=KD_DEFAULT,
                 timeout=TIMEOUT)
    primitive_list.append(("move2contact", param))

# displacement free space
vd = 0.05
for i in range(3):
    move_dir = np.zeros(6)
    for j in range(2):
        dp = (TRANSLATION_DISPLACEMENT_RANGE[1] - TRANSLATION_DISPLACEMENT_RANGE[0])/2
        p = TRANSLATION_DISPLACEMENT_RANGE[0] + dp/2 + j*dp
        move_dir[i] = 1
        param = dict(u=move_dir,
                     s=v, fs=SAFETY_FORCE,
                     ft=np.zeros(6),
                     delta_d=p,
                     kp=KP_DEFAULT,
                     kd=KD_DEFAULT,
                     timeout=TIMEOUT)

        primitive_list.append(("displacement", deepcopy(param)))
        param["u"][i] = -1
        primitive_list.append(("displacement", deepcopy(param)))


for i in range(3):
    move_dir = np.zeros(6)
    for j in range(2):
        dp = (ROTATION_DISPLACEMENT_RANGE[1] - ROTATION_DISPLACEMENT_RANGE[0])/2
        p = ROTATION_DISPLACEMENT_RANGE[0] + dp/2 + j*dp
        move_dir[i+3] = 1
        param = dict(u=move_dir,
                     s=v, fs=SAFETY_TORQUE,
                     ft=np.zeros(6),
                     delta_d=p,
                     kp=KP_DEFAULT,
                     kd=KD_DEFAULT,
                     timeout=TIMEOUT)

        primitive_list.append(("displacement", deepcopy(param)))
        param["u"][i+3] = -1
        primitive_list.append(("displacement", deepcopy(param)))

# free sub space
no_free_actions = len(primitive_list)
free_action_idx = list(range(no_free_actions))  # [0, 1, ... N-1]

# slide/rotate until contact
for i in range(2):
    move_dir = np.zeros(6)
    for j in range(2):
        for k in range(2):
            dv = (SLIDING_SPEED_FACTOR_RANGE[1] - SLIDING_SPEED_FACTOR_RANGE[0])/NO_QUANTIZATION
            v = SLIDING_SPEED_FACTOR_RANGE[0] + dv/2 + j*dv

            dfs = (FORCE_THRESH_RANGE[1] - FORCE_THRESH_RANGE[0]) / 2
            fs = FORCE_THRESH_RANGE[0] + dfs/2 + k*dv
            move_dir[i] = 1

            param = dict(u=move_dir,
                         s=v, fs=fs,
                         ft=np.array([0.]*6),
                         kp=KP_DEFAULT,
                         kd=KD_DEFAULT,
                         timeout=TIMEOUT)
            primitive_list.append(("move2contact", deepcopy(param)))
            param["u"][i] = -1
            primitive_list.append(("move2contact", deepcopy(param)))

for i in range(3):
    move_dir = np.zeros(6)
    for j in range(2):
        for k in range(2):
            dv = (SLIDING_SPEED_FACTOR_RANGE[1] - SLIDING_SPEED_FACTOR_RANGE[0])/NO_QUANTIZATION
            v = SLIDING_SPEED_FACTOR_RANGE[0] + dv/2 + j*dv

            dfs = (FORCE_THRESH_RANGE[1] - FORCE_THRESH_RANGE[0]) / 2
            fs = FORCE_THRESH_RANGE[0] + dfs/2 + k*dv

            move_dir[i+3] = 1
            param = dict(u=move_dir,
                         s=v, fs=fs,
                         ft=np.array([0, 0, -5, 0, 0, 0.]),
                         kp=KP_DEFAULT,
                         kd=KD_DEFAULT,
                         timeout=TIMEOUT)
            primitive_list.append(("move2contact", deepcopy(param)))
            param["u"][i+3] = -1
            primitive_list.append(("move2contact", deepcopy(param)))

# displacement on plane
vd = 0.008
for i in range(2):
    move_dir = np.zeros(6)
    for j in range(2):
        dp = (TRANSLATION_DISPLACEMENT_RANGE[1] - TRANSLATION_DISPLACEMENT_RANGE[0])/2
        p = TRANSLATION_DISPLACEMENT_RANGE[0] + dp/2 + j*dp
        move_dir[i] = 1
        param = dict(u=move_dir,
                     s=v, fs=SAFETY_FORCE,
                     ft=np.array([0, 0, -5, 0, 0, 0.]),
                     delta_d=p,
                     kp=KP_DEFAULT,
                     kd=KD_DEFAULT,
                     timeout=TIMEOUT)

        primitive_list.append(("displacement", deepcopy(param)))
        param["u"][i] = -1
        primitive_list.append(("displacement", deepcopy(param)))

for i in range(3):
    move_dir = np.zeros(6)
    for j in range(2):
        dp = (ROTATION_DISPLACEMENT_RANGE[1] - ROTATION_DISPLACEMENT_RANGE[0])/2
        p = ROTATION_DISPLACEMENT_RANGE[0] + dp/2 + j*dp
        move_dir[i+3] = 1
        param = dict(u=move_dir,
                     s=v, fs=SAFETY_TORQUE,
                     ft=np.array([0, 0, -5, 0, 0, 0.]),
                     delta_d=p,
                     kp=KP_DEFAULT,
                     kd=KD_DEFAULT,
                     timeout=TIMEOUT)

        primitive_list.append(("displacement", deepcopy(param)))
        param["u"][i+3] = -1
        primitive_list.append(("displacement", deepcopy(param)))

# admittance
stiffness =[500, 500, 500, 50, 50, 50.]
damping = [10.]*6
for j in range(NO_QUANTIZATION):
    for k in range(NO_QUANTIZATION):
        dkd = (KD_ADMITTANCE_ROT_RANGE[1] - KD_ADMITTANCE_ROT_RANGE[0])/NO_QUANTIZATION
        kd = KD_ADMITTANCE_ROT_RANGE[0] + dkd/2 + j*dkd

        df = (INSERTION_FORCE_RANGE[1] - INSERTION_FORCE_RANGE[0]) / NO_QUANTIZATION
        f = INSERTION_FORCE_RANGE[0] + df/2 + k*df

        param = dict(kd_adt=np.array([0.]*3 + [kd]*3),
                     ft=np.array([0, 0, -f, 0, 0, 0]),
                     depth_thresh=-HOLE_DEPTH*DEPTH_THRESH,
                     kp=stiffness,
                     kd=damping,
                     timeout=TIMEOUT)
        primitive_list.append(("admittance", param))

no_contact_actions = len(primitive_list)
contact_action_idx = list(range(no_contact_actions))
sub_spaces = [free_action_idx, contact_action_idx]

# ----- train config
env_config = {
    "id": "learn_seq:RealInsertionEnv-v0",
    "primitive_list": primitive_list,
    "hole_pos": hole_pos,
    "hole_quat": hole_quat,
    "hole_depth": HOLE_DEPTH,
    "peg_pos_range": ([-0.05]*3, [0.05]*3),
    "peg_rot_range": ([np.pi - 0.2] + [-0.2]*2, [np.pi + 0.2] + [0.2]*2),
    "initial_pos_range": ([-0.005]*2+ [-0.002], [0.005]*2+ [0.002]),
    "initial_rot_range": ([-5*np.pi/180]*3, [5*np.pi/180]*3),
    "depth_thresh": DEPTH_THRESH,
    "wrapper": StructuredActionSpaceWrapper,
    "wrapper_kwargs": {
        "hole_pos_error_range": ([-1./1000]*2+ [0.], [1./1000]*2+ [0.]),
        "hole_rot_error_range": ([-np.pi/180]*3, [np.pi/180]*3),
        "spaces_idx_list": sub_spaces
    }
}

agent_config = {
    "agent_class": PPOStructuredRealAgent,
    "model_kwargs": {
        "hidden_sizes": None,
        "hidden_nonlinearity": ReLU,
    }
}

sampler_config = {
    "sampler_class": SerialSampler,
    "sampler_kwargs":{
        "batch_T": 512, # no samples per iteration
        "batch_B": 1, # no environments, this will be divided equally to no. parallel envs
        "max_decorrelation_steps": 10
    }
}

algo_config = {
    "algo_class": PPO,
    "algo_kwargs":{
        "discount": 0.99,
        "minibatches": 8,
        "epochs": 10,
        "learning_rate": 1e-3,
        "normalize_advantage": False
    }
}

runner_config = {
    "n_parallel": 1,  # use parallel workers to step environment
    "runner_kwargs": {
        "n_steps": TRAINING_STEP,
        "seed": SEED,
        "log_interval_steps": 2048,
    }
}
