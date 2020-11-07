from copy import deepcopy
import numpy as np
from torch.nn import ReLU
from rlpyt.algos.pg.ppo import PPO
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from learn_seq.envs.wrapper import StructuredActionSpaceWrapper, FixedHolePoseErrorWrapper,\
                    RealControllerCommandWrapper, PrimitiveStateWrapper
from learn_seq.rlpyt.ppo_agent import PPOStructuredRealAgent
from learn_seq.utils.mujoco import mat2quat, mul_quat

# peg transformation matrix
# T_HOLE = np.array([0.998207,0.0595135,-0.00458621,0,0.0594896,-0.998206,-0.00517501,0,-0.00488606,0.00489299,-0.999976, 0,\
#                    0.530483,0.0755677,0.152598,1]).reshape((4, 4)).T
# square with ft sensor
T_HOLE = np.array([0.966059,0.258279,0.00169755,0,0.258283,-0.966054,-0.00327767,0,0.000793383,0.00360494,-0.999993, 0,\
                   0.530719,0.0753696,0.152813,1]).reshape((4, 4)).T
# trianlge with ft sensor
# T_HOLE = np.array([0.997746,0.0651894,-0.0153274,0,0.0649761,-0.997779,-0.0140256,0,-0.016208,0.0129983,-0.999784, 0,\
#                    0.534575,-0.0619296,0.143318,1]).reshape((4, 4)).T

hole_pos = T_HOLE[:3, 3]
hole_rot = T_HOLE[:3, :3]
hole_quat = mat2quat(hole_rot)
# make the z axis point out of the hole
qx = np.array([np.cos(np.pi/2), np.sin(np.pi/2), 0, 0])
hole_quat = mul_quat(hole_quat, qx)

#
SPEED_FACTOR_RANGE = [0.01, 0.02]
SLIDING_SPEED_FACTOR_RANGE = [0.008, 0.015]
# FORCE_THRESH_RANGE = [15, 25]
# TORQUE_THRESH_RANGE = [0.2, 1]
# with ft filter
FORCE_THRESH_RANGE = [8, 15]
TORQUE_THRESH_RANGE = [0.1, 0.5]
TRANSLATION_DISPLACEMENT_RANGE = [0.001, 0.005]
ROTATION_DISPLACEMENT_RANGE = [np.pi/180, 5*np.pi/180]
INSERTION_FORCE_RANGE = [5., 12]
KD_ADMITTANCE_ROT_RANGE = [0.01, 0.15]
ROTATION_TO_TRANSLATION_FACTOR = 8
SAFETY_FORCE = 15.
SAFETY_TORQUE = 0.5
# controller gains
KP_DEFAULT = [2500.] + [1500]*2 + [60.]*2 + [30.]
KD_DEFAULT = [2*np.sqrt(i) for i in KP_DEFAULT]
TIMEOUT = 2.

# no discretization
NO_QUANTIZATION = 2

# TODO change in environment
HOLE_DEPTH = 0.02
GOAL_THRESH = 3e-3
TRAINING_STEP = 1000000
SEED = 18

# ----- Primitive config
primitive_list = []
# move down until contact (16 primitives)
for i in range(NO_QUANTIZATION):
    for j in range(NO_QUANTIZATION):
        dv = (SPEED_FACTOR_RANGE[1] - SPEED_FACTOR_RANGE[0])/NO_QUANTIZATION
        v = SPEED_FACTOR_RANGE[0] + dv/2 + i*dv

        dfs = (FORCE_THRESH_RANGE[1] - FORCE_THRESH_RANGE[0]) / NO_QUANTIZATION
        fs = FORCE_THRESH_RANGE[0] + dfs/2 + j*dv

        param = dict(u=np.array([0, 0, -1, 0, 0, 0]),
                     s=v, fs=fs,
                     ft=np.array([0.]*6),
                     kp=KP_DEFAULT,
                     kd=KD_DEFAULT,
                     timeout=TIMEOUT)
        primitive_list.append(("move2contact", param))

# displacement free space
vd = 0.01
for i in range(3):
    move_dir = np.zeros(6)
    for j in range(NO_QUANTIZATION):
        dp = (TRANSLATION_DISPLACEMENT_RANGE[1] - TRANSLATION_DISPLACEMENT_RANGE[0])/NO_QUANTIZATION
        p = TRANSLATION_DISPLACEMENT_RANGE[0] + dp/2 + j*dp
        move_dir[i] = 1
        param = dict(u=move_dir,
                     s=vd, fs=SAFETY_FORCE,
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
    for j in range(NO_QUANTIZATION):
        dp = (ROTATION_DISPLACEMENT_RANGE[1] - ROTATION_DISPLACEMENT_RANGE[0])/NO_QUANTIZATION
        p = ROTATION_DISPLACEMENT_RANGE[0] + dp/2 + j*dp
        move_dir[i+3] = 1
        param = dict(u=move_dir,
                     s=vd*ROTATION_TO_TRANSLATION_FACTOR, fs=SAFETY_TORQUE,
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
    for j in range(NO_QUANTIZATION):
        for k in range(NO_QUANTIZATION):
            dv = (SLIDING_SPEED_FACTOR_RANGE[1] - SLIDING_SPEED_FACTOR_RANGE[0])/NO_QUANTIZATION
            v = SLIDING_SPEED_FACTOR_RANGE[0] + dv/2 + j*dv

            dfs = (FORCE_THRESH_RANGE[1] - FORCE_THRESH_RANGE[0]) / NO_QUANTIZATION
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
    for j in range(NO_QUANTIZATION):
        for k in range(NO_QUANTIZATION):
            dv = (SLIDING_SPEED_FACTOR_RANGE[1] - SLIDING_SPEED_FACTOR_RANGE[0])/NO_QUANTIZATION
            v = SLIDING_SPEED_FACTOR_RANGE[0] + dv/2 + j*dv

            dfs = (TORQUE_THRESH_RANGE[1] - TORQUE_THRESH_RANGE[0]) / NO_QUANTIZATION
            fs = TORQUE_THRESH_RANGE[0] + dfs/2 + k*dv

            move_dir[i+3] = 1
            param = dict(u=move_dir,
                         s=v*ROTATION_TO_TRANSLATION_FACTOR, fs=fs,
                         ft=np.array([0, 0, -3, 0, 0, 0.]),
                         kp=KP_DEFAULT,
                         kd=KD_DEFAULT,
                         timeout=TIMEOUT)
            primitive_list.append(("move2contact", deepcopy(param)))
            param["u"][i+3] = -1
            primitive_list.append(("move2contact", deepcopy(param)))

# displacement on plane
vd = 0.01
for i in range(2):
    move_dir = np.zeros(6)
    for j in range(NO_QUANTIZATION):
        dp = (TRANSLATION_DISPLACEMENT_RANGE[1] - TRANSLATION_DISPLACEMENT_RANGE[0])/NO_QUANTIZATION
        p = TRANSLATION_DISPLACEMENT_RANGE[0] + dp/2 + j*dp
        move_dir[i] = 1
        param = dict(u=move_dir,
                     s=vd, fs=SAFETY_FORCE,
                     ft=np.array([0, 0, -3, 0, 0, 0.]),
                     delta_d=p,
                     kp=KP_DEFAULT,
                     kd=KD_DEFAULT,
                     timeout=TIMEOUT)

        primitive_list.append(("displacement", deepcopy(param)))
        param["u"][i] = -1
        primitive_list.append(("displacement", deepcopy(param)))

for i in range(3):
    move_dir = np.zeros(6)
    for j in range(NO_QUANTIZATION):
        dp = (ROTATION_DISPLACEMENT_RANGE[1] - ROTATION_DISPLACEMENT_RANGE[0])/NO_QUANTIZATION
        p = ROTATION_DISPLACEMENT_RANGE[0] + dp/2 + j*dp
        move_dir[i+3] = 1
        param = dict(u=move_dir,
                     s=vd*ROTATION_TO_TRANSLATION_FACTOR, fs=SAFETY_TORQUE,
                     ft=np.array([0, 0, -3, 0, 0, 0.]),
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

        param = dict(kd_adt=np.array([0.01]*3 + [kd]*3),
                     ft=np.array([0, 0, -f, 0, 0, 0]),
                     pt=np.array([0, 0, -HOLE_DEPTH]),
                     goal_thresh=GOAL_THRESH,
                     kp=stiffness,
                     kd=damping,
                     timeout=TIMEOUT)
        primitive_list.append(("admittance", param))

#no_contact_actions = len(primitive_list)
no_contact_actions = len(primitive_list) - no_free_actions
contact_action_idx = list(range(no_free_actions, no_free_actions + no_contact_actions))
sub_spaces = [free_action_idx, contact_action_idx]

# ----- train config
wrapper_kwargs = [
{
    "hole_pos_error_range": ([-1./1000]*2+ [-1./1000], [1./1000]*2+ [1./1000]),
    "hole_rot_error_range": ([-1*np.pi/180]*3, [1*np.pi/180]*3),
    "spaces_idx_list": sub_spaces
},  {}, {},]

env_config = {
    "id": "learn_seq:RealInsertionEnv-v0",
    "primitive_list": primitive_list,
    "hole_pos": hole_pos,
    "hole_quat": hole_quat,
    "hole_depth": HOLE_DEPTH,
    "peg_pos_range": ([-0.2]*3, [0.2]*3),
    "peg_rot_range": ([-1]*3, [1]*3),
    "initial_pos_range": ([-0.0]*2+ [-0.0], [0.0]*2+ [0.0]),
    "initial_rot_range": ([-0.*np.pi/180]*3, [0.*np.pi/180]*3),
    "goal_thresh": GOAL_THRESH,
    # "wrapper": StructuredActionSpaceWrapper,

    "wrapper": [StructuredActionSpaceWrapper, RealControllerCommandWrapper, PrimitiveStateWrapper],
    "wrapper_kwargs": wrapper_kwargs
}

agent_config = {
    "agent_class": PPOStructuredRealAgent,
    "model_kwargs": {
        "hidden_sizes": [128, 128],
        # "hidden_nonlinearity": ReLU,
    }
}

sampler_config = {
    "sampler_class": SerialSampler,
    "sampler_kwargs":{
        "batch_T": 128, # no samples per iteration
        "batch_B": 1, # no environments, this will be divided equally to no. parallel envs
        "max_decorrelation_steps": 1
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
        "log_interval_steps": 128,
        "log_traj_window": 20
    }
}
