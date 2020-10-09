from copy import deepcopy
import numpy as np
from torch.nn import ReLU
from rlpyt.algos.pg.ppo import PPO
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from learn_seq.envs.wrapper import StructuredActionSpaceWrapper
from learn_seq.rlpyt.ppo_agent import PPOStructuredInsertionAgent
from learn_seq.controller.hybrid import StateRecordHybridController

#----- primitive params
NO_QUANTIZATION = 2
SPEED_FACTOR_RANGE = [0.01, 0.02]
SLIDING_SPEED_FACTOR_RANGE = [0.008, 0.015]
# FORCE_THRESH_RANGE = [15, 25]
# TORQUE_THRESH_RANGE = [0.2, 1]
# with ft filter
FORCE_THRESH_RANGE = [8, 15]
TORQUE_THRESH_RANGE = [0.1, 0.5]
TRANSLATION_DISPLACEMENT_RANGE = [0.001, 0.005]
ROTATION_DISPLACEMENT_RANGE = [np.pi/180, 5*np.pi/180]
INSERTION_FORCE_RANGE = [10., 20]
KD_ADMITTANCE_ROT_RANGE = [0.01, 0.15]
ROTATION_TO_TRANSLATION_FACTOR = 8
SAFETY_FORCE = 15.
SAFETY_TORQUE = 2.
# controller gains
KP_DEFAULT = [1000.]*3 + [60.]*3
KD_DEFAULT = [2*np.sqrt(i) for i in KP_DEFAULT]
TIMEOUT = 2.

# TODO change in environment
HOLE_DEPTH = 0.02
DEPTH_THRESH = 0.95 # the goal is achieved when insertion depth > HOLE_DEPTH * DEPTH_THRESH
TRAINING_STEP = 1000000
SEED = 18

# ----- Primitive config
primitive_list = []
# move down until contact (16 primitives)
for i in range(4):
    for j in range(4):
        dv = (SPEED_FACTOR_RANGE[1] - SPEED_FACTOR_RANGE[0])/4
        v = SPEED_FACTOR_RANGE[0] + dv/2 + i*dv

        dfs = (FORCE_THRESH_RANGE[1] - FORCE_THRESH_RANGE[0]) / 4
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
for i in range(3):  # x, y, z translation
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


for i in range(3):  # x, y, z rotation
    move_dir = np.zeros(6)
    for j in range(NO_QUANTIZATION):
        dp = (ROTATION_DISPLACEMENT_RANGE[1] - ROTATION_DISPLACEMENT_RANGE[0])/NO_QUANTIZATION
        p = ROTATION_DISPLACEMENT_RANGE[0] + dp/2 + j*dp
        move_dir[i+3] = 1
        param = dict(u=move_dir,
                     s=vd*ROTATION_TO_TRANSLATION_FACTOR, fs=fs,
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
for i in range(2):  # x, y translation
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

for i in range(3):  # x, y, z rotation
    move_dir = np.zeros(6)
    for j in range(NO_QUANTIZATION):
        for k in range(NO_QUANTIZATION):
            dv = (SLIDING_SPEED_FACTOR_RANGE[1] - SLIDING_SPEED_FACTOR_RANGE[0])/NO_QUANTIZATION
            v = SLIDING_SPEED_FACTOR_RANGE[0] + dv/2 + j*dv

            dfs = (FORCE_THRESH_RANGE[1] - FORCE_THRESH_RANGE[0]) / NO_QUANTIZATION
            fs = FORCE_THRESH_RANGE[0] + dfs/2 + k*dv

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
for i in range(2):  # x, y tranlation
    move_dir = np.zeros(6)
    for j in range(NO_QUANTIZATION):
        dp = (TRANSLATION_DISPLACEMENT_RANGE[1] - TRANSLATION_DISPLACEMENT_RANGE[0])/NO_QUANTIZATION
        p = TRANSLATION_DISPLACEMENT_RANGE[0] + dp/2 + j*dp
        move_dir[i] = 1
        param = dict(u=move_dir,
                     s=vd, fs=fs,
                     ft=np.array([0, 0, -3, 0, 0, 0.]),
                     delta_d=p,
                     kp=KP_DEFAULT,
                     kd=KD_DEFAULT,
                     timeout=TIMEOUT)

        primitive_list.append(("displacement", deepcopy(param)))
        param["u"][i] = -1
        primitive_list.append(("displacement", deepcopy(param)))

for i in range(3):  # x, y, z rotation
    move_dir = np.zeros(6)
    for j in range(NO_QUANTIZATION):
        dp = (ROTATION_DISPLACEMENT_RANGE[1] - ROTATION_DISPLACEMENT_RANGE[0])/NO_QUANTIZATION
        p = ROTATION_DISPLACEMENT_RANGE[0] + dp/2 + j*dp
        move_dir[i+3] = 1
        param = dict(u=move_dir,
                     s=vd*ROTATION_TO_TRANSLATION_FACTOR, fs=fs,
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
for j in range(4):
    for k in range(4):
        dkd = (KD_ADMITTANCE_ROT_RANGE[1] - KD_ADMITTANCE_ROT_RANGE[0])/4
        kd = KD_ADMITTANCE_ROT_RANGE[0] + dkd/2 + j*dkd

        df = (INSERTION_FORCE_RANGE[1] - INSERTION_FORCE_RANGE[0]) / 4
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
    "id": "learn_seq:MujocoInsertionEnv-v0",
    "primitive_list": primitive_list,
    "peg_pos_range": ([-0.05]*3, [0.05]*3),
    "peg_rot_range": ([np.pi - 0.2] + [-0.2]*2, [np.pi + 0.2] + [0.2]*2),
    "initial_pos_range": ([-0.005]*2+ [-0.002], [0.005]*2+ [0.002]),
    "initial_rot_range": ([-5*np.pi/180]*3, [5*np.pi/180]*3),
    "depth_thresh": DEPTH_THRESH,
    "controller_class": StateRecordHybridController,
    "wrapper": StructuredActionSpaceWrapper,
    "wrapper_kwargs": {
        "hole_pos_error_range": ([-1./1000]*2+ [0.], [1./1000]*2+ [0.]),
        "hole_rot_error_range": ([-np.pi/180]*3, [np.pi/180]*3),
        "spaces_idx_list": sub_spaces
    }
}

agent_config = {
    "agent_class": PPOStructuredInsertionAgent,
    "model_kwargs": {
        "hidden_sizes": None,
        # "hidden_nonlinearity": ReLU,
    }
}

sampler_config = {
    "sampler_class": CpuSampler,
    "sampler_kwargs":{
        "batch_T": 128, # no samples per iteration
        "batch_B": 16, # no environments, this will be divided equally to no. parallel envs
        "max_decorrelation_steps": 10
    }
}

algo_config = {
    "algo_class": PPO,
    "algo_kwargs":{
        "discount": 0.99,
        "minibatches": 32,
        "epochs": 10,
        "learning_rate": 1e-3,
        "normalize_advantage": False
    }
}

runner_config = {
    "n_parallel": 4,  # number of CPU cores used for paralellism
    "runner_kwargs": {
        "n_steps": TRAINING_STEP,
        "seed": SEED,
        "log_interval_steps": 2048,
    }
}
