"""Example to show how to set mujoco model and use multiple wrappers"""
from copy import deepcopy
import numpy as np
from torch.nn import ReLU
from rlpyt.algos.pg.ppo import PPO
from rlpyt.samplers.serial.sampler import SerialSampler
from learn_seq.envs.wrapper import StructuredActionSpaceWrapper, SetMujocoModelWrapper
from learn_seq.rlpyt.ppo_agent import PPOStructuredInsertionAgent
from learn_seq.controller.hybrid import StateRecordHybridController

#
SPEED_FACTOR_RANGE = [0.01, 0.05]
FORCE_THRESH_RANGE = [5, 10]
TORQUE_THRESH_RANGE = [0.2, 1]
TRANSLATION_DISPLACEMENT_RANGE = [0.001, 0.005]
ROTATION_DISPLACEMENT_RANGE = [np.pi/180, 5*np.pi/180]
INSERTION_FORCE_RANGE = [10., 25]
KD_ADMITTANCE_ROT_RANGE = [0.01, 0.15]
SAFETY_FORCE = 15.
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

# mujoco model config (None for default value)
TIMESTEP = 0.002
MASS = 0.5
FRICTION = [1, 0.05, 0.001]
JOINT_DAMPING = [10]*7
CLEARANCE = 0.0001

# ----- Primitive config
primitive_list = []
# move down until contact
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
vd = 0.05
for i in range(3):
    move_dir = np.zeros(6)
    for j in range(2):
        dp = (TRANSLATION_DISPLACEMENT_RANGE[1] - TRANSLATION_DISPLACEMENT_RANGE[0])/2
        p = TRANSLATION_DISPLACEMENT_RANGE[0] + dp/2 + j*dp
        move_dir[i] = 1
        param = dict(u=move_dir,
                     s=v, fs=fs,
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
                     s=v, fs=fs,
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
            dv = (SPEED_FACTOR_RANGE[1] - SPEED_FACTOR_RANGE[0])/NO_QUANTIZATION
            v = SPEED_FACTOR_RANGE[0] + dv/2 + j*dv

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
            dv = (SPEED_FACTOR_RANGE[1] - SPEED_FACTOR_RANGE[0])/NO_QUANTIZATION
            v = SPEED_FACTOR_RANGE[0] + dv/2 + j*dv

            dfs = (FORCE_THRESH_RANGE[1] - FORCE_THRESH_RANGE[0]) / 2
            fs = FORCE_THRESH_RANGE[0] + dfs/2 + k*dv

            move_dir[i+3] = 1
            param = dict(u=move_dir,
                         s=v, fs=fs,
                         ft=np.array([0, 0, -3, 0, 0, 0.]),
                         kp=KP_DEFAULT,
                         kd=KD_DEFAULT,
                         timeout=TIMEOUT)
            primitive_list.append(("move2contact", deepcopy(param)))
            param["u"][i+3] = -1
            primitive_list.append(("move2contact", deepcopy(param)))

# displacement on plane
vd = 0.02
for i in range(2):
    move_dir = np.zeros(6)
    for j in range(2):
        dp = (TRANSLATION_DISPLACEMENT_RANGE[1] - TRANSLATION_DISPLACEMENT_RANGE[0])/2
        p = TRANSLATION_DISPLACEMENT_RANGE[0] + dp/2 + j*dp
        move_dir[i] = 1
        param = dict(u=move_dir,
                     s=v, fs=fs,
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
    for j in range(2):
        dp = (ROTATION_DISPLACEMENT_RANGE[1] - ROTATION_DISPLACEMENT_RANGE[0])/2
        p = ROTATION_DISPLACEMENT_RANGE[0] + dp/2 + j*dp
        move_dir[i+3] = 1
        param = dict(u=move_dir,
                     s=v, fs=fs,
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

        param = dict(kd_adt=np.array([0.]*3 + [kd]*3),
                     ft=np.array([0, 0, -f, 0, 0, 0]),
                     depth_thresh=-HOLE_DEPTH*DEPTH_THRESH,
                     kp=KP_DEFAULT,
                     kd=KD_DEFAULT,
                     timeout=TIMEOUT)
        primitive_list.append(("admittance", param))

no_contact_actions = len(primitive_list) - no_free_actions
contact_action_idx = list(range(no_free_actions, no_free_actions+no_contact_actions))
sub_spaces = [free_action_idx, contact_action_idx]

# ----- train config
wrapper_kwargs = []
config_dict = dict(
    timestep=TIMESTEP,
    mass=MASS,
    friction=FRICTION,
    joint_damping=JOINT_DAMPING,
    clearance=CLEARANCE
)
mujoco_model_kwargs = dict(config_dict=config_dict)
action_space_kwargs = dict(
    hole_pos_error_range=([-1./1000]*2+ [0.], [1./1000]*2+ [0.]),
    hole_rot_error_range=([-np.pi/180]*3, [np.pi/180]*3),
    spaces_idx_list=sub_spaces
)
wrapper_kwargs.append(mujoco_model_kwargs)
wrapper_kwargs.append(action_space_kwargs)

env_config = {
    "id": "learn_seq:MujocoInsertionEnv-v0",
    "primitive_list": primitive_list,
    "peg_pos_range": ([-0.05]*3, [0.05]*3),
    "peg_rot_range": ([np.pi - 0.2] + [-0.2]*2, [np.pi + 0.2] + [0.2]*2),
    "initial_pos_range": ([-0.005]*2+ [-0.002], [0.005]*2+ [0.002]),
    "initial_rot_range": ([-5*np.pi/180]*3, [5*np.pi/180]*3),
    "depth_thresh": DEPTH_THRESH,
    "controller_class": StateRecordHybridController,
    "wrapper": [SetMujocoModelWrapper, StructuredActionSpaceWrapper],
    "wrapper_kwargs": wrapper_kwargs,
}

agent_config = {
    "agent_class": PPOStructuredInsertionAgent,
    "model_kwargs": {
        "hidden_sizes": None,
        # "hidden_nonlinearity": ReLU,
    }
}

sampler_config = {
    "sampler_class": SerialSampler,
    "sampler_kwargs":{
        "batch_T": 256, # no samples per iteration
        "batch_B": 1, # no environments, this will be divided equally to no. parallel envs
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
    "n_parallel": 4,  # use parallel workers to step environment
    "runner_kwargs": {
        "n_steps": TRAINING_STEP,
        "seed": 15,
        "log_interval_steps": 2048,
    }
}
