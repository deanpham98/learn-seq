from copy import deepcopy
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from learn_seq.utils.general import load_config, get_exp_path
from learn_seq.utils.gym import append_wrapper
from learn_seq.envs.wrapper import HolePoseWrapper
from learn_seq.utils.rlpyt import gym_make, load_agent_state_dict
from learn_seq.utils.mujoco import quat2vec, integrate_quat
from learn_seq.controller.hybrid import StateRecordHybridController

SIM_EXP_NAME = "prim-rew-5mm-29"
SIM_RUN_NAME = "run_square_10_09_12_52"
REAL_EXP_NAME = "real-prim-rew-1-10"
REAL_RUN_NAME = "checkpoint-square"

def compare_envs(sim_config, real_config):
    sim_env_config = deepcopy(sim_config.env_config)
    # make hole pose same as in real world
    sim_env_config["wrapper_kwargs"]["hole_pos_error_range"] = (np.zeros(3), np.zeros(3))
    sim_env_config["wrapper_kwargs"]["hole_rot_error_range"] = (np.zeros(3), np.zeros(3))
    real_hole_pos = real_config.env_config["hole_pos"].copy()
    real_hole_quat = real_config.env_config["hole_quat"].copy()
    real_hole_pos[2] -= 0.04

    wrapper = HolePoseWrapper
    wrapper_kwargs = dict(
        hole_body_pos=real_hole_pos,
        hole_body_quat=real_hole_quat
    )
    sim_env_config = append_wrapper(sim_env_config,
                    wrapper, wrapper_kwargs=wrapper_kwargs, pos="first")
    sim_env = gym_make(**sim_env_config)

    #
    real_env_config = deepcopy(real_config.env_config)
    real_env_config["wrapper_kwargs"]["hole_pos_error"] = 0
    real_env_config["wrapper_kwargs"]["hole_rot_error"] = 0
    real_env = gym_make(**real_env_config)

    return sim_env, real_env

def evaluate():
    initial_pos = np.array([0.005, 0, 0.01])
    initial_quat = np.array([0, 1., 0, 0])
    r = np.array([0., 1., 0]) * 3*np.pi/180
    initial_quat = integrate_quat(initial_quat, r, 1)

    # load config
    sim_config = load_config(SIM_EXP_NAME)
    real_config = load_config(REAL_EXP_NAME)

    # run path
    sim_exp_path = get_exp_path(SIM_EXP_NAME)
    sim_run_path = os.path.join(sim_exp_path, SIM_RUN_NAME)
    real_exp_path = get_exp_path(REAL_EXP_NAME)
    real_run_path = os.path.join(real_exp_path, REAL_RUN_NAME)

    # sim xml model
    with open(os.path.join(sim_run_path, "params.json"), "r") as f:
        run_config = json.load(f)
        run_id = run_config["run_ID"]
    if "round" in run_id:
        sim_config.env_config["xml_model_name"] = "round_pih.xml"
    if "square" in run_id:
        sim_config.env_config["xml_model_name"] = "square_pih.xml"
    if "triangle" in run_id:
        sim_config.env_config["xml_model_name"] = "triangle_pih.xml"
    sim_config.env_config["controller_class"] = StateRecordHybridController

    # envs
    sim_env, real_env = compare_envs(sim_config, real_config)

    # agent
    sim_agent_class = sim_config.agent_config["agent_class"]
    state_dict = load_agent_state_dict(sim_run_path)
    model_kwargs = sim_config.agent_config["model_kwargs"]
    sim_agent = sim_agent_class(initial_model_state_dict=state_dict,
                        model_kwargs=model_kwargs)
    sim_agent.initialize(sim_env.spaces)


    real_agent_class = real_config.agent_config["agent_class"]
    state_dict = load_agent_state_dict(real_run_path)
    model_kwargs = real_config.agent_config["model_kwargs"]
    real_agent = real_agent_class(initial_model_state_dict=state_dict,
                        model_kwargs=model_kwargs)
    real_agent.initialize(real_env.spaces)

    # real envs
    real_seq = []
    real_obs_seq = []
    obs = real_env.reset_to(initial_pos, initial_quat)

    real_init_pos, real_init_quat = real_env.ros_interface.get_ee_pose(real_env.tf_pos, real_env.tf_quat)

    done = False
    print("real task frame {}".format((real_env.tf_pos, real_env.tf_quat)))
    print("real init pose {}".format((real_init_pos, quat2vec(real_init_quat))))
    real_traj_info = {
        "task_frame": [real_env.tf_pos, real_env.tf_quat],
        "type": [],
        "idx": [],
        "obs": []
        "t": [],
    }
    while not done:
        real_obs_seq.append(obs)
        pa = torch.tensor(np.zeros(6))
        pr = torch.tensor(0.)
        # action = agent.eval_step(torch.tensor(obs, dtype=torch.float32), pa, pr)
        action = real_agent.eval_step(torch.tensor(obs, dtype=torch.float32), pa, pr)
        action = action.action
        a = np.array(action)
        print(real_env.unwrapped.primitive_list[a])
        obs, reward, done, info = real_env.step(a)
        real_seq.append(action.item())
        sim_traj_info["t"].append(real_env.unwrapped.robot_state.get_ros_time())
        sim_traj_info["type"].append(real_env.unwrapped.primitive_list[a][0])
        sim_traj_info["idx"].append(action.item())
        sim_traj_info["obs"].append(obs.copy())

    print("real seq: {}".format(real_seq))
    real_env.ros_interface.stop_record("real_traj.npy")
    np.save("real_traj_info.npy", real_traj_info)
    # real_env.ros_interface.plot_pose()
    # real_env.ros_interface.plot_force()

    # sim
    sim_seq = []
    sim_obs_seq = []
    obs = sim_env.reset_to(real_init_pos, real_init_quat)
    sim_init_pos, sim_init_quat = sim_env.robot_state.get_pose(sim_env.tf_pos, sim_env.tf_quat)
    done = False
    print("sim task frame {}".format((sim_env.tf_pos, sim_env.tf_quat)))
    print("sim init pose {}".format((sim_init_pos, quat2vec(sim_init_quat))))
    sim_env.controller.start_record()
    # sim traj info
    sim_traj_info = {
        "task_frame": [sim_env.tf_pos, sim_env.tf_quat],
        "type": [],
        "idx": [],
        "obs": []
        "t": [],
    }
    while not done:
        sim_obs_seq.append(obs)
        pa = torch.tensor(np.zeros(6))
        pr = torch.tensor(0.)
        # action = agent.eval_step(torch.tensor(obs, dtype=torch.float32), pa, pr)
        action = sim_agent.eval_step(torch.tensor(obs, dtype=torch.float32), pa, pr)
        action = action.action
        a = np.array(action)
        print(sim_env.unwrapped.primitive_list[a])
        obs, reward, done, info = sim_env.step(a)
        sim_seq.append(action.item())
        sim_traj_info["type"].append(sim_env.unwrapped.primitive_list[a][0])
        sim_traj_info["idx"].append(action.item())
        sim_traj_info["t"].append(sim_env.unwrapped.robot_state.get_sim_time())
        sim_traj_info["obs"].append(obs.copy())

    sim_env.controller.stop_record("sim_traj.npy")
    sim_traj_info = np.array(sim_traj_info)
    np.save("sim_traj_info.npy", sim_traj_info)
    # sim_env.controller.plot_pos()
    # sim_env.controller.plot_orient()
    # sim_env.controller.plot_key(["f", "fd"])

    print("sim seq: {}".format(sim_seq))
    real_obs_seq = real_obs_seq[:5]
    # plot
    # fig, ax = plt.subplots(3, 2)
    # for i in range(3):
    #     for j in range(2):
    #         ax[i, j].plot(range(len(sim_obs_seq)), np.array(sim_obs_seq)[:, i+3*j])
    #         ax[i, j].plot(range(len(real_obs_seq)), np.array(real_obs_seq)[:, i+3*j])
    #         ax[i, j].legend(["sim", "real"])
    # fig.suptitle("sim obs vs real obs")
    # plt.show()

if __name__ == '__main__':
    evaluate()
