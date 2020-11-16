import json
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch

from learn_seq.envs.wrapper import (FixedHolePoseErrorWrapper,
                                    FixedInitialPoseWrapper,
                                    InitialPoseWrapper)
from learn_seq.ros.logger import basic_logger
from learn_seq.utils.general import (get_dirs, get_exp_path, load_config,
                                     read_csv)
from learn_seq.utils.gym import append_wrapper
from learn_seq.utils.mujoco import integrate_quat, quat_error
from learn_seq.utils.rlpyt import gym_make, load_agent_state_dict


def plot(x_idx, y_idx, data, ax=None):
    key_list = list(data.keys())
    val = list(data.values())
    if ax is None:
        fig, ax = plt.subplots()
    x = val[x_idx]
    y = val[y_idx]
    ax.plot(x, y)
    ax.set_xlabel(key_list[x_idx])
    ax.set_ylabel(key_list[y_idx])
    return ax


def print_key(d):
    for i, k in enumerate(d.keys()):
        print("{}\t{}".format(i, k))


def plot_progress(run_path_list):
    progress_data = []
    x_idx = 4
    plot_ids = [36, 31, 41, 16]  # insertion_depth, success_rate, loss, reward
    axs = []
    for i in plot_ids:
        fig, ax = plt.subplots()
        axs.append(ax)
    legend = []
    for run_dir in run_path_list:
        with open(os.path.join(run_dir, "params.json"), "r") as f:
            config = json.load(f)
            legend.append(config["run_ID"])

        # train progress
        progress_path = os.path.join(run_dir, "progress.csv")
        progress_data = read_csv(progress_path)
        print_key(progress_data)
        # plot
        for i, idx in enumerate(plot_ids):
            plot(x_idx, idx, progress_data, axs[i])

    for i in range(len(plot_ids)):
        axs[i].legend(legend)
    plt.show()


def eval_envs(config):
    envs = []
    # test normal: fixed initial position, 1mm hole pose error
    env_config = deepcopy(config.env_config)
    if not isinstance(env_config["wrapper"], list):
        env_config["wrapper"] = [
            env_config["wrapper"],
        ]
        env_config["wrapper_kwargs"] = [
            env_config["wrapper_kwargs"],
        ]
    env_config["initial_pos_range"] = ([0.] * 3, [0.] * 3)
    env_config["initial_rot_range"] = ([0.] * 3, [0.] * 3)

    env_config["wrapper"][0] = FixedHolePoseErrorWrapper
    env_config["wrapper_kwargs"][0] = dict(
        hole_pos_error=0.5 / 1000,
        hole_rot_error=0.5 * np.pi / 180,
        spaces_idx_list=env_config["wrapper_kwargs"][0]["spaces_idx_list"])
    envs.append(gym_make(**env_config))

    # test generalization: fixed init position, 2mm hole pose error
    env_config["wrapper_kwargs"][0] = dict(
        hole_pos_error=0.0015,
        hole_rot_error=1.5 * np.pi / 180,
        spaces_idx_list=env_config["wrapper_kwargs"][0]["spaces_idx_list"])
    envs.append(gym_make(**env_config))

    # test robust: random initial position, 1mm hole pose error
    env_config["initial_pos_range"] = ([-0.001] * 3, [0.001] * 3)
    env_config["initial_rot_range"] = ([-np.pi / 180] * 3, [np.pi / 180] * 3)

    env_config["wrapper_kwargs"][0] = dict(
        hole_pos_error=0.5 / 1000,
        hole_rot_error=0.5 * np.pi / 180,
        spaces_idx_list=env_config["wrapper_kwargs"][0]["spaces_idx_list"])

    wrapper = FixedInitialPoseWrapper
    wrapper_kwargs = dict(dp=1. / 1000, dr=1. * np.pi / 180)

    env_config = append_wrapper(env_config,
                                wrapper=wrapper,
                                wrapper_kwargs=wrapper_kwargs)
    envs.append(gym_make(**env_config))

    return envs


# fixed initial pose
def real_eval_envs(config):
    # p0 = np.array([0, 0.001, 0.01])
    # r0 = np.array([1*np.pi/180, 0, 0])
    p0 = np.array([0.001, 0.001, 0.01])
    r0 = np.array([-0 * np.pi / 180, 0 * np.pi / 180, -0 * np.pi / 180])
    envs = []

    # test success rate
    env_config = deepcopy(config.env_config)
    env_config["wrapper"] = FixedHolePoseErrorWrapper
    env_config["wrapper_kwargs"] = dict(
        hole_pos_error=0.000,
        hole_rot_error=0 * np.pi / 180,
        spaces_idx_list=env_config["wrapper_kwargs"]["spaces_idx_list"])

    wrapper = InitialPoseWrapper
    wrapper_kwargs = dict(p0=p0, r0=r0)
    env_config = append_wrapper(env_config,
                                wrapper=wrapper,
                                wrapper_kwargs=wrapper_kwargs)
    env_config["initial_pos_range"] = ([-0.0] * 2 + [-0.], [0.0] * 2 + [0.0])
    env_config["initial_rot_range"] = ([0.] * 3, [0.] * 3)
    envs.append(gym_make(**env_config))
    return envs


def run_agent_single(agent, env, p=None, q=None, render=False, real=False):
    seq = []
    strat = []
    episode_rew = 0
    done = False
    if p is not None and q is not None:
        obs = env.reset_to(p, q)
    else:
        obs = env.reset()
    tf_pos, tf_quat = env.get_task_frame()
    if real:
        init_pos, init_quat = env.unwrapped.ros_interface.get_ee_pose(
            frame_pos=tf_pos, frame_quat=tf_quat)
    else:
        init_pos, init_quat = env.unwrapped.robot_state.get_pose(
            frame_pos=tf_pos, frame_quat=tf_quat)

    print("intial pos: {}".format(init_pos))
    print("intial pos: {}".format(init_quat))
    print("hole pos error: {}".format((env.tf_pos - env.hole_pos) * 1000))
    print("hole rot error: {}".format(180 / np.pi *
                                      quat_error(env.hole_quat, env.tf_quat)))
    while not done:
        pa = torch.tensor(np.zeros(6))
        pr = torch.tensor(0.)
        # action = agent.eval_step(torch.tensor(obs, dtype=torch.float32), pa, pr)
        action = agent.step(torch.tensor(obs, dtype=torch.float32), pa, pr)
        action = action.action
        a = np.array(action)
        print(env.unwrapped.primitive_list[a])
        if render:
            obs, reward, done, info = env.env.step(a, render=render)
        else:
            obs, reward, done, info = env.env.step(a)
        seq.append(action.item())
        strat.append(env.primitive_list[action.item()][0])
        episode_rew += reward
        if info["mp_status"] == 1:
            status = "success"
        else:
            status = "fail"
        print("T: {}, status: {}".format(info["mp_time"], status))
    # env.controller.stop_record("test.npy")

    return seq, strat, info["success"], info["eps_time"], info[
        "insert_depth"], episode_rew


# run the agent in a particular env for N episodes
def run_agent(agent, env, eps, render=False):
    print("initial pos range {}".format(env.initial_pos_range))
    print("initial rot range {}".format(env.initial_rot_range))
    print("initial pos mean {}".format(env.initial_pos_mean))
    print("initial rot mean {}".format(env.initial_rot_mean))
    no_success = 0
    total_rew = 0
    t_exec_list = []
    for i in range(eps):
        print("------")
        print("Episode {}".format(i))
        seq, strat, suc, t_exec, depth, rew = run_agent_single(agent,
                                                               env,
                                                               render=render)
        # episode info
        print("sequence idx: {}".format(seq))
        print("sequence name: {}".format(strat))
        print("success: {}".format(suc))
        print("execution time: {}".format(t_exec))
        print("insertion depth {}".format(depth))
        print("episode reward: {}".format(rew))
        no_success += int(suc)
        total_rew += rew
        if suc:
            t_exec_list.append(t_exec)

    print("success_rate {}".format(float(no_success) / eps))
    print("mean success time {}".format(np.mean(t_exec_list)))
    print("std success time {}".format(np.std(t_exec_list)))
    env.close()


# initialize the agent with the trained model, generate evaluation envs and
# run the evaluation
def evaluate(run_path_list, config, eval_eps=10, render=False):
    for run_path in run_path_list:
        with open(os.path.join(run_path, "params.json"), "r") as f:
            run_config = json.load(f)
            run_id = run_config["run_ID"]
        if "Real" not in config.env_config["id"]:
            if "round" in run_id:
                config.env_config["xml_model_name"] = "round_pih.xml"
            if "square" in run_id:
                config.env_config["xml_model_name"] = "square_pih.xml"
            if "triangle" in run_id:
                config.env_config["xml_model_name"] = "triangle_pih.xml"

        agent_class = config.agent_config["agent_class"]
        state_dict = load_agent_state_dict(run_path)
        model_kwargs = config.agent_config["model_kwargs"]
        agent = agent_class(initial_model_state_dict=state_dict,
                            model_kwargs=model_kwargs)

        eval_env_list = eval_envs(config)
        # eval_env_list = real_eval_envs(config)
        agent.initialize(eval_env_list[0].spaces)
        for env in eval_env_list:
            run_agent(agent, env, eps=eval_eps, render=render)


def evaluate_sequence(run_path_list, config, render=False):
    # test different initial pose
    dp = 4. / 1000
    dr = 4 * np.pi / 180
    p0 = [
        np.array([dp - 0.002, 0, 0.01]),
        np.array([-dp, 0, 0.01]),
        np.array([0, dp, 0.01]),
        np.array([-0, -dp, 0.01])
    ]

    r0 = [
        np.array([dr, 0, 0.]),
        np.array([-dr, 0, 0.]),
        np.array([0, dr, 0.]),
        np.array([-0, -dr, 0.])
    ]

    for run_path in run_path_list:
        with open(os.path.join(run_path, "params.json"), "r") as f:
            run_config = json.load(f)
            run_id = run_config["run_ID"]
        if "Real" not in config.env_config["id"]:
            if "round" in run_id:
                config.env_config["xml_model_name"] = "round_pih.xml"
            if "square" in run_id:
                config.env_config["xml_model_name"] = "square_pih.xml"

        agent_class = config.agent_config["agent_class"]
        state_dict = load_agent_state_dict(run_path)
        model_kwargs = config.agent_config["model_kwargs"]
        agent = agent_class(initial_model_state_dict=state_dict,
                            model_kwargs=model_kwargs)
        #
        if "Real" in config.env_config["id"]:
            eval_env_list = real_eval_envs(config)
        else:
            eval_env_list = eval_envs(config)
        print(eval_env_list[0].spaces)
        agent.initialize(eval_env_list[0].spaces)
        env = eval_env_list[0]
        for p in p0:
            for r in r0:
                env.unwrapped.initial_pos_mean = p
                env.unwrapped.initial_rot_mean = r

                print("initial pos range {}".format(env.initial_pos_range))
                print("initial rot range {}".format(env.initial_rot_range))
                print("initial pos mean {}".format(env.initial_pos_mean))
                print("initial rot mean {}".format(env.initial_rot_mean))
                for i in range(2):
                    print("------")
                    print("Episode {}".format(i))

                    seq, strat, suc, t_exec, depth, rew = run_agent_single(
                        agent, env, render=render)
                    # episode info
                    print("sequence idx: {}".format(seq))
                    print("sequence name: {}".format(strat))
                    print("success: {}".format(suc))
                    print("insertion depth {}".format(depth))
                    print("execution time: {}".format(t_exec))
                    print("episode reward: {}".format(rew))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", "-n", type=str, required=True)
    parser.add_argument("--render", "-r", action="store_true")
    parser.add_argument("--eval-eps",
                        "-e",
                        type=int,
                        default=10,
                        help="number of evaluation episode")
    parser.add_argument("--plot-only",
                        action="store_true",
                        help="only plot training performance")
    parser.add_argument("--run-name", "-rn", type=str)
    parser.add_argument("--log",
                        action="store_true",
                        help="run 1 episode for recording")
    parser.add_argument("--eval-seq",
                        action="store_true",
                        help="find various sequence")

    args = parser.parse_args()
    args = vars(args)

    # get experiment params
    exp_path = get_exp_path(exp_name=args["exp_name"])
    run_name = args.get("run_name", None)
    if run_name is None:
        run_path_list = get_dirs(exp_path)
    else:
        run_path_list = [os.path.join(exp_path, run_name)]

    config = load_config(args["exp_name"])
    # plot training performance
    if args["plot_only"] is True:
        plot_progress(run_path_list=run_path_list)
    # run once and record the trajectory
    elif args["log"]:
        logger = basic_logger(exp_path)
        logger.start_record()
        evaluate(run_path_list, config, 1)
        logger.stop()
    # eval sequence to get different trajectories
    elif args["eval_seq"]:
        evaluate_sequence(run_path_list, config, args["render"])
    # normal evaluation
    else:
        evaluate(run_path_list, config, args["eval_eps"], args["render"])
