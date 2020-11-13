import os
import json
import torch
import numpy as np
import mujoco_py
import matplotlib.pyplot as plt
from copy import deepcopy
import matplotlib.pyplot as plt
from learn_seq.utils.general import read_csv, get_exp_path, get_dirs, load_config
from learn_seq.utils.rlpyt import gym_make, load_agent_state_dict
from learn_seq.utils.gym import append_wrapper
from learn_seq.utils.mujoco import integrate_quat, quat_error
from learn_seq.controller.hybrid import StateRecordHybridController
from learn_seq.ros.logger import basic_logger
from rlpyt.agents.qpg.ddpg_agent import DdpgAgent

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
    plot_ids = [26, ]     # insertion_depth, success_rate, loss, reward
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

    # for i in range(len(plot_ids)):
    #     axs[i].legend(legend)
    plt.show()


def evaluate(run_path_list, eval_eps=1, render=False):
    for run_path in run_path_list:
        # load agent
        state_dict = load_agent_state_dict(run_path)
        agent = DdpgAgent(
            initial_model_state_dict= state_dict["model"],
            model_kwargs=dict(hidden_sizes=[128, 128]),
            q_model_kwargs=dict(hidden_sizes=[128, 128]),
            action_std=0.01
        )
        # environment
        env_kwargs = dict(
            id="learn_seq:SimpleMujocoSlidingEnv-v0",
            fd_range=[5., 5.],      # fix force
            speed_range=[0.01, 0.01],  # fix speed
        )
        env = gym_make(**env_kwargs)
        agent.initialize(env.spaces)
        # render
        if render:
            viewer = mujoco_py.MjViewer(env.sim)
            viewer.cam.distance = 0.8406425480019979
            viewer.cam.lookat[:] = [0.49437223, 0.03581988, 0.29160004]
            viewer.cam.elevation = -10.5
            viewer.cam.azimuth = 141.6
        for i in range(eval_eps):
            obs = env.reset()
            done = False
            env.controller.start_record()
            while not done:
                pa = torch.tensor(np.zeros(6))
                pr = torch.tensor(0)
                action = agent.step(torch.tensor(obs, dtype=torch.float32), pa, pr)
                a = np.array(action.action)
                obs, reward, done, info = env.env.step(a)
                if render:
                    viewer.render()
            env.controller.stop_record()
            env.controller.plot_key(["kp"])
            env.controller.plot_key(["f", "fd"])
            plt.show()

if __name__ == '__main__':
    import argparse

    # torch.random.seed()

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", "-n", type=str, required=True)
    parser.add_argument("--render", "-r", action="store_true")
    parser.add_argument("--eval-eps", "-e", type=int, default=1,
                        help="number of evaluation episode")
    parser.add_argument("--plot-only", action="store_true",
                        help="only plot progress")
    parser.add_argument("--run-name", "-rn", type=str)
    parser.add_argument("--log", action="store_true",
                        help="run 1 episode for recording")
    parser.add_argument("--eval-seq", action="store_true",
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

    # config = load_config(args["exp_name"])
    if args["plot_only"] == True:
        plot_progress(run_path_list=run_path_list)

    else:
        evaluate(run_path_list, args["eval_eps"], args["render"])
