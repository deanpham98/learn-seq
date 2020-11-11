import os
import json
import torch
import numpy as np

from copy import deepcopy
import matplotlib.pyplot as plt
from learn_seq.utils.general import read_csv, get_exp_path, get_dirs, load_config
from learn_seq.utils.rlpyt import gym_make, load_agent_state_dict
from learn_seq.utils.gym import append_wrapper
from learn_seq.utils.mujoco import integrate_quat
from learn_seq.envs.wrapper import InitialPoseWrapper, HolePoseWrapper, FixedHolePoseErrorWrapper
from learn_seq.controller.hybrid import StateRecordHybridController

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
    plot_ids = [36, 31, 41, 16]     # insertion_depth, success_rate, loss, reward
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
    # test when there is no hole pose error
    env_config = deepcopy(config.env_config)
    if not isinstance(env_config["wrapper"], list):
        env_config["wrapper"] = [env_config["wrapper"]]
        env_config["wrapper_kwargs"] = [env_config["wrapper_kwargs"]]

    if env_config["wrapper"][0] == HolePoseWrapper:
        rand_hole_idx = 1
    else:
        rand_hole_idx = 0

    env_config["wrapper"][rand_hole_idx] = FixedHolePoseErrorWrapper
    env_config["wrapper_kwargs"][rand_hole_idx] = dict(
        hole_pos_error= 0.,
        hole_rot_error= 0.,
        spaces_idx_list=env_config["wrapper_kwargs"][rand_hole_idx]["spaces_idx_list"]
    )
    # envs.append(gym_make(**env_config))

    # hole pose error = 1
    env_config["wrapper_kwargs"][rand_hole_idx] = dict(
        hole_pos_error= 0.5/1000,
        hole_rot_error= 0.5 * np.pi / 180,
        spaces_idx_list=env_config["wrapper_kwargs"][rand_hole_idx]["spaces_idx_list"]
    )
    envs.append(gym_make(**env_config))

    # generalization hole_pose_error = 2.
    env_config["wrapper_kwargs"][rand_hole_idx] = dict(
        hole_pos_error= 1.5/1000,
        hole_rot_error= 1.5*np.pi/180,
        spaces_idx_list=env_config["wrapper_kwargs"][rand_hole_idx]["spaces_idx_list"]
    )
    envs.append(gym_make(**env_config))

    # robustness
    env_config["initial_pos_range"] = ([-0.002]*2+ [-0.002], [0.002]*2+ [0.002])
    env_config["initial_rot_range"] = ([-2*np.pi/180]*3, [2*np.pi/180]*3)
    env_config["wrapper_kwargs"][rand_hole_idx] = dict(
        hole_pos_error= 0.5/1000,
        hole_rot_error= 0.5*np.pi / 180,
        spaces_idx_list=env_config["wrapper_kwargs"][rand_hole_idx]["spaces_idx_list"]
    )
    envs.append(gym_make(**env_config))
    return envs
# run the agent in a particular environment once
def run_agent_single(agent, env, render=False):
    seq = []
    strat = []
    episode_rew = 0
    done = False
    obs = env.reset()
    while not done:
        pa = torch.tensor(np.zeros(6))
        pr = torch.tensor(0.)
        # action = agent.eval_step(torch.tensor(obs, dtype=torch.float32), pa, pr)
        action = agent.step(torch.tensor(obs, dtype=torch.float32), pa, pr)
        action = action.action
        a = np.array(action)
        # type = env.primitive_list[action.item()][0]
        # if type=="admittance":
        #     env.controller.start_record()
        obs, reward, done, info = env.env.step(a, render=render)
        print(env.primitive_list[a])
        # if type=="admittance":
        #     env.controller.stop_record()
        #     env.controller.plot_pos()
        #     env.controller.plot_key(["p",])
        #     plt.show()
        seq.append(action.item())
        strat.append(env.primitive_list[action.item()][0])
        episode_rew += reward

    return seq, strat, info["success"], episode_rew

# run the agent in a particular env for N episodes
def run_agent(agent, env, eps, render=False):
    no_success = 0
    total_rew = 0
    for i in range(eps):
        seq, strat, suc, rew = run_agent_single(agent, env, render=render)
        # episode info
        print("------")
        print("Episode {}".format(i))
        print("sequence idx: {}".format(seq))
        print("sequence name: {}".format(strat))
        print("success: {}".format(suc))
        print("episode reward: {}".format(rew))
        no_success += int(suc)
        total_rew += rew

    print("success_rate {}".format(float(no_success)/eps))
    print("average rew: {}".format(total_rew/eps))
    env.close()

# initialize the agent with the trained model, generate evaluation envs and
# run the evaluation
def evaluate(run_path_list, config, eval_eps=10, render=False):
    for run_path in run_path_list:
        with open(os.path.join(run_path, "params.json"), "r") as f:
            run_config = json.load(f)
            run_id = run_config["run_ID"]
            config.env_config["xml_model_name"] = run_config["env_config"]["xml_model_name"]
            config.env_config["controller_class"] = StateRecordHybridController

        agent_class = config.agent_config["agent_class"]
        state_dict = load_agent_state_dict(run_path)
        model_kwargs = config.agent_config["model_kwargs"]
        agent = agent_class(initial_model_state_dict=state_dict,
                            model_kwargs=model_kwargs)
        #
        eval_env_list = eval_envs(config)
        agent.initialize(eval_env_list[0].spaces)
        for env in eval_env_list:
            run_agent(agent, env, eps=eval_eps, render=render)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", "-n", type=str, required=True)
    parser.add_argument("--render", "-r", action="store_true")
    parser.add_argument("--eval-eps", "-e", type=int, default=10,
                        help="number of evaluation episode")
    parser.add_argument("--plot-only", action="store_true",
                        help="only plot progress")
    parser.add_argument("--run-name", "-rn", type=str)

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
    if args["plot_only"] == True:
        plot_progress(run_path_list=run_path_list)
    else:
        evaluate(run_path_list, config, args["eval_eps"], args["render"])
