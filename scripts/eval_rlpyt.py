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
from learn_seq.envs.wrapper import InitialPoseWrapper, HolePoseWrapper
from learn_seq.controller.hybrid import StateRecordHybridController
from learn_seq.ros.logger import basic_logger

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

# return the evaluation environments. The policy will be tested on each environment
def eval_envs(config):
    envs = []
    # test when there is no hole pose error
    env_config = deepcopy(config.env_config)
    env_config["wrapper_kwargs"]["hole_pos_error_range"] = (np.zeros(3), np.zeros(3))
    env_config["wrapper_kwargs"]["hole_rot_error_range"] = (np.zeros(3), np.zeros(3))
    envs.append(gym_make(**env_config))

    # fix initial state
    # wrapper = InitialPoseWrapper
    # wrapper_kwargs = dict(
    #     p0 = np.array([0, -0.0, 0.01]),
    #     r0 = np.array([0., 0, 0])
    # )
    # env_config = append_wrapper(config.env_config,
    #         wrapper=wrapper, wrapper_kwargs=wrapper_kwargs)
    # env_config["initial_pos_range"] = ([-0.001]*2+ [-0.], [0.001]*2+ [0.0])
    # env_config["initial_rot_range"] = ([0.]*3, [0.]*3)

    # change hole pose in the mujoco environment
    rot = np.array([0, -60*np.pi/180, 0])
    hole_body_quat = integrate_quat(np.array([1., 0,0 ,0]), rot, 1)
    wrapper = HolePoseWrapper
    wrapper_kwargs = dict(
        hole_body_pos=np.array([0.53, 0.012, 0.5088]),
        hole_body_quat=hole_body_quat
    )
    env_config = append_wrapper(config.env_config, wrapper, wrapper_kwargs, pos="first")
    envs.append(gym_make(**env_config))

    return envs

def real_eval_envs(config):
    envs = []
    # fixed hole pose error, fixed initial state
    env_config = deepcopy(config.env_config)
    env_config["wrapper_kwargs"]["hole_pos_error_range"] = (np.zeros(3), np.zeros(3))
    env_config["wrapper_kwargs"]["hole_rot_error_range"] = (np.zeros(3), np.zeros(3))
    envs.append(gym_make(**env_config))

def run_agent_single(agent, env, render=False):
    seq = []
    strat = []
    episode_rew = 0
    done = False
    obs = env.reset()
    print("intial pos: {}".format(env.ros_interface.get_ee_pose()))
    while not done:
        pa = torch.tensor(np.zeros(6))
        pr = torch.tensor(0.)
        # action = agent.eval_step(torch.tensor(obs, dtype=torch.float32), pa, pr)
        action = agent.step(torch.tensor(obs, dtype=torch.float32), pa, pr)
        action = action.action
        a = np.array(action)
        if render:
            obs, reward, done, info = env.unwrapped.step(a, render=render)
        else:
            obs, reward, done, info = env.unwrapped.step(a)
        seq.append(action.item())
        strat.append(env.primitive_list[action.item()][0])
        episode_rew += reward

    return seq, strat, info["success"], episode_rew

# run the agent in a particular env for N episodes
def run_agent(agent, env, eps, render=False):
    no_success = 0
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
    print("success_rate {}".format(float(no_success)/eps))
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
    parser.add_argument("--log", action="store_true",
                        help="run 1 episode for recording")

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
    elif args["log"]:
        logger = basic_logger(exp_path)
        logger.start_record()
        evaluate(run_path_list, config, 1)
        logger.stop()
    else:
        evaluate(run_path_list, config, args["eval_eps"], args["render"])
