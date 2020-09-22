import os
import json
import matplotlib.pyplot as plt
from learn_seq.utils.general import read_csv, get_exp_path, get_dirs

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
    run_id = []
    progress_data = []
    x_idx = 4
    plot_ids = [36, 39, 41, 16]
    axs = []
    for i in plot_ids:
        fig, ax = plt.subplots()
        axs.append(ax)
    legend = []
    for run_dir in run_path_list:
        with open(os.path.join(run_dir, "params.json"), "r") as f:
            config = json.load(f)
            run_id.append(config["run_ID"])
        legend.append(run_id)

        # train progress
        progress_path = os.path.join(run_dir, "progress.csv")
        progress_data = read_csv(progress_path)
        # print_key(progress_data)
        # plot
        for i, idx in enumerate(plot_ids):
            plot(x_idx, idx, progress_data, axs[i])

    for i in range(len(plot_ids)):
        axs[i].legend(legend)
    plt.show()

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

    if args["plot_only"] == True:
        plot_progress(run_path_list=run_path_list)
    # else:
    #     evaluate_exp(exp_dir, eval_eps=params["eval_eps"], render=params["render"],
    #                  run_name=run_name)
