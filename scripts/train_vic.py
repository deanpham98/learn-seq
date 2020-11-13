import gym
from learn_seq.utils.rlpyt import CustomTrajInfo, gym_make, load_agent_state_dict
from learn_seq.utils.general import load_config, get_exp_path

def build_and_train():
    env_config = dict(
    )

if __name__ == '__main__':
    import os
    import sys
    import argparse
    import imp

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", "-n", type=str, required=True)
    parser.add_argument("--cpu_offset", "-c", type=int, default=0,
                        help="offset for cpus used")
    # NOTE: gpu only for the agent to sample action and training
    parser.add_argument("--gpu_idx", "-g", type=int, default=-1,
                        help="set to -1 if not use gpu")
    # parser.add_argument("--checkpoint", type=str, default="",
    # 		help="continue training from a previous experiment")
    args = parser.parse_args()

    args = vars(args)

    # load config
    # config = load_config(args["exp_name"])

    # merge parsed args from cmd
    # config.log_dir = get_exp_path(args["exp_name"])
    # config.gpu_idx = args["gpu_idx"]
    # config.cpu_offset = args["cpu_offset"]

    # # train from checkpoint
    # if args["checkpoint"] != "":
    #     exp_path = get_exp_path(args["exp_name"])
    #     run_path = os.path.join(exp_path, args["checkpoint"])
    #     agent_state_dict = load_agent_state_dict(run_path)
    # else:
    #     agent_state_dict = None

    # build_and_train(config=config, agent_state_dict=agent_state_dict)
    build_and_train()
