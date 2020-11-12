import datetime
import numpy as np
import torch

from rlpyt.runners.minibatch_rl import MinibatchRlEval, MinibatchRl
from rlpyt.utils.logging.context import logger_context
from rlpyt.samplers.serial.sampler import SerialSampler

from learn_seq.utils.rlpyt import CustomTrajInfo, gym_make, load_agent_state_dict
from learn_seq.utils.general import load_config, get_exp_path

def build_and_train(config, agent_state_dict=None):
    # ----- env
    env_kwargs = config.env_config

    # ----- sampler
    SamplerCls = config.sampler_config.get("sampler_class", SerialSampler)
    sampler_kwargs = config.sampler_config["sampler_kwargs"]

    sampler = SamplerCls(
        EnvCls=gym_make,
        TrajInfoCls=CustomTrajInfo,
        env_kwargs=env_kwargs,
        eval_env_kwargs=env_kwargs,
        **sampler_kwargs
    ) # NOTE CpuResetCollector

    # ----- agent
    AgentCls = config.agent_config.get("agent_class")
    model_kwargs = config.agent_config["model_kwargs"]
    if agent_state_dict is None:
        agent = AgentCls(model_kwargs=model_kwargs)
    else:
        agent = AgentCls(initial_model_state_dict=agent_state_dict,
                            model_kwargs=model_kwargs)

    # ----- algorithm
    AlgoCls = config.algo_config.get("algo_class")
    algo_kwargs = config.algo_config["algo_kwargs"]
    algo = AlgoCls(**algo_kwargs)

    # ----- runner
    if config.gpu_idx < 0:
        cuda_idx = None
    else:
        cuda_idx = config.gpu_idx
    n_parallel = config.runner_config.get("n_parallel", 1)
    c = config.cpu_offset
    runner_kwargs = config.runner_config["runner_kwargs"]
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=dict(cuda_idx=cuda_idx, workers_cpus=list(range(c, c + n_parallel))),
        **runner_kwargs
    )

    # run training
    now = datetime.datetime.now()
    name = "run_" + now.strftime("%m_%d_%H_%M")
    # save environment configs for reproduce
    # NOTE: ndarray type in env_kwargs cannot be saved
    run_config=dict(env_config=env_kwargs, runner_config=config.runner_config,
                algo_config=config.algo_config, sampler_config=config.sampler_config,
                agent_config=config.agent_config)

    # id (name of data dir)
    now = datetime.datetime.now()
    xml_model_name = env_kwargs.get("xml_model_name", "round")
    id = xml_model_name.split("_")[0] + now.strftime("_%m_%d_%H_%M")
    log_dir = config.log_dir
    with logger_context(log_dir, id, name, run_config,
                        override_prefix=True,  # save in local log dir
                        snapshot_mode="last"):
        runner.train()

if __name__ == '__main__':
    import os
    import sys
    import argparse
    import imp

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", "-n", type=str, required=True)
    parser.add_argument("--env", "-e", type=str, default="round",
                        help="\"round\", \"square\"")
    parser.add_argument("--cpu_offset", "-c", type=int, default=0,
                        help="offset for cpus used")
    # NOTE: gpu only for the agent to sample action and training
    parser.add_argument("--gpu_idx", "-g", type=int, default=-1,
			help="set to -1 if not use gpu")
    parser.add_argument("--checkpoint", type=str, default="",
			help="continue training from a previous experiment")
    args = parser.parse_args()

    args = vars(args)

    # load config
    config = load_config(args["exp_name"])

    # merge parsed args from cmd
    config.log_dir = get_exp_path(args["exp_name"])
    # xml model name
    if args["env"] == "round":
        config.env_config["xml_model_name"] = "round_pih.xml"
    if args["env"] == "square":
        config.env_config["xml_model_name"] = "square_pih.xml"
    if args["env"] == "triangle":
        config.env_config["xml_model_name"] = "triangle_pih.xml"

    # hardware stuffs
    config.gpu_idx = args["gpu_idx"]
    config.cpu_offset = args["cpu_offset"]

    # train from checkpoint
    if args["checkpoint"]!="":
        exp_path = get_exp_path(args["exp_name"])
        run_path = os.path.join(exp_path, args["checkpoint"])
        agent_state_dict = load_agent_state_dict(run_path)
    else:
        agent_state_dict = None

    build_and_train(config=config, agent_state_dict=agent_state_dict)
