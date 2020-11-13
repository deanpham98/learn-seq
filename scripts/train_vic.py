import mujoco_py
import gym
import datetime
from learn_seq.utils.rlpyt import CustomTrajInfo, gym_make, load_agent_state_dict
from learn_seq.utils.general import load_config, get_exp_path
from rlpyt.algos.qpg.ddpg import DDPG
from rlpyt.agents.qpg.ddpg_agent import DdpgAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.utils.logging.context import logger_context
from rlpyt.replays.non_sequence.uniform import UniformReplayBuffer
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector

def build_and_train(log_dir, n_parallel=4):
    # environment
    env_kwargs = dict(
        id="learn_seq:SimpleMujocoSlidingEnv-v0",
        fd_range=[5., 5.],      # fix force
        speed_range=[0.01, 0.01],  # fix speed
    )
    # env = gym_make(**env_kwargs)
    # viewer = mujoco_py.MjViewer(env.sim)

    # sampler
    sampler = CpuSampler(
        EnvCls=gym_make,
        env_kwargs=env_kwargs,
        eval_env_kwargs=env_kwargs,
        CollectorCls=CpuResetCollector,
        batch_T=16,
        batch_B=n_parallel,
        max_decorrelation_steps=0,
        eval_n_envs=2,
        eval_max_steps=int(1e4),
        eval_max_trajectories=10,
    )
    # algorithm
    algo = DDPG(
        discount=0.99,
        batch_size=100,
        replay_ratio=100,
        target_update_tau=0.01,
        target_update_interval=1,
        policy_update_interval=1,
        learning_rate=1e-3,
        q_learning_rate=1e-3,
        ReplayBufferCls=UniformReplayBuffer,
        bootstrap_timelimit=False
    )
    # agent
    agent = DdpgAgent(
        model_kwargs=dict(hidden_sizes=[128, 128]),
        q_model_kwargs=dict(hidden_sizes=[128, 128]),
    )
    # runner
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=dict(workers_cpus=list(range(n_parallel))),
        n_steps=1e6,
        log_interval_steps=1e4,
    )
    # run training
    now = datetime.datetime.now()
    id = "sliding" + now.strftime("_%m_%d_%H_%M")
    name = "sliding" + now.strftime("_%m_%d_%H_%M")
    config = dict()
    with logger_context(log_dir, id, name, config,
                        override_prefix=True,
                        snapshot_mode="last"):
        runner.train()

if __name__ == '__main__':
    import os
    import sys
    import argparse
    import imp

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", "-n", type=str, required=True)
    # parser.add_argument("--cpu_offset", "-c", type=int, default=0,
    #                     help="offset for cpus used")
    # # NOTE: gpu only for the agent to sample action and training
    # parser.add_argument("--gpu_idx", "-g", type=int, default=-1,
    #                     help="set to -1 if not use gpu")
    # # parser.add_argument("--checkpoint", type=str, default="",
    # # 		help="continue training from a previous experiment")
    args = parser.parse_args()

    args = vars(args)
    log_dir = get_exp_path(args["exp_name"])
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
    build_and_train(log_dir)
