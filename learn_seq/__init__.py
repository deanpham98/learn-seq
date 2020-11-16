from gym.envs.registration import register

register(
    id='MujocoInsertionEnv-v0',
    entry_point='learn_seq.envs.mujoco.insertion:MujocoInsertionEnv',
)

register(
    id='RealInsertionEnv-v0',
    entry_point='learn_seq.envs.real.insertion:RealInsertionEnv',
)
