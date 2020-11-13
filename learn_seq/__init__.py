from gym.envs.registration import register

register(
    id='MujocoInsertionEnv-v0',
    entry_point='learn_seq.envs.mujoco.insertion:MujocoInsertionEnv',
)

# register(
#     id='RealInsertionEnv-v0',
#     entry_point='learn_seq.envs.real.insertion:RealInsertionEnv',
# )

register(
    id='MujocoSlidingEnv-v0',
    entry_point='learn_seq.envs.mujoco.sliding:MujocoFrankaSlidingEnv',
)

register(
    id='SimpleMujocoSlidingEnv-v0',
    entry_point='learn_seq.envs.mujoco.sliding:SimpleStateSlidingEnv',
)
