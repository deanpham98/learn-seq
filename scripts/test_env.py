import numpy as np
import mujoco_py
import gym
import matplotlib.pyplot as plt

def test_env():
    env = gym.make(id="learn_seq:SimpleMujocoSlidingEnv-v0")
    viewer = mujoco_py.MjViewer(env.sim)
    env.reset()
    done = False
    env.controller.start_record()
    while not done:
        a = env.action_space.sample()
        obs, rew, done, _ = env.step(a)
        print(obs, rew)
        # viewer.render()
    env.controller.stop_record()
    env.controller.plot_key(["kp"])
    env.controller.plot_key(["f", "fd"])
    plt.show()

if __name__ == '__main__':
    test_env()
