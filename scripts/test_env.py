import numpy as np
import mujoco_py
import gym
import matplotlib.pyplot as plt

def test_env():
    env = gym.make(id="learn_seq:RealSlidingEnv-v0", speed_range=[0.01, 0.01], fd_range=[5, 5])
    # viewer = mujoco_py.MjViewer(env.sim)
    
    for i in range(1):
        env.reset()
        # done = False
        
        # env.controller.start_record()
        # env.ros_interface.start_record()
        # while not done:
        #     a = env.action_space.sample()
        #     obs, rew, done, _ = env.step(a)
        # env.ros_interface.stop_record("test.npy")
        # env.ros_interface.plot_pose()
        # env.ros_interface.plot_force()
        # plt.show()
        # print(obs, rew)
        # viewer.render()
    # env.controller.stop_record()
    # env.controller.plot_key(["kp"])
    # env.controller.plot_key(["f", "fd"])
    # plt.show()

if __name__ == '__main__':
    
    test_env()
