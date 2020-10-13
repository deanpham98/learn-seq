import numpy as np
from learn_seq.utils.general import get_exp_path, load_config
from learn_seq.utils.mujoco import integrate_quat, quat_error
from learn_seq.utils.rlpyt import gym_make

SIM_EXP_NAME = "prim-rew-5mm-29"
REAL_EXP_NAME = "real-prim-rew-1-10"

def test_position_error():
    config = load_config(SIM_EXP_NAME)
    config.env_config["xml_model_name"] = "round_pih.xml"
    env = gym_make(**config.env_config)
    p = [np.array([0, 0., 0.01]),
         np.array([-0.001, -0.001, 0.01])]
    r = np.array([1., 1., 0]) * np.pi/180
    q2 = integrate_quat(env.target_quat, r, 1)
    q = [env.target_quat, q2]

    for i in range(len(p)):
        ep = []
        er = []
        for j in range(10):
            env.reset_to(p[i], q[i])
            p0, q0 = env.robot_state.get_pose(env.tf_pos, env.tf_quat)
            ep.append(p0 - p[i])
            er.append(quat_error(q0, q[i]))
        ep = np.array(ep)
        er = np.array(er)
        print("pos mean: {}".format(np.mean(ep, axis=0)))
        print("pos std: {}".format(np.std(ep, axis=0)))
        print("rot mean: {}".format(np.mean(er)))
        print("rot std: {}".format(np.std(er)))

    # real env
    real_config = load_config(REAL_EXP_NAME)
    real_env = gym_make(**real_config.env_config)
    for i in range(len(p)):
        ep = []
        er = []
        for j in range(5):
            real_env.reset_to(p[i], q[i])
            p0, q0 = real_env.ros_interface.get_ee_pose(frame_pos=real_env.tf_pos, frame_quat=real_env.tf_quat)
            real_env.ros_interface.move_up()
            ep.append(p0 - p[i])
            er.append(quat_error(q0, q[i]))
        ep = np.array(ep)
        er = np.array(er)
        print("pos mean: {}".format(np.mean(ep, axis=0)))
        print("pos std: {}".format(np.std(ep, axis=0)))
        print("rot mean: {}".format(np.mean(er)))
        print("rot std: {}".format(np.std(er)))


if __name__ == '__main__':
    test_position_error()
