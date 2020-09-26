import os
import numpy as np
import matplotlib.pyplot as plt
from learn_seq.utils.general import get_exp_path
from learn_seq.utils.ros import extract_error, read_bag, extract_pose
from learn_seq.utils.mujoco import quat2vec
import rosbag

BAG_FILE = "data/controller.bag"
EXP_NAME = "real-21-9"

exp_path = get_exp_path(EXP_NAME)
bag_path = os.path.join(exp_path, BAG_FILE)

t, msgs, _, _ = read_bag(bag_path)
t = np.array(t)
t = t - t[0]

p, q, pd, qd = extract_pose(msgs)
r = []
rd = []

for qi, qdi in zip(q, qd):
    r.append(quat2vec(qi))
    rd.append(quat2vec(qdi))
r = np.array(r)
rd = np.array(rd)

fig, ax = plt.subplots(3, 2)
for i in range(3):
    ax[i, 0].plot(t, pd[:, i])
    ax[i, 0].plot(t, p[:, i])
    ax[i, 0].legend(["pd", "p"])
    ax[i, 1].plot(t, rd[:, i])
    ax[i, 1].plot(t, r[:, i])
    ax[i, 1].legend(["rd", "r"])

plt.show()
