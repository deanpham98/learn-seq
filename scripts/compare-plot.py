import os
import numpy as np
import matplotlib.pyplot as plt
from learn_seq.utils.general import get_module_path
from learn_seq.utils.mujoco import quat2vec, pose_transform, inverse_frame
from python_utils.latex_format import latexify, format_axes
import matplotlib.ticker as ticker

SIM_DATA_NAME = "sim_traj.npy"
REAL_DATA_NAME = "real_traj.npy"
SIM_TRAJ_INFO_NAME = "sim_traj_info.npy"
REAL_TRAJ_INFO_NAME = "real_traj_info.npy"

module_path = get_module_path()
module_path = os.path.join(module_path, "compare-data")
sim_data_path = os.path.join(module_path, SIM_DATA_NAME)
real_data_path = os.path.join(module_path, REAL_DATA_NAME)
sim_traj_info_path = os.path.join(module_path, SIM_TRAJ_INFO_NAME)
real_traj_info_path = os.path.join(module_path, REAL_TRAJ_INFO_NAME)

sim_traj = np.load(sim_data_path, allow_pickle=True)
sim_traj = sim_traj.item()

real_traj = np.load(real_data_path, allow_pickle=True)
real_traj = real_traj.item()

sim_traj_info = np.load(sim_traj_info_path, allow_pickle=True)
sim_traj_info = sim_traj_info.item()

real_traj_info = np.load(real_traj_info_path, allow_pickle=True)
real_traj_info = real_traj_info.item()

# task frame
sim_tf_pos = sim_traj_info["task_frame"][0]
sim_tf_quat = sim_traj_info["task_frame"][1]
sim_inv_tf_pos, sim_inv_tf_quat = inverse_frame(sim_tf_pos, sim_tf_quat)

real_tf_pos = real_traj_info["task_frame"][0]
real_tf_quat = real_traj_info["task_frame"][1]
real_inv_tf_pos, real_inv_tf_quat = inverse_frame(real_tf_pos, real_tf_quat)

# transform to task frame
rref = np.array([1, 0, 0])
psim_tf = []
rsim_tf = []
tsim = np.array(sim_traj["t"])
# align time
tsim = tsim - tsim[0]
psim = sim_traj["p"]
qsim = sim_traj["q"]
for p, q in zip(psim, qsim):
    ptf, qtf = pose_transform(p, q, sim_inv_tf_pos, sim_inv_tf_quat)
    psim_tf.append(ptf)
    rsim_tf.append(quat2vec(qtf, rref))
    rref = rsim_tf[-1].copy()
psim_tf = np.array(psim_tf)
rsim_tf = np.array(rsim_tf)

latexify()

fig, ax_pos = plt.subplots(3, 2, sharex="col")
for i in range(3):
    ax_pos[i, 0].plot(tsim, psim_tf[:, i])
    ax_pos[i, 1].plot(tsim, rsim_tf[:, i])
# real
rref = np.array([1, 0, 0])
preal_tf = []
rreal_tf = []
preal = np.array([i[1] for i in real_traj["p"]])
tpreal = np.array([i[0] for i in real_traj["p"]])
qreal = np.array([i[1] for i in real_traj["q"]])
tqreal = np.array([i[0] for i in real_traj["q"]])
tpreal = tpreal - tpreal[0]
tqreal = tqreal - tqreal[0]
for i in range(len(tpreal)):
    p = preal[i]
    q = qreal[i]
    ptf, qtf = pose_transform(p, q, real_inv_tf_pos, real_inv_tf_quat)
    preal_tf.append(ptf)
    rreal_tf.append(quat2vec(qtf, rref))
    rref = rreal_tf[-1].copy()
preal_tf = np.array(preal_tf)
rreal_tf = np.array(rreal_tf)

labely = ["x", "y", "z"]
for i in range(3):
    ax_pos[i, 0].plot(tpreal, preal_tf[:, i])
    ax_pos[i, 0].set_ylabel(labely[i] + " (m)")
    ax_pos[i, 1].plot(tqreal, rreal_tf[:, i])
    ax_pos[i, 1].set_ylabel(labely[i] + " (rad)")
    # ax_pos[2, 0].set_xlim(0, end)

ax_pos[0, 0].legend(["sim", "real"])
ax_pos[2, 0].set_xlabel("Time (s)")
ax_pos[2, 1].set_xlabel("Time (s)")
ax_pos[0, 0].set_title("Position")
ax_pos[0, 1].set_title("Orientation")
start, end = ax_pos[2, 0].get_xlim()
ax_pos[2, 0].set_xlim(0, end)
ax_pos[2, 0].xaxis.set_ticks(np.arange(0, end, 5))
start, end = ax_pos[2, 1].get_xlim()
ax_pos[2, 1].set_xlim(0, end)
ax_pos[2, 1].xaxis.set_ticks(np.arange(0, end, 5))
# ax_pos[0, 0].yaxis.set_major_locator(ticker.MultipleLocator(0.002))
# fmt = ticker.StrMethodFormatter("{x}")
# ax_pos[0, 0].xaxis.set_major_formatter(fmt)
# ax_pos[0, 0].yaxis.set_major_formatter(fmt)

for i in range(3):
    for j in range(2):
        format_axes(ax_pos[i, j])
# format_axes(ax_force)
# plt.tight_layout()
# plt.savefig("traj-pos.pdf")

# pot force
fsim = np.array(sim_traj["f"])

fig, ax_force = plt.subplots(3, 2, sharex="col")
for i in range(3):
    for j in range(2):
        ax_force[i, j].plot(tsim, fsim[:, i + 3*j])

freal = np.array([i[1] for i in real_traj["f"]])
tfreal = np.array([i[0] for i in real_traj["f"]])
tfreal = tfreal - tfreal[0]

for i in range(3):
    for j in range(2):
        ax_force[i, j].plot(tfreal, freal[:, i + 3*j])
        if j ==0:
            ax_force[i, j].set_ylabel(labely[i] + " (N)")
        else:
            ax_force[i, j].set_ylabel(labely[i] + " (Nm)")
ax_force[0, 0].legend(["sim", "real"])
ax_force[2, 0].set_xlabel("Time (s)")
ax_force[2, 1].set_xlabel("Time (s)")
ax_force[0, 0].set_title("Force")
ax_force[0, 1].set_title("Torque")
start, end = ax_force[2, 0].get_xlim()
ax_force[2, 0].set_xlim(0, end)
ax_force[2, 0].xaxis.set_ticks(np.arange(0, end, 5))
start, end = ax_force[2, 1].get_xlim()
ax_force[2, 1].set_xlim(0, end)
ax_force[2, 1].xaxis.set_ticks(np.arange(0, end, 5))
plt.tight_layout()
plt.savefig("traj-force.pdf")
# plt.show()
