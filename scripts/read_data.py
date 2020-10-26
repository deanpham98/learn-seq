import os
import numpy as np
import matplotlib.pyplot as plt
from learn_seq.utils.general import get_module_path

FILE_NAME = "fix-seq-traj.npy"
module_path = get_module_path()
file_path = os.path.join(module_path, FILE_NAME)

data = np.load(file_path, allow_pickle=True)
data = data.item()

f = [i[1] for i in data["f"]]
tf = [i[0] for i in data["f"]]

plt.plot(tf, f)
plt.show()
