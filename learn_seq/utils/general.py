import os
import numpy as np
import learn_seq
from functools import partial

def create_file(path):
    '''
    Check if the path exists. If not, create a new one
    '''
    if not os.path.exists(path):
        os.mknod(path)

def get_module_path():
    return os.path.dirname(os.path.dirname(learn_seq.__file__))

def get_mujoco_model_path():
    module_path = get_module_path()
    return os.path.join(module_path, "mujoco/franka_pih")

def saturate_vector(v1, v2, dmax):
    """Limit the difference |v2 - v1| <= dmax, i.e.
    vout[i] = v2[i]             if |v2[i] - v1| <= dmax
    vout[i] = v1[i] + dmax      if  v2[i] - v1 > dmax
    vout[i] = v1[i] - dmax      if  v2[i] - v1 > dmax

    :param np.array v1:
    :param np.array v2:
    :param double dmax: a positive number
    :return:
    :rtype: np.array
    """
    assert dmax >= 0
    dv =  np.minimum(dmax, np.maximum(-dmax, v2 - v1))
    return v1 + dv
