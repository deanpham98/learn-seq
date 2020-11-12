import os
import imp
import csv
import numpy as np
import learn_seq
from functools import partial
from collections import OrderedDict

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

def get_exp_path(exp_name):
    module_path = get_module_path()
    return os.path.join(module_path, "exp/" + exp_name)

def load_config(exp_name):
    exp_path = get_exp_path(exp_name)
    config_path = os.path.join(exp_path, "config.py")
    if not os.path.exists(config_path):
        sys.exit("Experiment '%s' does not exist.\nDid you create '%s'?" %
            (exp_name, config_path))
    config = imp.load_source("config" + exp_name, config_path)

    return config

def read_csv(file_path):
    data = OrderedDict()
    with open(file_path) as f:
        ptr = csv.reader(f)
        key_list = None
        data_list = []
        for i, row in enumerate(ptr):
            if i==0:
                key_list = row
            else:
                data_ptr= []
                for r in row:
                    try:
                        data_ptr.append(float(r))
                    except ValueError:
                        if r=="True":
                            data_ptr.append(1)
                        elif r=="False":
                            data_ptr.append(0)
                data_list.append(data_ptr)
    for i, k in enumerate(key_list):
        data[k] = [d[i] for d in data_list]
    return data

def get_dirs(path):
    dir_list = []
    for file in os.scandir(path):
        if file.is_dir():
            dir_list.append(file)
    return dir_list

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
