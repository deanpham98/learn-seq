import os
import learn_seq

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
