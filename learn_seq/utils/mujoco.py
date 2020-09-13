import os
import mujoco_py
from learn_seq.utils.general import get_mujoco_model_path

def load_model(xml_name="round_pih.xml"):
    """Load a model from `mujoco/franka_pih`

    :param type xml_name: Description of parameter `xml_name`.
    :param type primitive: Description of parameter `primitive`.
    :return: Description of returned object.
    :rtype: type

    """
    model_path = get_mujoco_model_path()
    xml_path = os.path.join(model_path, xml_name)
    model = mujoco_py.load_model_from_path(xml_path)
    sim = mujoco_py.MjSim(model)

    return sim

def attach_viewer(sim):
    return mujoco_py.MjViewer(sim)
