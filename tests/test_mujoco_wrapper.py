import numpy as np
import pytest

from learn_seq.mujoco_wrapper import MujocoModelWrapper
from learn_seq.utils.mujoco import load_model


@pytest.fixture
def wrapper():
    sim = load_model("round_pih.xml")
    return MujocoModelWrapper(sim.model)


@pytest.fixture
def square_wrapper():
    sim = load_model("square_pih.xml")
    return MujocoModelWrapper(sim.model)


def test_wrapper(wrapper, square_wrapper):
    assert wrapper.get_shape() == "round"
    assert square_wrapper.get_shape() == "square"
    # size
    round_size = wrapper.get_size()
    assert round_size[0] == 0.0149
    assert round_size[1] == 0.015

    square_size = square_wrapper.get_size()
    assert square_size[0] == 0.01495
    assert square_size[1] == 0.015
    # clearance
    assert np.abs(square_wrapper.get_clearance() - 0.00005) < 1e-7
    assert np.abs(wrapper.get_clearance() - 0.0001) < 1e-7
    # friction
    assert square_wrapper.get_friction()[0] == 0.5
    assert wrapper.get_friction()[1] == 0.05
    # mass
    assert wrapper.get_mass() == 0.5
    # timestep
    assert wrapper.get_timestep() == 0.002
    # damping
    assert wrapper.get_joint_damping()[0] == 10.


def test_set(wrapper):
    wrapper.set("mass", 1)
    assert wrapper.get_mass() == 1

    wrapper.set("joint_damping", np.array([15] * 7))
    assert wrapper.get_joint_damping()[2] == 15

    wrapper.set("timestep", 0.001)
    assert wrapper.get_timestep() == 0.001

    wrapper.set("friction", [0.1, 0.001, 0.001])
    assert wrapper.get_friction()[0] == 0.1

    wrapper.set("clearance", 0.00001)
    assert np.abs(wrapper.get_clearance() - 0.00001) < 1e-7
