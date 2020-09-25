import numpy as np
from mujoco_py import functions
from learn_seq.utils.mujoco import MJ_BODY_OBJ, MJ_GEOM_OBJ, MJ_SITE_OBJ,\
            MJ_BOX, MJ_MESH, MJ_CYLINDER
from learn_seq.utils.mujoco import get_geom_size, get_geom_friction, get_body_mass,\
            set_geom_friction, set_geom_size, set_body_mass, get_body_pose,\
            set_body_pose


class MujocoModelWrapper:
    def __init__(self, model):
        self.model = model
        self.peg_name = "peg"
        self.hole_body_name  = "hole"
        self.peg_geom_id = functions.mj_name2id(self.model, MJ_GEOM_OBJ, self.peg_name)
        self.peg_body_id = functions.mj_name2id(self.model, MJ_BODY_OBJ, self.peg_name)
        # joint_names = ["panda0_joint{}".format(i) for i in range(1, 8)]
        # # dof_id = joint_id in the case of revolute joints
        # self.joint_idx = [functions.mj_name2id(self.model, 4, n) for n in joint_names]

        self.map_set = {
            "friction": self.set_friction,
            "mass": self.set_mass,
            "clearance": self.set_clearance,
            "timestep": self.set_timestep,
            "joint_damping": self.set_joint_damping,
        }

    def get_shape(self):
        shape_type = self.model.geom_type[self.peg_geom_id]
        if shape_type == MJ_BOX:
            return "square"
        elif shape_type == MJ_CYLINDER:
            return "round"
        elif shape_type == MJ_MESH:
            return "mesh"

    def get_size(self):
        """Return the peg and hole size. Assume the `box` type is a square shape.
        Also assume the hole size is a interger in mm
        """
        shape = self.get_shape()
        if shape == "round":
            peg_dia = get_geom_size(self.model, self.peg_name)[0]
            hole_dia = np.ceil(peg_dia *1000) / 1000.
            return peg_dia, hole_dia
        elif shape == "square":
            peg_side = get_geom_size(self.model, self.peg_name)[0]
            hole_side = np.ceil(peg_side *1000) / 1000.
            return peg_side, hole_side

    def get_friction(self):
        """Return the 3D frictional vector between peg and hole """
        friction = get_geom_friction(self.model, self.peg_name)
        return friction

    def get_clearance(self):
        """Assume that the diameter of peg is lower than the hole, and the clearance is
        between [0 - 1]mm. Return the clearance in m Only for round peg now
        """
        peg_size, hole_size = self.get_size()
        return hole_size - peg_size

    def get_mass(self):
        return get_body_mass(self.model, self.peg_name)

    def get_timestep(self):
        return self.model.opt.timestep

    def get_joint_damping(self):
        return self.model.dof_damping

    def get_hole_pose(self):
        return get_body_pose(self.model, "hole")

    def set_friction(self, friction):
        set_geom_friction(self.model, self.peg_name, friction)

    def set_clearance(self, clearance):
        """Only for round peg currently. clearance is in m"""
        _, hole_size = self.get_size()
        peg_size_arr = get_geom_size(self.model, self.peg_name)
        if self.get_shape() == "round":
            new_peg_size = [hole_size - clearance, peg_size_arr[1], peg_size_arr[2]]
        else:
            new_peg_size = [hole_size - clearance, hole_size - clearance, peg_size_arr[2]]

        set_geom_size(self.model, self.peg_name, new_peg_size)

    def set_mass(self, mass):
        set_body_mass(self.model, self.peg_name, mass)

    def set_timestep(self, dt):
        self.model.opt.timestep = dt

    def set_joint_damping(self, damping):
        """Short summary.

        :param type damping: a 7D vector
        """
        if len(damping) == 7:
            old_damping = self.model.dof_damping
            damping = np.concatenate((damping, old_damping[7:]))
        self.model.dof_damping[:] = damping

    def set_hole_pose(self, pos, quat):
        set_body_pose(self.model, self.hole_body_name, pos, quat)

    def set(self, key, val):
        self.map_set[key](val)
