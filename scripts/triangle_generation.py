import os
from numpy import *
from learn_seq.utils.general import get_mujoco_model_path

SIZE = 0.025
HEIGHT = 0.04
CLEARANCE = 0.00005

def generate_hole(point_list, bound, height):
    """
    generate hole surface by creating mesh based on the 2d point lists,
        the surface area is bounded inside the bounding box

    point_list: [list] Nx2
    bound: [array(2), array(2)] = [p_low, p_up], where p=(x,y)
    """
    no_points = len(point_list)
    p_low = array(bound[0])
    p_up = array(bound[1])
    vertex = []
    line_seg = []
    for i in range(no_points):
        p = array(point_list[i])

        # find intersection with the bounding box
        ix = zeros(2)
        if p[0] == 0:
            ix = array([99, 99.])
        else:
            if p[0]*p_low[0] >= 0:
                ix[0] = p_low[0]
            else:
                ix[0] = p_up[0]
            ix[1] = p[1] * ix[0] / p[0]

        iy = zeros(2)
        if p[1] == 0:
            iy = array([99, 99.])
        else:
            if p[1]*p_low[1] >= 0:
                iy[1] = p_low[1]
            else:
                iy[1] = p_up[1]
            iy[0] = p[0] * iy[1] / p[1]

        if linalg.norm(ix) > linalg.norm(iy):
            vertex.append(iy)
            line_seg.append(array([0, iy[1]]))
        else:
            vertex.append(ix)
            line_seg.append(array([ix[0], 0]))

    mujoco_model_path = get_mujoco_model_path()
    save_path = os.path.join(mujoco_model_path, "triangle/hole_mesh.xml")
    if not os.path.exists(save_path):
        os.mknod(save_path)

    t = "  "
    # create vertex and append to files

    with open(save_path, "w+") as f:
        f.write("<mujoco>\n")
        f.write("<asset>\n")

        for i in range(no_points):
            # vertex of mesh i
            vm = []
            p1 = array(point_list[i])
            p2 = array(point_list[(i+1)%no_points])
            in1 = array(vertex[i])
            in2 = array(vertex[(i+1)%no_points])
            vm.append(append(p1, -height/2))
            vm.append(append(p1, height/2))
            vm.append(append(p2, -height/2))
            vm.append(append(p2, height/2))
            vm.append(append(in1, -height/2))
            vm.append(append(in1, height/2))
            vm.append(append(in2, -height/2))
            vm.append(append(in2, height/2))

            l1 = line_seg[i]
            l2 = line_seg[(i+1)%no_points]
            if (l1 != l2).any():
                if l1[0] == p_low[0] or l1[0] == p_up[0]:
                    if l2[0] == p_low[0] or l2[0] == p_up[0]:
                        if in1[1] < 0:
                            c1 = array([l1[0], p_low[1]])
                            c2 = array([l2[0], p_low[1]])
                        else:
                            c1 = array([l1[0], p_up[1]])
                            c2 = array([l2[0], p_up[1]])

                        vm.append(append(c1, -height/2))
                        vm.append(append(c1, height/2))
                        vm.append(append(c2, -height/2))
                        vm.append(append(c2, height/2))
                    else:
                        c = array([l1[0], l2[1]])
                        vm.append(append(c, -height/2))
                        vm.append(append(c, height/2))

                else:
                    if l2[1] == p_low[1] or l2[1] == p_up[1]:
                        if in1[0] < 0:
                            c1 = array([p_low[0], l1[1]])
                            c2 = array([p_low[0], l2[1]])
                        else:
                            c1 = array([p_up[0], l1[1]])
                            c2 = array([p_up[0], l2[1]])

                        vm.append(append(c1, -height/2))
                        vm.append(append(c1, height/2))
                        vm.append(append(c2, -height/2))
                        vm.append(append(c2, height/2))
                    else:
                        c = array([l2[0], l1[1]])
                        vm.append(append(c, -height/2))
                        vm.append(append(c, height/2))
            v_str = ""
            for p in vm:
                for k in p:
                    v_str = v_str + "{:1.8f} ".format(k)

            f.write(t + "<mesh name=\"hole_prism{}\" vertex=\"{}\"/>\n".format(i, v_str))
        f.write("</asset>\n")
        f.write("</mujoco>\n")



def generate_prism(size, height, clearance=0.0001):
    """
    Generate mesh data for a triangular prism whose base is a
    equlateral triangle

    size: size of 1 side of a triangle
    height: height of the prism
    """
    # Create the file
    mujoco_model_path = get_mujoco_model_path()
    save_path = os.path.join(mujoco_model_path, "triangle/triangular_prism.xml")
    if not os.path.exists(save_path):
        os.mknod(save_path)

    # prism vertex
    # prism vertex
    a = size - clearance*2
    h = height
    v = []
    v.append([0., -a*sqrt(3)/3, h/2])
    v.append([a/2, a*sqrt(3)/6, h/2])
    v.append([-a/2, a*sqrt(3)/6, h/2])
    v.append([0., -a*sqrt(3)/3, -h/2])
    v.append([a/2, a*sqrt(3)/6, -h/2])
    v.append([-a/2, a*sqrt(3)/6, -h/2])

    # generate hole
    a_hole = size
    v_hole = []
    v_hole.append([0., a_hole*sqrt(3)/3])
    v_hole.append([-a_hole/2, -a_hole*sqrt(3)/6])
    v_hole.append([a_hole/2, -a_hole*sqrt(3)/6])
    bound = [array([-0.12, -0.12]), array([0.12, 0.12])]
    generate_hole(v_hole, bound, 0.02)

    # list to string
    v_str = ""
    for l in v:
        for k in l:
            v_str = v_str + "{:1.8f} ".format(k)

    # tab character
    t = "  "

    with open(save_path, "w+") as f:
        f.write("<mujoco>\n")
        f.write("<asset>\n")
        f.write(t + "<mesh name=\"prism\" vertex=\"{}\"/>\n".format(v_str))
        f.write("</asset>\n")
        f.write("</mujoco>\n")

if __name__ == '__main__':
    generate_prism(SIZE, HEIGHT, CLEARANCE)
