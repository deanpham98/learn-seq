import numpy as np
import os
import sys
from learn_seq.utils.general import create_file, get_mujoco_model_path

RADIUS = 0.015
LENGTH = 0.05
DEPTH = 0.02
NO_BLOCKS = 100

def generate_hole_xml(radius, length, depth, no_blocks):
    '''
    Approximate hole using primitive objects
    Input:
        - radius: of the hole
        - length: width of each block
        - depth: of the hole
        - no_blocks used to approximate the hole
    Output: the xml file
    '''
    # Create the file
    model_path = get_mujoco_model_path()
    asset_file = os.path.join(model_path, "round/asset.xml")
    create_file(asset_file)

    # write to file
    # tab character
    t = "  "

    # mass of each block
    mass = 1. / (no_blocks + 1)
    alpha = 2*np.pi / no_blocks
    block_edge = 2*radius*np.tan(alpha/2)
    base_height=0.02

    size = [length/2, block_edge/2 , depth/2]
    # r = radius + block_edge/2
    r = radius + length/2

    # calculate vertex
    l = np.sqrt(length**2 + block_edge**2) /2
    phi = -np.arctan(block_edge/length)
    P0 = np.array([r, 0, size[2] + base_height])
    P1 = np.array([r * np.cos(alpha), r * np.sin(alpha), size[2] + base_height])
    V1 = np.array([P0[0] - l*np.cos(phi), P0[1] - l*np.sin(phi), base_height])
    V2 = np.array([P0[0] - l*np.cos(phi), P0[1] - l*np.sin(phi), base_height + depth])
    V3 = np.array([P0[0] + l*np.cos(-phi), P0[1] + l*np.sin(-phi), base_height])
    V4 = np.array([P0[0] + l*np.cos(-phi), P0[1] + l*np.sin(-phi), base_height + depth])
    V5 = np.array([P1[0] + l*np.cos(alpha+phi), P1[1] + l*np.sin(alpha+phi), base_height])
    V6 = np.array([P1[0] + l*np.cos(alpha+phi), P1[1] + l*np.sin(alpha+phi), base_height + depth])


    with open(asset_file, "w+") as f:
        print("File Opened")
        # the general parameters
        f.write("<default class=\"hole\">\n")
        f.write(t+"<geom type=\"box\" mass=\"{:1.4f}\" size=\"{:1.5f} {:1.5f} {:1.5f}\"/>\n"
                    .format(mass, size[0], size[1], size[2]))
        f.write("</default>\n\n")


        # triangular prism mesh
        f.write("<asset>\n")
        f.write(t+"<mesh name=\"prism\" vertex=\"{:1.4f} {:1.5f} {:1.4f}  {:1.4f} {:1.5f} {:1.4f}  {:1.5f} {:1.5f} {:1.5f}  {:1.5f} {:1.5f} {:1.5f}  {:1.5f} {:1.5f} {:1.5f}  {:1.5f} {:1.5f} {:1.5f}\"/>\n"
                .format(V1[0], V1[1], V1[2],V2[0], V2[1], V2[2],
                        V3[0], V3[1], V3[2],V4[0], V4[1], V4[2],
                        V5[0], V5[1], V5[2],V6[0], V6[1], V6[2]))
        f.write("</asset>\n\n")


        # create hole, origin at (0,0,0)
        f.write("<body name=\"hole\" pos=\"0 0 0\">\n")

        # hole base
        f.write(t + "<geom name=\"base\" type=\"box\" mass=\"{:1.5f}\" pos=\"0 0 {}\" size=\"{:1.5f} {:1.5f} {}\"/>\n"
                    .format(mass, base_height/2, radius + block_edge/2 +0.002, radius + block_edge/2 +0.002,base_height/2))

        # hole blocks
        for i in range(no_blocks):
            pos = [r * np.cos(i*alpha), r * np.sin(i*alpha), size[2] + base_height]
            rotate_angle = i*alpha
            # f.write(t*i + "<body name=\"hole_b{}\" pos=\"{:1.5f} {:1.5f} {:1.5f}\" euler=\"0 0 {:1.3f}\">\n"
            #         .format(i+1, pos[0], pos[1], pos[2], rotate_angle))

            f.write(t + "<geom name=\"hole{}\" class=\"hole\" pos=\"{:1.5f} {:1.5f} {:1.5f}\" euler=\"0 0 {:1.3f}\"/>\n"
                    .format(i+1, pos[0], pos[1], pos[2], rotate_angle))

            if i == 0:
                f.write(t+ "<geom name=\"prism1\" type=\"mesh\" mesh=\"prism\"/>\n")
            elif i>0:
                f.write(t+ "<geom name=\"prism{}\" type=\"mesh\" mesh=\"prism\" euler=\"0 0 {:1.5f}\"/>\n".format(i+1,rotate_angle))


        # create a plane above the hole
        # margin = 0.0005
        # # f.write(t*i + "<body name=\"hole_surface1\">\n")
        # f.write(t +"<geom name=\"upper_plate1\" type=\"box\" size=\"{:1.3f} {:1.3f} 0.0005\" pos=\"{:1.5f} {:1.5f} {:1.3f}\" rgba=\"0.5 0.5 0.5 0.3\"/>\n"
        #         .format(0.05-margin, radius+0.1, radius + 0.05, 0, size[2]*2+0.001/2 + base_height))
        #
        # # f.write(t*i + "<body name=\"hole_surface2\">\n")
        # f.write(t +"<geom name=\"upper_plate2\" type=\"box\" size=\"{:1.3f} {:1.3f} 0.0005\" pos=\"{:1.5f} {:1.5f} {:1.3f}\" rgba=\"0.5 0.5 0.5 0.3\"/>\n"
        #         .format(0.05-margin, radius+0.1, -radius - 0.05, 0, size[2]*2+0.001/2 + base_height))
        #
        # # f.write(t*i + "<body name=\"hole_surface3\">\n")
        # f.write(t +"<geom name=\"upper_plate3\" type=\"box\" size=\"{:1.3f} {:1.3f} 0.0005\" pos=\"{:1.5f} {:1.5f} {:1.3f}\" rgba=\"0.5 0.5 0.5 0.3\"/>\n"
        #         .format(radius, 0.05-margin, 0, radius + 0.05, size[2]*2+0.001/2 + base_height))
        #
        # # f.write(t*i + "<body name=\"hole_surface4\">\n")
        # f.write(t +"<geom name=\"upper_plate4\" type=\"box\" size=\"{:1.3f} {:1.3f} 0.0005\" pos=\"{:1.5f} {:1.5f} {:1.3f}\" rgba=\"0.5 0.5 0.5 0.3\"/>\n"
        #         .format(radius, 0.05-margin, 0, -radius - 0.05, size[2]*2+0.001/2 + base_height))



        f.write("</body>")


    f.close()


generate_hole_xml(RADIUS, LENGTH,DEPTH,NO_BLOCKS)
