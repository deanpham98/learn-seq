# learn-seq
This is a sim2real framework that learn a sequence in simulation and transfer to
real-robot.

## INSTALL
- install mujoco_py manually

        pip install "mujoco-py=<2.1,>=2.0"

- install learn_seq package

        git clone --recurse-submodules https://github.com/nvuong21/learn-seq.git
        cd learn-seq
        python setup.py install

## TODO
- adjust `round_hole_generation.py` so that the `round_pih.xml` can include the generated file
- fixed divide by 0 (quaternion)
- add controller gain to configure
- because `robot_state` is a member multiple objects, should have a isUpdate to check whether the sim is updated at each timestep
- change obs_up_limit and obs_low_limit to be defined relative to the goal -> change peg_pos_range and peg_rot_range specification
- add run_ID to argparse
