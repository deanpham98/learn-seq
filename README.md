# learn-seq
This is a sim2real framework that learn a sequence in simulation and transfer to
real-robot.

## INSTALL
- install mujoco_py manually

        pip install "mujoco-py=<2.1,>=2.0"

- clone learn_seq package

        git clone --recurse-submodules https://github.com/nvuong21/learn-seq.git && cd learn-seq

- install dependency for `rlpyt`

    cpu version

        pip install --requirement rlpyt-cpu.txt

    OR gpu with cuda 10.2 (not tested)

        pip install --requirement rlpyt-cuda10.txt

- install package for developing

        python setup.py develop

    NOTE: `python setup.py insttall` does **not** work with the module at the moment

## TODO
- adjust `round_hole_generation.py` so that the `round_pih.xml` can include the generated file
- fixed divide by 0 (quaternion)
- change obs_up_limit and obs_low_limit to be defined relative to the goal -> change peg_pos_range and peg_rot_range specification
- add run_ID to argparse
