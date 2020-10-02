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

## Instruction
- Mujoco model of franka robot is stored in `mujoco/franka_sim`
- Insertion environment model is stored in `mujoco/franka_pih`
- Experiments are put in a seperate branch `exp`
- To create a training experiment named $EXP_NAME, first create a directory in the `exp` directory, then create a `config.py` file to setup the configurations (primitives, algorithm, ...). Refer to the `exp/example_rlpyt` for more detail.
- To run the training experiment for the round peg-in-hole insertion environment using cpus only

        python scripts/train_rlpyt -n $EXP_NAME -e round
    The training progress, trained model, and training info will be stored in the subdirectory of the $EXP_NAME directory.

- To plot the training performance of all trials in an experiment:

        python scripts/eval_rlpyt -n $EXP_NAME --plot-only

- To run the evaluation of all trials in an experiment for 5 episodes, and render the progress, run

        python scripts/eval_rlpyt -n $EXP_NAME -e 5 --render

- To run the evaluation for a single trial in an experiment, add `-rn $RUN_NAME` at the end

        python scripts/eval_rlpyt -n $EXP_NAME -e 5 --render -rn $RUN_NAME

## TODO
- adjust `round_hole_generation.py` so that the `round_pih.xml` can include the generated file
- fixed divide by 0 (quaternion)
- change obs_up_limit and obs_low_limit to be defined relative to the goal -> change peg_pos_range and peg_rot_range specification
- add run_ID to argparse
