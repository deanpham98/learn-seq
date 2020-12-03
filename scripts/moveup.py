from learn_seq.ros.ros_interface import FrankaRosInterface

def main():
    ros_interface = FrankaRosInterface()
    ros_interface.move_up(timeout=6)

if __name__ == "__main__":
    main()