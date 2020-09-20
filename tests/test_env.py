
def test_reset(env):
    import matplotlib.pyplot as plt
    env.controller.start_record()
    env.reset()
    env.controller.stop_record()
    env.controller.plot_error()
    env.controller.plot_pos()
    env.controller.plot_orient()
    plt.show()
    assert False
