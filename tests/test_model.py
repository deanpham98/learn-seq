from learn_seq.utils.mujoco import load_model, attach_viewer

def test_round_model():
    sim = load_model("round_pih.xml")
    viewer = attach_viewer(sim)
    while sim.data.time < 1:
        sim.step()
        viewer.render()
    assert True

def test_square_model():
    sim = load_model("square_pih.xml")
    viewer = attach_viewer(sim)
    while sim.data.time < 1:
        sim.step()
        viewer.render()
    assert True
