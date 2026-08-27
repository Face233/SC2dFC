import matplotlib.pyplot as plt
import numpy as np

from scdfc.visualization import FIXED_EDGE_COUNT, fixed_edge_indices


def test_fixed_edge_selection_is_stable_and_unique():
    first = fixed_edge_indices()
    second = fixed_edge_indices()
    assert len(first) == FIXED_EDGE_COUNT
    assert np.array_equal(first, second)
    assert np.array_equal(first, np.sort(first))
    assert len(np.unique(first)) == FIXED_EDGE_COUNT


def test_constrained_layout_keeps_header_inside_canvas(tmp_path):
    figure = plt.figure(figsize=(7, 4), layout="constrained")
    header, body = figure.subfigures(2, 1, height_ratios=(0.2, 0.8))
    header.text(0.5, 0.7, "Report title", ha="center")
    axis = body.subplots()
    axis.plot([0, 1], [0, 1])
    axis.set(xlabel="Minutes", ylabel="Fisher-z")
    output = tmp_path / "layout.png"
    figure.savefig(output, dpi=100)
    renderer = figure.canvas.get_renderer()
    box = figure.get_tightbbox(renderer)
    width, height = figure.get_size_inches()
    assert box.x0 >= 0 and box.y0 >= 0
    assert box.x1 <= width and box.y1 <= height
    plt.close(figure)
