import matplotlib.pyplot as plt
import numpy as np

from scdfc.visualization import FIXED_EDGE_COUNT, _overview_architecture_text, _overview_loss_text, fixed_edge_indices


def test_fixed_edge_selection_is_stable_and_unique():
    first = fixed_edge_indices()
    second = fixed_edge_indices()
    assert len(first) == FIXED_EDGE_COUNT
    assert np.array_equal(first, second)
    assert np.array_equal(first, np.sort(first))
    assert len(np.unique(first)) == FIXED_EDGE_COUNT


def test_overview_loss_text_shows_only_active_terms_and_loss_family():
    assert _overview_loss_text({"loss_type": "mse", "loss_weights": {"edge": 0, "difference": 3, "variance": 3}}) == "MSE: 3·L_diff + 3·L_var"
    assert _overview_loss_text({"loss_type": "huber", "huber_beta": 0.5, "loss_weights": {"edge": 1}}) == "Huber (β=0.5): 1·L_edge"


def test_overview_architecture_reports_effective_output_head_state():
    base = {"name": "gru", "gru_layers": 2, "sc_encoder": "hcp_gcn"}
    frozen = _overview_architecture_text(
        {**base, "output_head": "e0003_reconstruction_decoder"},
        {"finetune_fc_decoder": False, "decoder_frozen_epochs": 20},
    )
    assert frozen.endswith("E0003 reconstruction decoder (frozen throughout)")
    assert "20 epochs" not in frozen
    direct = _overview_architecture_text({**base, "output_head": "direct_edge_linear"}, {})
    assert direct.endswith("direct edge linear head (trainable)")


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
