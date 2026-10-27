"""E0050 vs E0036 seed42 validation figures, using pinned runs and frozen inputs.

Inputs and PNG outputs: see the shared build_pca_ae_comparison.py in E0051/visual.
Generates overview, fixed_edges, dynamic_calibration and horizon_audit PNGs here.
"""
from pathlib import Path
import runpy

if __name__ == "__main__":
    source = Path(__file__).resolve().parents[2] / "E0051/visual/build_pca_ae_comparison.py"
    runpy.run_path(str(source))["main"](only="E0050")
