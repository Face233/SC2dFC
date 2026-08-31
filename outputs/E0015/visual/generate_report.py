"""Generate the static validation report for E0015/E0015-s42-20260828T082041Z-1b39bec.

Inputs: config_resolved.yaml, metrics_best.json, evaluation_val.json, and
dynamic_audit_val/summary.json from this managed run.
Outputs: four validation PNG figures plus a visual-manifest JSON file.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from scdfc.visualization import generate_report

if __name__ == "__main__":
    generate_report(ROOT, "E0015", "E0015-s42-20260828T082041Z-1b39bec")
