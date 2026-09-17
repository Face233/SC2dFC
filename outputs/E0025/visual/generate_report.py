"""Generate the static validation report for E0025/E0025-s42-20260917T032721Z-3e974d5.

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
    generate_report(ROOT, "E0025", "E0025-s42-20260917T032721Z-3e974d5")
