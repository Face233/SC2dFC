"""Save the archived E0044 PCA as a reusable frozen artifact without refitting."""
import argparse
from pathlib import Path

from scdfc.fc_pca import export_frozen_fc_pca


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--diagnostic-dir",
                        default="outputs/E0044/diagnostics/E0044-20261020T070318Z-b83fc41d")
    parser.add_argument("--artifact-id", default="A0044")
    args = parser.parse_args()
    print(export_frozen_fc_pca(args.diagnostic_dir, args.root, args.artifact_id))
