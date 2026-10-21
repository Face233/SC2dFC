"""E0044 val; frozen JSON/CSV/private_examples inputs; regenerate six PNGs."""
from pathlib import Path
import runpy
diagnostic = Path(__file__).resolve().parent.parent / 'diagnostics' / 'E0044-20261020T070318Z-b83fc41d'
renderer = runpy.run_path(str(diagnostic / 'ae_diagnostic_visual.py'))
renderer['render_diagnostic'](diagnostic)
