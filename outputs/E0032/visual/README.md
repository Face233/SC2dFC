# E0032 validation figures

Validation figures for seed 42 (`n=158` subjects). Their layout, blue/orange time-series colors, gray/orange baseline comparison, axes, and grid follow the E0024 report style.

- `E0032_alpha_decay`: training-only fitted retention coefficient by future horizon; `alpha=1` keeps the full FC1 offset and `alpha=0` uses only the group template.
- `E0032_horizon_mse`: mean subject-level edge MSE by horizon; error bars are 95% subject-bootstrap intervals.
- `E0032_subject_mse_gain`: paired long-horizon MSE differences; positive values mean E0032 has lower error than the named baseline.
- `E0032_subject_retrieval`: subject retrieval from future average FC. This describes static identity and is not evidence of dynamic phase prediction.
- `E0032_pairwise_prediction_similarity`: cross-subject predicted dynamic similarity versus observed target similarity, shown as a gray/orange comparison line with subject-bootstrap intervals.
- `E0032_static_dynamic_topk`: static future-average FC retrieval and full dynamic-trajectory retrieval, shown with the old baseline-gray/current-orange bar style and chance levels.
- `E0032_multisubject_dynamic_comparison`: the established six-edge time-series layout, using the same four validation subjects and six edges as the E0029 comparison; each experiment has a distinct prediction color.
- `E0032_true_residual_components`: training/validation residual MSE split into time-mean stable and de-meaned dynamic targets, using the established gray/orange bar format.
- `E0032_true_residual_horizon`: training/validation residual MSE and between-subject variance by future window, using the established line format.

The metric figures use the E0024 report's line and bar formats. The multi-subject plot uses the older six-edge time-series layout. Figures are saved as PNG files to match the established experiment-report format.

See `subject_similarity_report.md` for metric definitions and the numeric comparison. Regenerate the main figures with `generate_report.py`, the similarity and Top-k figures with `generate_similarity_report.py`, and the multi-subject time-series figure with `build_old_style_multisubject_comparison.py`. Recompute the comparison metrics first with `analyze_subject_similarity.py` if the validation predictions change.

See `E0032_true_residual_audit.md` for the target-only stable/dynamic audit. Recompute its JSON, per-subject CSV, and two PNG figures with `audit_true_residual.py` using the saved training-only alpha and templates.
