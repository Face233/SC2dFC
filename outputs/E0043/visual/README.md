# E0040 / E0041 / E0042 / E0043 validation curves

This figure follows the E0037 six-edge, four-subject layout and reuses its fixed validation subjects and edge selection. Blue shows the true FC values; red is E0040 (K=1), green is E0041 (K=5), dashed purple is E0042 (K=10), and dash-dot orange is E0043 (K=20). The fifth column shows the training-set mean target sequence. All runs use seed 42.

All four runs share the cutoff at FC[19] and predict identical FC[20:] targets (204 windows). The gray region marks the first 17 future indices, and the dotted line marks the conservative nonoverlap interval. Each edge row shares one vertical scale across the five columns. The script verifies completed runs, frozen configuration hashes, checkpoint hashes, common targets, group templates, sample manifests, and the common forecast protocol before plotting.

The fixed visual sample is descriptive; the following metrics summarize all 158 validation subjects in the nonoverlap interval:

| Experiment | K | MSE | Temporal r | Adjusted temporal r | Predicted / true temporal standard deviation |
| --- | ---: | ---: | ---: | ---: | ---: |
| E0040 | 1 | 0.254220 | 0.051673 | 0.031186 | 0.106947 |
| E0041 | 5 | 0.252961 | 0.056256 | 0.036736 | 0.092414 |
| E0042 | 10 | 0.253119 | 0.055558 | 0.035217 | 0.090231 |
| E0043 | 20 | 0.253417 | 0.057651 | 0.034599 | 0.093013 |

The selected E0043 curves remain smooth and capture little of the large fluctuations in the true curves. Its mean predicted/true temporal standard deviation ratio is 0.093, consistent with this limited variation. K=20 has the highest unadjusted temporal correlation among these four runs, but its MSE and adjusted temporal correlation are not the best.

The manifest records paired 10,000-replicate subject bootstrap comparisons (seed 20261013), with differences defined as E0043 minus baseline. Against E0040, temporal r increases by 0.005978 (95% CI 0.001347 to 0.011042; simultaneous 98.333% CI 0.000374 to 0.012228). The MSE difference is -0.000803 (95% CI -0.003350 to 0.001777), and the adjusted temporal r difference is 0.003413 (95% CI -0.003887 to 0.010719). Against E0041 and E0042, all three metric intervals include zero. These single-seed results do not establish a further improvement from increasing K to 20 over K=5 or K=10.

The PNG, generation script, and JSON manifest are stored together here. Inputs are the frozen managed run configurations, best checkpoints, validation reports, shared training history statistics, and E0037 selection manifest. No test data are used, and full prediction arrays are not exported.

Regenerate locally with:

```powershell
C:\ProgramData\miniconda3\envs\fmri\python.exe outputs/E0043/visual/build_e0040_e0041_e0042_e0043_multisubject_comparison.py
```
