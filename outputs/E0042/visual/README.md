# E0040 / E0041 / E0042 validation curves

The figure follows the E0037 six-edge, four-subject layout and reuses its fixed validation subjects and edge selection. Blue shows the true FC values; red is E0040 (K=1), green is E0041 (K=5), and dashed purple is E0042 (K=10). The fifth column shows the training-set mean target sequence.

All three runs use the same cutoff at FC[19] and predict the same FC[20:] targets. The light gray region marks the first 17 future indices; the dotted vertical line marks the predeclared conservative nonoverlap interval. Each edge row uses one shared vertical scale across all five columns.

The four subjects and six edges are a fixed visual sample. The JSON manifest records the run IDs, checkpoint hashes, aggregate validation metrics, and paired 10,000-replicate subject bootstrap comparisons for E0042 versus E0040 and E0041. On the 158 validation subjects, the E0042−E0040 nonoverlap MSE difference is -0.00110 (95% CI -0.00405 to 0.00190); the temporal correlation difference is +0.00389 (95% CI -0.00040 to 0.00855). E0042−E0041 differences are +0.00016 (95% CI -0.00131 to 0.00166) for MSE and -0.00070 (95% CI -0.00276 to 0.00137) for temporal correlation. These intervals include zero.

The plotting script reads frozen configurations, best checkpoints, the local shared history statistics, and the E0037 selection manifest. It uses only validation data and does not export full prediction arrays.

Regenerate locally with:

```powershell
C:\ProgramData\miniconda3\envs\fmri\python.exe outputs/E0042/visual/build_e0040_e0041_e0042_multisubject_comparison.py
```
