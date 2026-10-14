# E0040 / E0041 validation curves

The figure follows the E0037 six-edge, four-subject layout and reuses its fixed validation subjects and edge selection. Blue shows the true FC values; red shows E0040 (K=1), and green shows E0041 (K=5). The fifth column shows the training-set mean target sequence.

Both experiments use the same cutoff at FC[19] and predict the same FC[20:] targets. The light gray region marks the first 17 future indices; the dotted line marks the predeclared conservative nonoverlap interval. Each edge row uses one shared vertical scale across all five columns.

The four subjects and six edges are a fixed visual sample. Aggregate paired metrics for all 158 validation subjects are in E0041's `evaluation_val.json`. The plotting script reads frozen configurations, best checkpoints, the local shared history statistics, and the E0037 selection manifest; it does not use test data or export full prediction arrays.

Regenerate locally with:

```powershell
C:\ProgramData\miniconda3\envs\fmri\python.exe outputs/E0041/visual/build_e0040_e0041_multisubject_comparison.py
```
