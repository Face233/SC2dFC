# E0040 validation curves

The figure follows the E0037 six-edge, four-subject layout and reuses its fixed subject and edge selection. Blue is observed Fisher-z FC; red is the seed-42 E0040 best-checkpoint forecast. The fifth column is the training-set mean target sequence aligned to E0040's forecast origin.

The first target is FC[20], after the last observed FC[19]. Light gray shading marks the first 17 target windows. The dotted line marks the predeclared conservative nonoverlap interval starting at future index 17. Each row uses one shared vertical scale across all five columns.

The E0037 and E0040 experiments have different forecast origins, so their curves are not directly overlaid. These four subjects and six edges are a fixed visual sample, not an estimate of average model performance; aggregate results are in the run's `evaluation_val.json`.

Regenerate locally with:

```powershell
C:\ProgramData\miniconda3\envs\fmri\python.exe outputs/E0040/visual/build_e0040_multisubject_comparison.py
```

The script reads the frozen E0040 run, its best checkpoint, the private FC cache, and the E0037 selection manifest. No test data or full prediction export is used.
