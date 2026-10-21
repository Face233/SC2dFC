# E0045 / E0046 / E0047 validation curves

This figure follows the E0037 layout: six fixed edge rows, four fixed validation subjects, and the training set group mean in the fifth column. Blue shows the validation target; E0045, E0046, and E0047 predictions use purple, green, and red. Each edge row uses one y range across all five columns.

All three experiments forecast from zero-based origin 9 and use the same FC[10:] target (214 windows). Their FC history lengths differ: E0045 uses K=1, E0046 K=5, and E0047 K=10. The four subjects and six edges are copied from the E0037 figure selection. Only validation data are plotted; the group mean comes from training-set statistics.

The JSON manifest records run IDs, checkpoint hashes, forecast alignment, colors, and selection. Regenerate the figure from the repository root with:

```powershell
C:\ProgramData\miniconda3\envs\fmri\python.exe outputs/E0047/visual/build_e0045_e0046_e0047_multisubject_comparison.py
```
