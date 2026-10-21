# E0032 / E0036 / E0037 / E0045 validation curves

This figure follows the E0043 six-edge, four-subject layout: six shared edge rows, four fixed validation subjects, and the training-set group mean in the fifth column. Blue is the validation target; E0032, E0036, E0037, and E0045 use purple, green, red dashed, and orange dash-dot lines. The gray region covers the first 17 shared forecast windows and the dotted line marks the conservative nonoverlap interval.

All curves use the same validation targets, FC[10:] (214 windows), and the same four subjects and six edges selected for the E0037 figure. The original E0032/E0036/E0037 forecasts start after FC[0]; their native forecasts are clipped to FC[10:] for display. E0045 predicts from FC[9]. The figure makes this difference in forecast origin explicit, so the curves are aligned on target windows but do not have the same amount of observed history.

The metrics printed in the figure and listed below cover only these four plotted validation subjects and six plotted edges, over the nonoverlap interval. They are descriptive and are not full-validation-set aggregates.

| Experiment | Nonoverlap MSE | Temporal r | Predicted / true temporal standard deviation |
| --- | ---: | ---: | ---: |
| E0032 | 0.181510 | 0.187051 | 0.115159 |
| E0036 | 0.189089 | 0.186994 | 0.115404 |
| E0037 | 0.186761 | 0.187382 | 0.115569 |
| E0045 | 0.186054 | 0.184525 | 0.108929 |

No test data are used. The manifest records run IDs, checkpoint and alpha-fit hashes, subject/edge selection, forecast alignment, and the scope of the descriptive metrics.

Regenerate locally with:

```powershell
C:\ProgramData\miniconda3\envs\fmri\python.exe outputs/E0045/visual/build_e0032_e0036_e0037_e0045_comparison.py
```
