## Troubleshooting

### Windows: `DLL load failed while importing _errors` (h5py)

**Symptom:** Running without `--headless` crashes at startup with `ImportError: DLL load failed while importing _errors`. Headless mode works.

**Cause:** Likely a DLL conflict (`hdf5.dll`) between h5py and Isaac Sim's GUI components.

**Fix:** Make sure h5py is imported before Isaac Sim starts, using one of the two options below.

**Option 1: environment only (no code change).** With the conda environment activated:

```
echo import h5py> "%CONDA_PREFIX%\Lib\site-packages\h5py_first.pth"
```

To undo, delete `h5py_first.pth`.

**Option 2: in the code.** Add this at the top of `isaaclab_experiments/utils/launcher.py`, before `AppLauncher` is created:

```python
import h5py  # noqa: F401  # must be imported before AppLauncher / SimulationApp
```