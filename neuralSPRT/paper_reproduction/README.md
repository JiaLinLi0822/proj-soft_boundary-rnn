# neuralSPRT Python Reproduction

This folder contains a fresh Python reproduction of the MATLAB pipeline in
`neuralSPRT.m` from the original repository.

## Files

- `neuralSPRT.py`: top-level runner equivalent to MATLAB `neuralSPRT.m`
- `neuralSPRT_BEH.py`: behavioral analyses (Fig. 2A-2F)
- `neuralSPRT_PHY.py`: physiology analyses (Fig. 3A-4B)
- `ecodeRTshape.py`: event-code lookup table equivalent to MATLAB script
- `figures/`: output folder for saved figures

## Run

From project root:

```bash
python neuralSPRT_python_repro/neuralSPRT.py --id 2 --beh 1 2 3 4 5 6 --phy 1 2 3 4
```

To reproduce Fig. 2C/2E with paper-like non-decision-time settings,
run those panels without including behavior switch `2` in the same call.
