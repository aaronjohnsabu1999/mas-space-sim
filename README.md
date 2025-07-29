# ORBEX-A: Orbital Rendezvous and Capture Experiment with Adaptive Tube MPC

**ORBEX-A** is a Python-based simulator for spacecraft performing cooperative orbital rendezvous and capture tasks. The system implements Adaptive Dynamic Tube Model Predictive Control (ADTMPC) for the capture of tumbling targets.

This repository contains simulation code, configuration files, visualizations, and evaluation data for the system described in the IEEE Aerospace Conference paper:  
“Capturing Tumbling Objects in Orbit with Adaptive Tube Model Predictive Control” (Aaron John Sabu and Brett T Lopez).

## Features

- Introduction of Adaptive Tube-based MPC (ADTMPC)
- Chaser-target orbital dynamics and estimation
- Distributed task allocation and multi-agent coordination
- High-resolution trajectory visualizations

## File Structure

- `src/orbexa/`: All simulation source code and core logic
- `docs/`: Evaluation plots, JSON logs, and image-based analysis
- `results/`: HTML/MP4 outputs and benchmark result folders
- `run.py`: Entry point for running a sample adaptive MPC scenario\
- `pyproject.toml`: Project metadata and dependencies
- `.gitignore`: Excludes simulation outputs, caches, IDE artifacts

## Getting Started

You can install dependencies and run a test scenario:

```bash
conda create -n orbexa python=3.9
conda activate orbexa
pip install -e .
python run.py
```

This will run a sample adaptive MPC simulation.

## License

GPLv2 License. See `LICENSE` for details.

## Citation

If you use this work, please cite:

```
@inproceedings{johnsabu2024orbex,
  title={Capturing Tumbling Objects in Orbit with Adaptive Tube Model Predictive Control},
  author={Aaron John Sabu and Brett T. Lopez},
  booktitle={IEEE Aerospace Conference},
  year={2024}
}
```
