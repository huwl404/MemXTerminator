# MemXTerminator

A software for membrane analysis and subtraction in cryo-EM.

![overview](https://memxterminator.github.io/wiki/assets/images/0-1.png)

## Overview

This software utilizes 2D averages and their corresponding alignment information, employing methods such as Radon transform, cross-correlation, L1 norm, Bezier curves, Monte Carlo simulations, and Genetic Algorithm. It analyzes and subtracts membranes of any shape in cryo-EM, ultimately producing particle stacks and micrographs with membrane signals removed, which are suitable for subsequent membrane protein analysis.

## Features

* Capable of analyzing biological membranes of any shape, including simple lines and arcs, as well as more complex shapes like S or W curves;
* Accurately locates and subtracts biological membrane signals;
* Utilizes GPU and CUDA acceleration to enhance computational speed;
* Features a user-friendly GUI for ease of use.

## Requirements

* This software requires a GPU and CUDA acceleration. So, the installation of CUDA drivers and libraries is necessary.
* [pyem](https://github.com/asarnow/pyem) is also needed to convert cryoSPARC’s `.cs` files to Relion’s `.star` format for processing.

## Wiki

This software has a very accessible [wiki](https://memxterminator.github.io/wiki/) for better tutorial organization.

## Installation

For specific installation methods, please refer to the [wiki installation section](https://memxterminator.github.io/wiki/tutorials/installation/).

## Usage

This software has a user-friendly GUI. To use this software, simply type:

```bash

MemXTerminator gui &

```

For detailed usage tutorials, please refer to the [wiki usage section](https://memxterminator.github.io/wiki/tutorials/usage/).

## Fork maintenance notes

This fork keeps the upstream workflow, with a few operational fixes for HPC use:

* The GUI configures Linux X11 sessions before importing Qt: `QT_X11_NO_MITSHM=1` is set to avoid fragile shared-memory behavior over SSH forwarding. Matplotlib backend selection is left to the running Qt GUI framework.
* RadonFit particle membrane subtraction treats `--procs` as GPU worker processes; the default `0` auto-detects visible CUDA devices. Each worker logs its assigned CUDA device at startup.
* `--batch_size` is now a progress/reporting window; real parallelism is controlled by `--procs`, avoiding minibatch barriers that can leave GPUs idle.
* RadonFit particle and micrograph membrane subtraction accept `--output_dirname`; use the same value for both steps so MMS finds the matching PMS stacks and `.mxt` sidecars.
* On HPC systems where CUDA modules prepend their own Python, install/run from the conda env explicitly, e.g. `$CONDA_PREFIX/bin/python -m pip install -e .`; GUI-launched jobs prefer `$CONDA_PREFIX/bin/python` when available.
* Runtime startup filters the known `starfile`/`pkg_resources` deprecation warning.

## License

This software is licensed under GPL v3.0.

## Acknowledgement

Thanks to Jack(Kai) Zhang@Yale MBB for his guidance.

## Contributing

**Always welcome!** This software may still has room for improvement such as updating the usage documentation, improving the GUI design, and enhancing the software's usability.

I am still working on improving this software. More exciting features are on the way!

## Contact

If you have any questions, please contact me: [zhen.victor.huang@gmail.com](mailto:zhen.victor.huang@gmail.com)
