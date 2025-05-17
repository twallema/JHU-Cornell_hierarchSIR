# JHU-Cornell_hierarchSIR

A hybrid multi-strain SIR - Bayesian hierarchical discrepancy model

## Installation

Made and tested for macOS, and should work on Linux too. Will not work on Windows because the OS-dependent path to the Boost libraries is not included in `setup.py`.

### Setup and activate a conda environment

Update conda to make sure your version is up-to-date,

    ```bash
    conda update conda
    ```

Setup/update the `environment`: All dependencies needed to run the scripts are collected in the conda `hierarchSIR_env.yml` file. To set up the environment,

    ```bash
    conda env create -f hierarchSIR_env.yml
    conda activate HIERARCHSIR
    ```

or alternatively, to update the environment (needed after adding a dependency),

    ```bash
    conda activate HIERARCHSIR
    conda env update -f hierarchSIR_env.yml --prune
    ```

### Install the Boost libraries 

Install the C++ Boost libraries needed to integrate the multi-strain SIR model,

    ```bash
    sudo apt-get update && sudo apt-get install -y libboost-all-dev
    ```

Note: Boost is a C++ library and is not installed "inside" the conda environment.

### Install the `hierarchSIR` package

Install the `hierarchSIR` Python package inside the conda environment using,

    ```bash
    pip install -e . --force-reinstall
    ```

Note: The installation script requires the use of `pybind11` to "bind" the multi-strain SIR model in C++ to a Python function. This is the purpose of `pyproject.toml`.
Note: If you make any changes to the C++ files you need to reinstall `hierarchSIR`.
