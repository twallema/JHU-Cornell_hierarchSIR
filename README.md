# JHU-Cornell_hierarchSIR

A hybrid multi-strain SIR - Bayesian hierarchical discrepancy model for infectious disease forecasting.

## Installation (local)

Available platforms: macOS and Linux.

Note: Will not work on Windows because the OS-dependent path to the C++ Boost libraries is not included in `setup.py`.

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

Note: Boost is a C++ library and is not installed "inside" the conda environment but rather on your local machine. In `setup.py` the software is pointed to the location of the Boost library.

### Install the `hierarchSIR` package

Install the `hierarchSIR` Python package inside the conda environment using,

    ```bash
    pip install -e . --force-reinstall
    ```

Note: The installation script requires the use of `pybind11` to "bind" the multi-strain SIR model in C++ to a Python function. This is the purpose of `pyproject.toml`.
Note: If you make any changes to the C++ files you need to reinstall `hierarchSIR`.

### Model training and forecasting

See `scripts/code/hierarchical_training.py` and `scripts/code/incremental_calibration.py`.

## Running on the JHU Rockfish cluster

See `JHU-ROCKFISH_README.md`.
