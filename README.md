# JHU-Cornell_hierarchSIR

## Installation

```bash
pip install -e . --force-reinstall
```

## Julia code

### Structure

 * `src`
    * `HierarchicalSIR.jl`: imports and exports of functionality
    * `utils.jl`: data loading and definition of default set-up
    * `models.jl`: The definition of the central ODE SIR-model, and two Turing models for bayesian inference of its parameters (a complete, and a smaller one). All priors are hard-coded here, thus have to be adjusted in the file.
    * `postprocess.jl`: functionality to inspect and evalutate the produced samples.
 * `scripts/julia`
    * `training.jl` File to do the inferences, see below
    * `interpreation.jl`: script to produce figures based on the samples

### Installation

Install `julia` using [juliaup](https://github.com/JuliaLang/juliaup):

```bash
curl -fsSL https://install.julialang.org | sh
```

Then set up the project environment:

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```

### Usage

The most important script is `scripts/julia/training.jl`, which does the inference.
It can be run interactively in a julia REPL, or as a script, with some options.
Many parameters can be changed in the script.

As it stands, running as a script will do the following:
* without arguments, it will benchmark `Emcee` and `NUTS` (with different autodiff backends) on the small model with limited number of samples.
```bash
julia --project -t 16 scripts/julia/training.jl
```
The `-t` argument specifies the number of threads to use, the `--project` argument ensures the correct environment is used.

* The first argument can be `--emcee` or `--nuts` to sample the model with the given sampler, the second argument can be `--small` or `--full` to choose the model size (default `--full`). If the nuts sampler is chosen, the third argument can be `--forward` or `--reverse` to choose the autodiff backend (default `--forward`), e.g.
```bash
julia --project -t 16 scripts/julia/training.jl --emcee
julia --project -t 16 scripts/julia/training.jl --nuts --small --reverse
```
The first script 
To adjust more parameters, the script has to be edited.

Chains are stored in `data/julia_chains`.

In `scripts/julia/interpretation.jl`, figures can be produced based on the chains.
The script uses functions from `src/postprocess.jl`.
It is meant to be used interactively in a julia REPL.
