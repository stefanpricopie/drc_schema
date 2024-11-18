# Schema Dynamic Resource Constraints

## Thesis Note

This repository contains the code and resources corresponding to **Chapter 3** of my thesis: *Tuning Bayesian Optimization for Dynamic Resource Constraints*.

For code and resources for the other chapters, see the following repositories:
- **Chapter 4**: [Chapter 4 Repository Link](https://github.com/stefanpricopie/drc_setup)
- **Chapter 5**: [Chapter 5 Repository Link](https://github.com/stefanpricopie/drc_lookahead)
- **Chapter 6**: [Chapter 6 Repository Link](https://github.com/username/chapter6-repo)

## Getting started

From the base `drc_schema` directory run:

`pip install --use-pep517 -e .`

## Structure

The code is structured in three parts.
- The utilities for constructing the acquisition functions and other helper methods are defined in `dynamic_resource_constraints/`.
- The experiments are found in and ran from within `experiments/`. The `main.py` is used to run the experiments, and the experiment configurations are found in the `config.json` file of each sub-directory.

The individual experiment outputs were left out to avoid inflating the file size.

## Running Experiments

To run a basic benchmark based on the `config.json` file in `experiments/<experiment_name>` using `<algorithm>`:

```
cd experiments
python main.py <experiment_name> <algorithm> <seed>
```

The code refers to the algorithms using the following labels:
```
algorithms = [
]
```
Each folder under `experiments/` corresponds to the experiments in the paper according to the following mapping:
```
experiments = {
}
```
