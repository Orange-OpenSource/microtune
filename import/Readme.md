# Imported Dataset files

## Overview

This folder `import` must contains pickle files with Datasets used by `run_gen_pickles.py` to prepare the 3 datasets `Train`, `Eval`and `Test` as expected by the configurated experimentation.
    - Train: Used for models training
    - Eval: Used to evaluate and elect best learned models
    - Test: Used to test the elected `best`learned model.

The experimentations to use are located in `configs/experiment` and named `dataset*.yaml`.

This 

## Usage example

### Example 1 - Prepare the datasets for Microtune's paper experimentations

```
  $ python run_gen_pickles.py +experiment=dataset_legacy.yaml
```

This will import the Dataset V11 and prepare a dataset named `workloads_legacy_full_XX.pickle` which contains the whole data for Train, Eval, Test stages as temporary data in local folder + some pre-computed columns required during training for example.

Then according to the configuration, it splits the `...full_XX.pickle` for the 3 stages into temporary files in local folder.


## Dataset experiments

- `config/experiment/dataset_legacy.yaml`


## List of possible Dataset files to import

Some files are private to Orange, but the one used in Microtune's paper is delivered as a release in this Git public repo. 

We describe here only the public datasets.

- `workloads_c098_full_11.pickle`, all data
