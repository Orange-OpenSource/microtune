# Imported Dataset files

## Overview

This folder `import` must contains pickle files with Datasets used by `run_gen_pickles.py` to prepare the 3 datasets `Train`, `Eval`and `Test` as expected by the configurated experimentation.
    - Train: Used for models training
    - Eval: Used to evaluate and elect best learned models
    - Test: Used to test the elected `best`learned model.

The experimentations to use are located in `configs/experiment/` and named `dataset*.yaml`.

This 

## Usage example

### Example 1 - Prepare the datasets for Microtune's paper experimentations

```
  # Prepare the dataset before training or tests
  $ python run_gen_pickles.py +experiment=dataset_legacy

  # Train and Test. Use the shell script which take an experiment as argument
  $ ./expetraintest.sh legacy-ppo
```
Both experiments `configs/experiment/dataset_legacy.yaml` and `configs/experiment/legacy-ppo.yaml` must be aligned on the same `datasets_prefix`, `version` and `dataset` config selection (in `configs/datasets`).

This will import the Dataset V11 and prepare a dataset named `workloads_legacy_full_XX.pickle` which contains the whole data for Train, Eval, Test stages as temporary data in local folder + some pre-computed columns required during training for example.

Then according to the configuration, it splits the `...full_XX.pickle` for the 3 stages into temporary files in local folder.


## Main Dataset experiments for dataset preparation

- `config/experiment/dataset_legacy.yaml`, imports data used for Microtune's paper and prepare datasets
- `config/experiment/dataset_legacylessv63.yaml`, imports data `workloads_legacyless63.pickle` and prepare dataset
- `config/experiment/dataset_simuv15.yaml`, import v15 dataset and simulated one. Prepare all of them for Orange internal experimentations.


## List of possible Dataset files to import

Some files are private to Orange, but the one used in Microtune's paper is delivered as a Release in this Git public repo. 

We describe here only the public datasets.

- `workloads_fe_11.pickle`, v11 (actually V9+V10) data collected in Flexible Engine, used in Microtune's paper and experiments E11. Source: Git Release 2.x (found as workloads_c098_xxx in release 1.0)
- `workloads_gke_14.pickle`, v14 data is a set of workloads collected on GKE/GCP, less data than the v11 collection but on a different infrastrucure and VM flavour. Source: please ask
- `workloads_fe_15.pickle`, v15 is v11 data + additional workloads observations collected on Flexible Engine like v11. Source: please ask
- `workloads_feless63_15.pickle`, v15 data while removing many workloads observations where SLA tipping point have a buffer index >= 63 (over 256MB). Source: please ask
- Misc simulated data. Source: Orange S3, not available externally.
