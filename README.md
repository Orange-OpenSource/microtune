# Microtune - a micro tuner for Autonomous DBMS (ADBMS) using RL algorithms Bandits (LinUCB) & DeepRL like those in Stable-Baselines3

Microtune purpose is to perform Up and Down scaling of a DBMS memory cache with respect of a SLA on latency viewed by the client application. 

You'll find here the necessary to train and test RL algorithms like Bandits (LinUCB) and several Stable-Baselines3 neural models. Microtune can even drive the cache memory of an on-line database under a workload.

This repo contains:

+ A complete dataset that contains the data observed on a DBMS during 592 workloads with 64 states at different cache size from 128MB up to 8GB (`innodb_buffer_pool_size`).
+ Exploration phase (see [Microtune paper](ask_us)): the code to launch all experimentations for training and models tests
+ Real-time Exploitation phase (see [Microtune paper](ask_us)): the code to run an agent operating an on-line MariaDB database and tuning its InnoDB's buffer pool size

[Research Paper](ask_us): Ask us. VLDB 25, is still under review(s) for publication...

The experimentations details in `Experiments.md` [file](Experiments.md) in this repository, can be replayed.

Otherwise, detailed Results (internal Orange users only) are availables here: 
    + S3 CAV (OB)
    + Bucket: s3selfcare-vstune

## Architecture

RL algorithms are all adressed through a common API to indiferently launch:
+ LinUCB disjoint arms (2 differents implementations)
+ A2C, DQN, DDPG, PPO, SAC (thanks to the framework Stable-Baselines3)
+ In BETA version:
	+ a Neural LinUCB
 	+ a Discounted LinUCB

The code uses actively python's Hydra, a framework for AI, aiming to simplify the management of configuration parameters. In our case, we want to experiment the evaluation of different algorithms with a consistent and reliable method. The application has to share common parameters/data (dataset to use, learning methods, ...) and use specific elements for each RL algo (hyper-parameters, code). Hydra allows us to maintain a flexible and consistent implementation of the configuration, experiments and code to run at each stage.

To configure the environement, we rely also on a Gym Env API as defined by [Gymnasium project](https://gymnasium.farama.org/index.html).

See the `requirements.txt` [file](requirements.txt) to know all Python's dependencies.

## Getting started

Pickup dataset (last version) in Git's Releases. 

Files (`workloads_c098_xxx.pickle`) must be located at the top level folder of the code.

Or ask contributors to provide them.

Install python (min is 3.10, tested with 3.10.12).

Run `pip install -r requirements.txt`


## Run Exploration phase

We describe here a short view of differents stages in game to train, test and generate several curves of results through several experiments (in the Hydra's terms) or scripts. The full process is more detailed in the next paragraph.

Each stage has a python or shell script:
+ Train/Eval/Test `expetraintest.sh`:
	+ train and evaluation, with some graphs (html)
	+ determine best model
	+ test best model and generate graphs (html)
	+ Optionaly, publish results on S3, disabled from a command line option.
+ Optional `python run_simple_test.py`: 
        + SLA curve, i.e. latency curve on N steps during evaluation and tests. Each workload is crossed twice, one time starting from the maximum buffer size, the second time starting from the minimmu buffer size
+ Variability tests 

### Chained execution from learning models up to variability tests

The purpose is to train models with different settings of hyper parameters, evaluate results, choose the best evaluated model, test it.
Models are trained on a dataset different than the dataset used for tests.
Results and models are pushed to a S3 (can be disabled from an option of the Bash commmand line)

We use Optuna to tune models's hyper-parameters on X trials. For each trials we learn N models using N different Random Seeds (from 4242 up to 4242+N).

See in `Experiments.md` the list of experiments already created and their description.

+ Create or choose an experiment to use. Experiments are located in `configs/experiments`.
+ A Train/test experiment for Optuna are usually named XX-opt_algo_xxx (i.e. `01-opt_a2c_alphabeta_sigmoid`)
+ Run train/test experiment, i.e. training, model evaluation, selection of the best then test it, + optional push of results on S3:
    + `./expetraintest.sh --optuna <experiment_name>`
    + Optionally, produce a graph of the SLA performance tested with the best model produced: 
        + Edit `configs/simple_agent_test.yaml` to define the retained trial and the `minor_version` that will be, in this case, `01-opt_a2c_alphabeta_sigmoid` 
        + `python run_simple_test.py`
        + This should produce a file, i.e. `outputs/<experiment_folder>/agent-T27S0-test-sla_perf-SB3A2C_-1_1D-best.html`
+ Run experiment for a variability tests:
    + Experiments for variability tests are usually named algo_xxx_sweepseed (i.e. `a2c_best_sweepseed`)
    + The configuration file `a2c_best_sweepseed.yaml` must contain the specification of the best hyper-parameters retained with the best model we want to test
    + `./expetraintest.sh <experiment_name>`
    + This will produce:
        + X trials (10 usually) trained with a different random seed at each trial. Each trial generate 1 trained model. The best trial is also determined as done for a classic train/test experiment. Note tha optuna is not used here (a basic sweep from Hydra's definition)
        + If S3 is used, a folder is publish on S3 with the name '<experiment_folder>_disq'. It contains all trained models, the best and the disqualified one for the X trials.
    + Generate the graph of the performance in test for each model trained with a different randomm seed (shows variability):
        + Ensure (create it or pick up it on S3) that the folder `outputs/<experiment_folder>_disq' is present and contains all models for different seeds
        + Run `./expetestdisqualified.sh <experiment_name>` I.e. a2c_best_sweepseed
        + For example, this will generate and push to S3: `./outputs/E11_57_a2c_best_sweepseed_disq/agent-Tall-perfmeters_comp-SB3A2C_-1_1D-disq.html`

### Misc. examples
+ Manual steps execution:
    + `python run_agent.py +experiment=ppo ++n_trials=2 ++n_seeds=3`
    + `python run_best_agent.py +experiment=ppo`
    + `python run_best_agent_test.py +experiment=ppo`
    + `vi configs/simple_agent_test.yaml` # Indiquer les numéros de trial et seed du modèle, etc...
    + `python run_simple_test.py`
+ Ie., script for a complete experimentation chain with multi-models (train+eval, best agent extraction, best agent test, S3 storage):
    + `./expetraintest.sh linucb_kfoofw,ppo,dqn,a2c,ddpg,sac`
+ Ie., train and evaluate in trial mode without using experimentation (hydra meaning) configuration:
    + `python run_agent.py tuner=linucb_kfoofw n_seeds=2` # Pas de sweep d'hyper paramètres (1 seul trial), 2 random seed
+ Ie., Baseline Test with a specific reward choice:
    + `python run_baseline.py tuner=bfr tuner/reward=sigmoid`
+ Ie., Baseline Test with default reward and Sweep:
    + `python run_baseline.py tuner=bfr tuner.agent.policy.threshold='range(0.5,0.99, step=0.05)' n_trials=20`
+ Logs output destiation is in:
    + In case of expérimentation: `./outputs\E<version>_<version_minor>_<experiment_name>/logs`
    + In case of No experimentation: `./outputs/<version>_<version_minor>_<tuner_name>/logs`
+ Watch out results during execution:
    + `./lstrain.sh <log_dir>`. Ie. `./lstrain.sh outputs/11_21_ppo/logs`
    + Hydra's config of runs, ie.: `outputs/E11_50_opt_ppo_alphabeta_obsminimal_opt/logs/20241220_181058_run_agent_optuna3/.hydra`

+ In detail:
    + Generates models and "sweep" on hyper-parameters. Based on Optuna's sweeper, the script loops on N values of the random seed:
        + To see all possible experiments, look at `configs/experiments\*.yaml`
            + In case of Optuna sweeper usage, our convention is to name the experiments with the prefix `opt_`. Then, incude `base_optuna` as base defaults:
            + In case of Basic Sweeper, inclure instead `base`
        + `n_seeds` in `config/agent.yaml`: defines the count of RND_SEEDS per set of hyper-parameters
        + `n_jobs` s a sweeper parameer and works with `n_seeds` and specify the level of parallelization (from 1 to N). Recommendation: take care to the available vCPU.
        + `n_trials` in `config/base.yaml` specify the count of trials (as meant by Optuna sweeper)
        + Generate a model (and sweep): 
            + I.e. run the experimentation <experiment_name> in parallel on 4 local jobs: `python run_agent.py +experiment=<experiment_name> ++n_jobs=4`
        + `E11_4_ppo_termonstay/agent-T16S3-SB3PPO_-1_1D.pickle`: Agent on V11 dataset, Experimentation version 11 minor 4, 
            + Experimentation ppo_termonstay, PPO model, actions (arms for LinUCB) in range -1, 1, action space `D` for Discrete, or `C` for Continous. 
            + En: Experimentation, dataset n, see parameter `version`
            + Tn: Trial n, see parameter `n_trials` with n in range [0, n_trials-1]
            + Sn: Seed number n, see parameter `seeds` n in range [0, seeds-1]
        + Other exemples:
            + `python run_agent.py +experiment=linucb_kfoofw n_seeds=1` # Seulement 1 RND_SEED par trial
        + To reduce the number of jobs to sweep in same time, fix `hydra.sweeper.n_jobs=1`. Ie.:
            + `python run_agent.py +experiment=ppo xtraverbosity=1 ++hydra.sweeper.n_jobs=1` => No parallel sweep
    + Output the best model (with best regret at tests stage): `python run_best_agent.py +experiment=linucb_kfoofw`
        + Take care to use the same parameter than the one used at the previous steps with `run_agent.py`:
        + The best agent is renamed. Ie: `agentV9_expeJ1L0SB3PPO(-1, 1)D-best.pickle`
    + Logs ouputs into the folder `multirun/<date>-EN_n` or in:
        + If Experimentation:  `outputs\E<version>_<version_minor>_<experiment_name>/logs`
        + If Not an Experimentation:  `outputs\<version>_<version_minor>_<tuner_name>/logs`
    + Please report to the Hydra's documentation for the different command lie options


+ Othe exemples: 
    + `run_best_agent.py +experiment=linucb_kfoofw ++n_jobs=4 ++n_trials=10 ++n_seeds=10`

## Run Exploitation phase

This phase as for purpose to run an agent, in real-time, on an on-line database.

An on-line database is required, its configuration has to be updated in the configuration parameters. A example is the config file `config/db/xxx.yaml`.

At this time of the project life, the MariaDB database must run: 
+ as a standalone server, we do not yet support a Galera Cluster,
+ with a DB schema named `adbms`, this name is not yet put as a parameter, however our Python connection driver class has a parameter for that,
+ with a DB user named `adbms` with privileges:
    ```
        GRANT RELOAD ON *.* TO "adbms"@"%";
        GRANT BINLOG_ADMIN ON *.* TO "adbms"@"%";
        GRANT CONNECTION_ADMIN ON *.* TO "adbms"@"%" WITH GRANT OPTION;
        GRANT SYSTEM_VARIABLES_ADMIN ON *.* TO "adbms"@"%";
        GRANT Select,Update,Drop ON performance_schema.* TO "adbms"@"%";
    ```
+ Operating System (Linux) requirements are usual settings founds for a database server and are mainly:
    ```
        vm.swappiness=0
        vm.overcommit_memory=1
        vm.zone_reclaim_mode=0
        fs.aio-max-nr=2097152
    ```
+ See configs/db/variables/mariadb.yaml` to see the supported MariaDB version, and collected data from MariaDB `Global Status` and `Information Schema`. Other versions are not tested.

Running the Microtune agent:

+ Requires an up to date configuration defined in `config/best_agent_live.yaml` providing information on which trained model to use:
    + Ensure that output dir contains the model to use:
        + `<pickle_path>\<live_iterations_name>-xxx.pickle` is present as defined in config file (use `python run_best_agent_live.py --cfg job --resolve` if needed)
+ Run `python run_best_agent_live.py`.

It is up to you to load the database server as wanted and measure results. For the paper's purpose, we loaded the database with several Sysbench calls with different workload profiles. 

## Copyright and license

Code released under the [MIT License](https://github.com/Orange-OpenSource/microtune/blob/main/LICENSE).

Datasets released under the [Creative Commons](https://github.com/Orange-OpenSource/microtune/blob/main/LICENSE_CC), see [canonical URL](https://creativecommons.org/licenses/by-sa/4.0/).

