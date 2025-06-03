
# Intro

Experiments configuration overrides are located in Hydra's config folder: `configs/experiments`.

All latest experiments use the Sigmoïd reward that revealed its superiority againts reward V3E for example.

We do not list here, experiments done for:
    - determining best ALPHA and BETA values. We identified ALPHA=0.2 and BETA=0 are best.

Some backuped experiments are located in `configs/experiments/bak`


# Hyper-parameters optimization with Optuna

Expermiments names <experiment_name> are located in are prefixed either by `00-opt` or `01-opt`.

The folder name, created in `outputs` folder is `E11_56_<experiment_name>`

Hyper-parameters tuning experiments use Optuna on several trials and multiple seeds per trial, to find the best hyper-parameters.

Experiments config files located in `configs/exeperiments` are:

	- `00-opt_a2c_alphabeta_sigmoid`, gamma: range(0.90,0.996, step=0.004), Best hyper-params: (learning_rate=0.0002 gamma=0.908 n_steps=18)
	- `00-opt_ddpg_alphabeta_sigmoid`, (/!\ folder prefix `E11_55)`, Best hyper-params: (learning_rate=0.0007999999999999999 buffer_size=1000000 batch_size=512 gamma=0.94 noise_sigma=0.05 train_freq=1)
	- `01-opt_dqn_alphabeta_sigmoid`, gamma:range(0.01,1, step=0.05) => Best gamma evaluation is 0.76 (learning_rate=0.00054 batch_size=64 train_freq=128)
	- `01-opt_linucb_kfoofw_sigmoid_randomtrained_hlr`, learning_rate: range(0.8, 2, step=0.1), coeff exploration non cappé (2.0), best is alpha=1.4
	- /!\ `01-opt_ppo_alphabeta_sigmoid`, gamma: range(0.01,1, step=0.05), Best: (learning_rate=0.0005 batch_size=128 normalize_advantage=True n_epochs=3 gamma=0.81) => ATTENTION, résultats à reprendre dans papier
	- `00-opt_sac_alphabeta_sigmoid`, gamma: range(0.90,0.996, step=0.005), Best: (learning_rate=0.000119 buffer_size=2000000 batch_size=512 gamma=0.995 train_freq=49 gradient_steps=-1)

Not retained:
	- `00-opt_dqn_alphabeta_sigmoid`, gamma:range(0.90,0.996, step=0.005) => Not retained because gamma range is too small
	- `01-opt_dqn_alphabeta_sigmoid_gamma0`, gamma=0 => Not retained, gamma=0 is not a good idea
	- `00-opt_linucb_kfoofw_alphabeta_sigmoid`, learning_rate: range(0.5, 1.2, step=0.1), coeff exploration cappé, entrainement séquentiel. => Not retained, Sequential learning on dataset works badly
	- `01-opt_linucb_kfoofw_sigmoid_randomtrained`, learning_rate: range(0.1, 1.2, step=0.1), coeff exploration cappé, entrainement séquentiel  => Not retained, Alpha(badly named LR) has a too small range

# Basic sweep on parameters with constant hyper-parameters

Experiments with suffix `_sweepseed` and `_sweepBETA` do NOT use Optuna. 

Elle fixent les hyper-paramètres avec les valeurs qu'on a trouvé depuis les expériences d'optimisation Optuna `00-opt_xxx` ou `01-opt_xxx`.

Expériences sweep, `<version_prefix>_<hydra_experiment>`.

Experiments list (post-optimisation of hyper-parameters) for the variability tests regarding the choice of the Random Seed, listed by folder name (`E11_<minorversion><experiment_name>`):
	- `E11_56_a2c_alphabeta_sweepseed`,  
	- `E11_56_ddpg_alphabeta_sweepseed`,  
	- `E11_56_linucb_kfoofw_randomtrained_hlr_sweepseed`, sweepseed with exploration coef non cappé (alpha=1.4)
	- `E11_56_linucb_kfoofw_sweepBETA`, OK, clearly BETA must be set to 0
	- `E11_56_sac_alphabeta_sweepseed`,
	- *... `E11_57_dqn_alphabeta_gammaopti_sweepseed` gamma=0.76 (on s'est rendu compte que abaisser le gamma sous 0.9 est bénéfique)
	- * `E11_57_dqn_sweepBETA, use gamma=0.76
	- `E11_57_ppo_alphabeta_sweepseed`, with gamma=0.81 and other hyper parameters fixed
	- `E11_57_ppo_sweepBETA` => BETA=0 is the best, however, results do not decrease linearly while BETA increase up to 1 (like with LinUCB)

Not retained experiments:
	- `E11_56_dqn_alphabeta_sweepseed`, sweepseed avec un gamma=0.9
	- `E11_56_dqn_alphabeta_gamma0`, sweepseed avec gamma=0
	- /!\ `E11_56_ppo_alphabeta_sweepseed`, is wrong because of hyper parameters choices are not the good ones


Experiments with large gamma range, with `version_minor: 57 or 58` (*E11_57_xxx*):
	- 01-opt_a2c_alphabeta_sigmoid => OK, results sounds very good. Best hyper-params: (learning_rate=0.0001 n_steps=13 gamma=0.91 gae_lambda=0.2)
	- a2c_best_sweepseed On going => OK
	- 01-opt_ddpg_alphabeta_sigmoid => OK, interesting results. Best params: (learning_rate=0.0005 buffer_size=500000 batch_size=128 gamma=0.8600000000000001 tau=0.246 noise_sigma=0.35000000000000003 train_freq=33)
	- ddpg_best_sweepseed => OK
	- 01-opt_sac_alphabeta_sigmoid => KO, too long training time, stopped and results lost

Results give DQN and A2C the best NN algo. Interestingly, both are the only working with discrete Action Space (in Gym's env and passed to the SB3 policy). Other NN algo use a Continous Action Space (DDPG, SAC, PPO). We decide to run same PPO experiment (01-opt_ppo_alphabeta_sigmoid) with a discrete action space.

Experiment with Discrete continous action for PPO, in `version_minor: 58` (*E11_58_xxx*). Not put in our first research paper, while these results came too late:
	- 02-opt_ppo_sigmoid_discrete_as => Better results here! (learning_rate=0.0001 batch_size=64 normalize_advantage=False n_epochs=24 gamma=0.86)
	- ppo_as_discrete_sweepseed => In the same league as A2C and DQN, even a bit better.

