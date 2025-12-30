
# E16_21sall Intro

Experiments configuration overrides are located in Hydra's config folder: `configs/experiments`.

All latest experiments use the Distance to the Optimal Policy reward which is simpler than sigmoïd. Model's Hyperparams and ALPHA,BETA parameters are re-tuned here.

Determining best ALPHA and BETA values are done during Optuna optimisations of Hyperparametes, new trend is ALPHA closed to 0 and BETA over 0.5.

Some backuped experiments are located in `configs/experiments/bak...`


# Hyper-parameters optimization with Optuna

Expermiments names <experiment_name> are located in are prefixed either by `03-opt` or `04-opt`.

- "E16_21sall_03-opt-xxx", with Arms (-1, +6) Reward is OptimalPolicyDistance, Continous action space (default) :
	- PPO is best than A2C, best than DQN. LinUCB buggy (matrix inversion problem).
	- On observe des phénomènes d'oscillation dans la zone optimale mais le fait de pouvoir augmenter de +6 incréments de buffers, induit de grosses variations (sur-allocations) qu'on n'aurait peut-être pas avec un incrément limité à +1. Ex regarder ce qui se passe sous les 20ms: https://s3selfcare-vstune.s3-region01.cloudavenue.orange-business.com/E16_21sall_03-opt-dqn-n_arms/agent-T47S2-test-sla_perf-SB3DQN_-1_6D-best.html
	- Pendant les test (sla), le paramètre "deterministic" est positionné à False, on a bien des comportements différents pour 2 inputs identiques. Sur les expériences PPO avec "99-xxx" et "99-xxx-nd", on observe bien qu'il y a plus de fluctuations autour des valeurs cibles avec DETERMINISTIC=False (ND). Le résultat est finalement moins bon en DETERMINISTIC=True ^^ (plus de VIOLATIONS 618 vs 380), USLA (855 vs 536), CRAM proches
- "E16_21sall_04-opt-ppo-xxx", with Arms (-1, +6) Reward is OptimalPolicyDistance, Discrete action space (default) and DETERMINISTIC in test OFF :

# Basic sweep on parameters with constant hyper-parameters

Experiments `99-xxx-sweep-seeds` do NOT use Optuna but the Hydra basic sweeper to sweep over (usually 10) Seeds values only. 


