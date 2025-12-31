
# E16_21sall Intro

Experiments configuration overrides are located in Hydra's config folder: `configs/experiments`.

Sub-folders with configurations like `configs/experiments/bakxxx/yyy.yaml` cannot be used directly. Sub-folders are there to be used as backuped experimentations.

All latest experiments use the Distance to the Optimal Policy (`optimal_policy_distance`) reward which is simpler than sigmoïd. Model's Hyperparams and ALPHA,BETA parameters are re-tuned here.

Determining best ALPHA and BETA values are done during Optuna optimisations of Hyperparametes, new trend is ALPHA closed to 0 and BETA over 0.5.


# Hyper-parameters optimization with Optuna

Expermiments names <experiment_name> are located in are prefixed either by `03-opt` or `04-opt`.

- "E16_21sall_03-opt-xxx", with Arms (-1, +6) Reward is OptimalPolicyDistance, Continous action space (default) :
	- PPO is best than A2C, best than DQN. LinUCB buggy (matrix inversion problem).
	- We observe oscillation phenomena in the optimal zone, but the ability to increase buffer size by +6 increments induces large variations (over-allocations) that might not occur with an increment limited to +1. For example, look at what happens below 20ms.: https://s3selfcare-vstune.s3-region01.cloudavenue.orange-business.com/E16_21sall_03-opt-dqn-n_arms/agent-T47S2-test-sla_perf-SB3DQN_-1_6D-best.html
	- During the tests (SLA), with the "deterministic" parameter set to False, we observed different behaviors for two identical inputs. In the PPO experiments with "99-xxx" and "99-xxx-nd", we clearly saw more fluctuations around the target values ​​with DETERMINISTIC=False (ND). The result was ultimately worse with DETERMINISTIC=True (more VIOLATIONS: 618 vs. 380), USLA (855 vs. 536), and CRAM values ​​close.
- "E16_21sall_04-opt-ppo-xxx", Discrete ActionSpace + Hyper params & ALPHA,BETA optim with Optuna + rew=optimal_policy_distance + Arms (-1,+6) + NOT DETERMINISTIC in tests (default)
- "E16_21sall_04-opt-ppo-n_arms-ot", Discrete ActionSpace + Hyper params & ALPHA,BETA optim with Optuna + rew=optimal_policy_distance + Arms (-1,+6) + NOT DETERMINISTIC in tests (default) + on_terminate rewarding. The 'on_terminate' option in Env, has for purpose to reward more when the agent decide to use the STAY arm with NO Regret! 
If on_terminate>=0 we'll Stop the episode when STAY action is done after on_terminate times, without any regret! In such a case, the reward is increased by adding to the reward value count of on_terminate subsequents True conditions
- All digital twin experiments are named as follow '<simuname>-<modelname>-training'. the simu named 'orig' use a dataset not simulated.


# Basic sweep on parameters with constant hyper-parameters

Experiments `99-xxx-sweep-seeds` do NOT use Optuna but the Hydra basic sweeper to sweep over (usually 10) Seeds values only. 


