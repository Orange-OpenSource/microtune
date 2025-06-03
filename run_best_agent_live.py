"""
/*
 * Software Name : Microtune
 * SPDX-FileCopyrightText: Copyright (c) Orange SA
 * SPDX-License-Identifier: MIT
 *
 * This software is distributed under the MIT license,
 * see the "LICENSE" file for more details
 *
 * Authors: see CONTRIBUTORS.md
 * Software description: MicroTune is a RL-based DBMS Buffer Pool Auto-Tuning for Optimal and Economical Memory Utilization. Consumed RAM is continously and optimally adjusted in conformance of a SLA constraint (maximum mean latency).
 */
"""
import os
import numpy as np

import logging
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

import hydrauti as hu
from bandits.gym_env import VSEnv, VSMonitorCallback

# Command: python run_best_agent_test.py +experiment=linucb_kfoofw ++hydra.sweeper.n_trials=1 verbosity=1

# A logger for this file
log = logging.getLogger(__name__)

class LiveVSMonitorCallback(VSMonitorCallback):
    def __init__(self):
        pass

    def onStart(self, env: VSEnv):
        # Happens only when training, test or inference starts
        log.info('Start Live session')

    def onReset(self, env: VSEnv):
        log.info('Reset Live session')

    def onStep(self, env: VSEnv, info: dict={}):
        log.info('Step Live session')
        log.info(f'ObsElems:{env.ds.contextElems()}')
        log.info(f'ObsData:{env.ds.context()}')


@hydra.main(version_base=None, config_path="configs", config_name="best_agent_live")
def run(cfg: DictConfig) -> None:
    #logging.basicConfig(filename=cfg.logfile)
    # Performs only 1 trial from any Hydra launcher
    if not hu.prepare_run(cfg):
        return np.inf

    # TEST  ONLY
    agent = instantiate(cfg.tuner.agent)
    #            tid=trial, sid=sid, seed=RND_SEED, min_max_scaler=minmax_scaler,
    #            train_regrets=train_regret_cumsum, train_reg_perf=train_reg_perf, train_reg_results=train_reg_results, 
    #            eval_regrets=eval_regret_cumsum, eval_reg_perf=env_eval.getRegretPerformance(), 
    #            eval_iregrets=eval_iregret_cumsum, eval_ireg_perf=env_eval.getIRegretPerformance(), 
    #            sweeper_params=sweeper_params, sweep_perf=sweep_perf, eval_perf_meter=env_eval.perf_meter, oracle_eval_perf_meter=oracle_perf_meter,
    #            eval_perf_meter_list=[],
    #            config=OmegaConf.to_yaml(cfg, resolve=True))    
    _, optdict = agent.load(filepath=cfg.pickles_path, filever=f'{cfg.live_iterations_name}*', verbose=0, dftext='.pickle')

    log.info(f'Agent Filename: {agent.filename}')
    log.info(f'Agent learning params: {optdict["sweeper_params"]}')
    scaler = optdict["min_max_scaler"]
    if scaler:
        print(f'MinMaxScaler:{type(scaler)}')

    log.info("Test Live DB with agent...")
    env_test = hu.instantiate_env_wrapper_live_wa(cfg.tuner.env, with_scaler=scaler)
    assert agent.policy.context() == env_test.unwrapped.ds.contextElems(), 'Discrepancy between policy context and env context...'

    log.info(f"Agent {agent.policy.name}...")
    agent.predict(env_test, episodes_max=cfg.tuner.TEST_EPISODES_COUNT, deterministic=cfg.DETERMINISTIC, verbose=cfg.verbosity, baseline_perf_meter=None, label=f'Best one:{optdict["sweeper_params"]}')

    return 0


if __name__ == "__main__":
    run()


