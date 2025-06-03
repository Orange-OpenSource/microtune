"""
/*
 * Software Name : Microtune
 * SPDX-FileCopyrightText: Copyright (c) Orange SA
 * SPDX-License-Identifier: MIT
 *
 * This software is distributed under the <license-name>,
 * see the "LICENSE.txt" file for more details or <license-url>
 *
 * <Authors: optional: see CONTRIBUTORS.md
 * Software description: MicroTune is a RL-based DBMS Buffer Pool Auto-Tuning for Optimal and Economical Memory Utilization. Consumed RAM is continously and optimally adjusted in conformance of a SLA constraint (maximum mean latency).
 */
"""
import os
import glob
import re
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig

from bandits.agent import AgentsLoaderLoop
import bandits.datasource.adbms_dataframe as ds
import hydrauti as hu

# A logger for this file
import logging
log = logging.getLogger(__name__)

# Get dictionary val in a context where missing key can be ignored
def dgv(optdict={}, key=""):
    val = optdict.get(key)
    return "NA" if val is None else val

@hydra.main(version_base=None, config_path="configs", config_name="best_agent")
def run(cfg: DictConfig) -> None:
    log.info('===== RUN BEST AGENT task =====')
    #logging.basicConfig(filename=cfg.logfile)
    # Performs only 1 trial from any Hydra launcher
    if not hu.prepare_run(cfg):
        return np.inf
    
    n_trials = 1 #cfg.n_trials #HydraConfig.get().sweeper.n_trials

    # THIS beacuse we run this command with n_trials=1 while experiment is still not completed may be?
    if n_trials == 1:
        dummy_agent = instantiate(cfg.tuner.agent)
        #tor=r'\((-?\d+),\s*(-?\d+)\)'
        #dummy_name = re.sub(tor, '*', dummy_agent.policy.name)
        dummy_name = dummy_agent.policy.name
        print(f'Search for {cfg.pickles_path}/{cfg.iterations_name}*{dummy_name}.pickle ...')
        lst = glob.glob(f'{cfg.pickles_path}/{cfg.iterations_name}*{dummy_name}.pickle')
        n_trials = len(lst)
        if n_trials > 0:
            print(f'Found {n_trials} pickles ex: {lst[0]}')
        else:
            return np.inf

    remaining_best_agent = None
    best_trial_perf = np.inf
    best_trial = -1
    best_sid = -1
    best_trial_seed = -1
    agent_eval_perf_meter_list=[]
    oracle_eval_perf_meter = None

    log.info(f'# Trials:{n_trials}')
    for trial in range(n_trials):
        agent = instantiate(cfg.tuner.agent)

        filever = f'{cfg.iterations_name}{trial}S*'
        htmlfiles, optdict = agent.load(filepath=cfg.pickles_path, filever=filever, verbose=cfg.xtraverbosity)

        for ff in htmlfiles:
            os.remove(ff)

        # OPTDICT is:
        #        tid=trial, sid=sid, seed=RND_SEED, min_max_scaler=minmax_scaler, train_lr=train_lr,
        #        train_regrets=train_regret_cumsum, train_reg_perf=env_train.getRegretPerformance(),
        #        train_reg_results=env_train.resultPerWorkload(), 
        #        eval_regrets=eval_regret_cumsum, eval_reg_perf=env_eval.getRegretPerformance(), 
        #        eval_iregrets=eval_iregret_cumsum, eval_ireg_perf=env_eval.getIRegretPerformance(), 
        #        sweeper_params=sweeper_params, sweep_perf=sweep_perf, eval_perf_meter=env_eval.perf_meter, oracle_eval_perf_meter=oracle_perf_meter,
        #        config=OmegaConf.to_yaml(cfg, resolve=True))
        best_sid =  dgv(optdict,"sid")
        eval_perf_meter = optdict["eval_perf_meter"]          
        agent_eval_perf_meter_list.append(eval_perf_meter)

        if trial==0:
            oracle_eval_perf_meter = optdict["oracle_eval_perf_meter"]
            opm_tuple = oracle_eval_perf_meter.getSessionPerformanceMultiObj()
            oracle_perf = np.array(opm_tuple)

        # Get on axis normed distance to oracle for the sweep perf
        sweep_perf = optdict["sweep_perf"]
        log.info(f'T{trial} S{best_sid} LR:{dgv(optdict,"train_lr")} TrainRegPerf:{dgv(optdict,"train_reg_perf")} EvalRegPerf:{dgv(optdict,"eval_reg_perf")} EvalPerfMO:{eval_perf_meter.getSessionPerformanceMultiObj()} SweepPerf:{sweep_perf}')
        # Is it the best trial?
        if sweep_perf < best_trial_perf:
            best_trial_perf = sweep_perf
            best_trial = trial
            best_trial_seed = optdict["sid"]
            remaining_best_agent = agent
        else:
            log.info(f'T{trial} Cur:{sweep_perf} Oracle:{oracle_perf} SweepPerf:{sweep_perf}')

    if remaining_best_agent:
        log.info(f'BEST is #Trial: {best_trial} #S:{best_trial_seed} Best seed performance (MO): {eval_perf_meter.getSessionPerformanceMultiObj()} Best Sweep perf:{best_trial_perf}')
        bestfilename = remaining_best_agent.filename.replace('.pickle', '-best.pickle')
        log.info(f"BEST FILE {bestfilename}")
        #os.system(f'cp "{remaining_best_agent.filename}"  "{bestfilename}"')
        agent = instantiate(cfg.tuner.agent)
        files_list, optdict = agent.load(filepath=cfg.pickles_path, filever=f'{cfg.iterations_name}{best_trial}*', verbose=cfg.xtraverbosity)
        best_files_list = []
        print(f"Move HTML files {files_list} with -best extension...")
        for htmlfile in files_list:
            bestfilename = htmlfile.replace('.html', '-best.html')
            os.system(f'mv "{htmlfile}"  "{bestfilename}"')
            best_files_list.append(bestfilename)

        filever=f'{cfg.iterations_name}{best_trial}S{optdict["sid"]}'
        optdict["eval_perf_meter_list"] = agent_eval_perf_meter_list
        agent.save(filepath=cfg.pickles_path, filever=filever, ext='-best.pickle', verbose=1, optfiles=best_files_list, **optdict)

    else:
        log.info('No best agent found!!')
        return np.inf

    return best_trial_perf


if __name__ == "__main__":
    run()


# A tuple:
#    "scheme": "big-40-300000",
#    "tables": tables,
#    "rows": rows,
#    "db_size_mb": max(dbsz_list),
#    "wl_clients": wl_clients,
#    "randtype":
#    "displayable": True/False (empty or not)