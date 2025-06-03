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

#qpslat_weights="01"   # QPS is not used in performance computation (weigth is 0%), the objective is to maintain te Latency (weigth is 100%). Other options is "19" (10% for QPS, 90% for Latency)

import os
#from multiprocessing import Lock
import logging
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import joblib

import hydrauti as hu
import bandits.datasource.adbms_dataframe as ds


# A logger for this file
log = logging.getLogger(__name__)
#lock = Lock()

def eval_baseline(cfg, datasets, baseline="oracle", use_real_perf=True):
    log.info(f"Load, if present, {baseline} to compare performances on Evaluation dataset...")
    _, _, ooptdict = hu.instanciate_agent_and_save(cfg, datasets=datasets, tuner=baseline, eval=True)

    bl_eval_perf_meter = ooptdict.get("perf_meter")

    return bl_eval_perf_meter

# Save best model, close environments and return Sweep Performance
# use_real_perf: Use "real" perf or regret to compare models against each others ?
def save_best_and_close(cfg, sweeper_params, trial, agent_results=[], datasets=None, use_real_perf=True) -> int:
    renderer = cfg.graph_renderer if cfg.graph_renderer != "None" else None
    best_res = None
    best_perf = np.inf
    eval_perf_list = []
    htmlfiles = []

    oracle_eval_perf_meter = eval_baseline(cfg, datasets=datasets, baseline="oracle", use_real_perf=use_real_perf)
    bfr_eval_perf_meter = eval_baseline(cfg, datasets=datasets, baseline="bfr", use_real_perf=use_real_perf)
    basic_eval_perf_meter = eval_baseline(cfg, datasets=datasets, baseline="basic", use_real_perf=use_real_perf)
    hpa_eval_perf_meter = eval_baseline(cfg, datasets=datasets, baseline="hpa", use_real_perf=use_real_perf)
    chr_eval_perf_meter = eval_baseline(cfg, datasets=datasets, baseline="chr", use_real_perf=use_real_perf)

    for res in agent_results:
        pm_eval = res["env_eval"].perf_meter
        cur_regret = res["env_eval"].getRegretPerformance()
        if use_real_perf:
            cur_performance = pm_eval.getSessionScalarPerformance(oracle_perf_meter=oracle_eval_perf_meter)
        else:
            cur_performance = cur_regret

        log.info(f'Env eval for seed={res["RND_SEED"]}, Performance: {cur_performance} CReg: {cur_regret} USLA,CRAM:{(pm_eval.getSessionPerformanceMultiObj())} VIOLATIONS:{sum(pm_eval.violations_count_per_ep)}')
        eval_perf_list.append(cur_performance)
        if cur_performance < best_perf:
            best_perf = cur_performance
            best_res = res
        else:
            res["env_train"].close()
            res["env_eval"].close()

    sid = best_res["sid"]
    RND_SEED = best_res["RND_SEED"]
    agent = best_res["agent"]
    env_train = best_res["env_train"]
    env_eval = best_res["env_eval"]

    filever = f'{cfg.iterations_name}{trial}S{sid}'

    # Save learning figures ?    
    if env_train:
        minmax_scaler = env_train.unwrapped.ds.min_max_scaler
        htmlgraph = agent.saveLearnFig(filepath=cfg.pickles_path, head_title=f"LEARN/{agent.policy.name} TotalSteps:{env_train.unwrapped.total_steps}", filever=filever+"-train")
        htmlfiles.extend(htmlgraph)

        # Show learning graph?
        if renderer != None:
            agent.showLearnFig(head_title=f"LEARN/{agent.policy.name} TotalSteps:{env_train.unwrapped.total_steps}", renderer=renderer)

        total_ep = env_train.unwrapped.cur_episode
        train_regret_cumsum = env_train._regret_per_episode[:total_ep].cumsum()

        htmlgraph = agent.savePredictFig(filepath=cfg.pickles_path, head_title=f"EVAL/{agent.policy.name} TotalSteps:{env_eval.unwrapped.total_steps}", filever=filever+"-eval")
        htmlfiles.extend(htmlgraph)
        train_reg_perf=env_train.getRegretPerformance()
        train_reg_results=env_train.resultPerWorkload()
    else:
        minmax_scaler = None
        train_reg_perf=None
        train_reg_results=None
        train_regret_cumsum = None

    total_ep = env_eval.unwrapped.cur_episode
    eval_regret_cumsum = env_eval._regret_per_episode[:total_ep].cumsum()
    eval_iregret_cumsum = env_eval._iregret_per_episode[:total_ep].cumsum()

    train_info = f"{env_train.unwrapped.msgStatus()} {env_train.unwrapped.desc()}" if env_train else "NA"
    graph_evals = env_eval.graphPerWorkload(f"Cumulated Regret/workload. Trained on: {train_info}")
    graph_evals.addCurve(f"{agent.policy.name} CReg:{round(env_eval.getRegretPerformance(),2)} S:{RND_SEED}", y=env_eval.resultPerWorkload(), perf=env_eval.getRegretPerformance())

    fig = graph_evals.figure()
    htmlgraph = os.path.join(cfg.pickles_path, f'{filever}-eval-creg_perWL-{agent.policy.name}.html')
    fig.write_html(htmlgraph)
    htmlfiles.append(htmlgraph)

#    if use_real_perf:
#        sweep_perf1 = np.average([ i[0] for i in eval_perf_list ]) # "under_sla" 
#        sweep_perf2 = np.average([ i[1] for i in eval_perf_list ]) # "memory" 
#        sweep_perf = (sweep_perf1, sweep_perf2)
#    else:
#        sweep_perf = np.average(eval_perf_list)
    sweep_perf = np.average(eval_perf_list)

    agent.save(filepath=cfg.pickles_path, filever=filever, verbose=1, optfiles=htmlfiles, 
                tid=trial, sid=sid, seed=RND_SEED, min_max_scaler=minmax_scaler,
                train_regrets=train_regret_cumsum, train_reg_perf=train_reg_perf, train_reg_results=train_reg_results, 
                eval_regrets=eval_regret_cumsum, eval_reg_perf=env_eval.getRegretPerformance(), 
                eval_iregrets=eval_iregret_cumsum, eval_ireg_perf=env_eval.getIRegretPerformance(), 
                sweeper_params=sweeper_params, sweep_perf=sweep_perf, eval_perf_meter=env_eval.perf_meter, oracle_eval_perf_meter=oracle_eval_perf_meter,
                bfr_eval_perf_meter=bfr_eval_perf_meter,
                basic_eval_perf_meter=basic_eval_perf_meter,
                hpa_eval_perf_meter=hpa_eval_perf_meter,
                chr_eval_perf_meter=chr_eval_perf_meter,
                config=OmegaConf.to_yaml(cfg, resolve=True))    

    log.info(f'Saved {filever} {agent.policy.name} Seed:{RND_SEED} Params:{sweeper_params} BestEvalPerf:{best_res["env_eval"].perf_meter.getSessionPerformanceMultiObj()} vs Oracle:{oracle_eval_perf_meter.getSessionPerformanceMultiObj()}')
    
    if not cfg.graph_keep_html:
        for ff in htmlfiles:
            os.remove(ff)

    if renderer != None:
        fig = graph_evals.figure()
        fig.show(renderer=renderer) # svg is cleaner but not in Gitlab. None is the best in VSCode

    if env_train:
        env_train.close()
    env_eval.close()

    log.info(f'Sweep with: {sweep_perf} with {eval_perf_list}') # (Best:{cur_performance})')

    return sweep_perf


def run_agent_by_seed(cfg, trial, sid, df_train, df_eval, label) -> dict:
    out = {}
    RND_SEED = cfg.RND_SEED+sid

    #TRAIN IT ?
    if cfg.tuner.env.wrapper:
        env_train = hu.instantiate_env_wrapper_wa(cfg.tuner.env, to_train=True, dataframe=df_train)
        agent = instantiate(cfg.tuner.agent, policy={"seed": RND_SEED}) #, "verbose": cfg.xtraverbosity})
    
        log.info(f'Train with sid:{sid} seed:{RND_SEED} Dataset coverage:{cfg.tuner.TRAINING_COVERAGE}')
        agent.learn(env_train, cfg.tuner.TRAINING_COVERAGE, verbose=cfg.verbosity)
        minmax_scaler = env_train.unwrapped.ds.min_max_scaler
    else:
        agent = instantiate(cfg.tuner.agent, policy={"seed": RND_SEED}) #, "verbose": cfg.xtraverbosity})
        minmax_scaler = None
        env_train = None
        log.info(f"No training for {agent.policy.name}")

    # /!\ Evaluate with evaluation DataSet !!!
    log.info(f'Evaluate trained model with sid:{sid} seed:{RND_SEED} Ep count: {cfg.tuner.TEST_EPISODES_COUNT}')
    env_eval = hu.instantiate_env_wrapper_wa(cfg.tuner.env, to_train=False, dataframe=df_eval, with_scaler=minmax_scaler, perf_meter_args={"name": f'{agent.policy.shortname}T{trial}S{sid}', "stage": "eval"})
    agent.predict(env_eval, episodes_max=cfg.tuner.TEST_EPISODES_COUNT, deterministic=cfg.DETERMINISTIC, verbose=cfg.xtraverbosity, label=label)

    out["sid"] = sid
    out["RND_SEED"] = RND_SEED
    out["agent"] = agent
    out["env_train"] = env_train
    out["env_eval"] = env_eval

    return out

from typing import Tuple

def run(cfg: DictConfig) -> Tuple[float, float]:
    log.info('===== RUN AGENT task =====')
    #logging.basicConfig(filename=cfg.logfile)
    #cfg.env.observation_space.elems.remove("sysbench_filtered.latency_mean") # Ensure latency is not used in context
    trial, sweeper_params = hu.prepare_sweeper_trial(cfg)

    datasets = instantiate(cfg.datasets)
    datasets.load()
    df_train = datasets.df_train
    df_eval = datasets.df_eval

    n_jobs = min(cfg.n_seeds, cfg.n_jobs)
    log.info(f"RunInfo: {cfg.run_info}")
    n_trials = "na" if cfg.n_trials <= 0 else cfg.n_trials
    log.info(f"#Trial:{trial+1}/{n_trials} Run {cfg.n_seeds} seeded training on {n_jobs} jobs...")
    # Parallel(n_jobs=n_jobs, prefer="threads")(
    agent_results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(run_agent_by_seed)(cfg, trial, sid, df_train, df_eval, f'Eval with {sweeper_params}') for sid in range(cfg.n_seeds))

    log.info(f"#Jobs:{n_jobs} #Results: {len(agent_results)}")
    perf = save_best_and_close(cfg, sweeper_params=sweeper_params, trial=trial, agent_results=agent_results, datasets=datasets, use_real_perf=cfg.use_real_perf)
    log.info(f"Perf: {perf}")

    return perf #[perf[0], perf[1]]

# Run with Hydra's basic sweeper
@hydra.main(version_base=None, config_path="configs", config_name="agent")
def run_with_basic(cfg: DictConfig) -> Tuple[float, float]:
    return run(cfg)


if __name__ == "__main__":
    run_with_basic()

