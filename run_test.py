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
import logging
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
import joblib

import bandits.datasource.adbms_dataframe as ds
#from bandits.datasource.dataframes.obs_samples_dataframes import ObsSamplesDF 

#from minio import Minio

#mm = Minio()

# A logger for this file
log = logging.getLogger(__name__)

# Save best model, close environments and return Sweep Performance
def save_best_and_close(cfg, sweeper_params, trial, agent_results=[]) -> int:
    renderer = cfg.graph_renderer if cfg.graph_renderer != "None" else None
    best_res = None
    best_perf = np.inf
    eval_perf = []
    htmlfiles = []

    for res in agent_results:
        cur_performance = res["env_eval"].getRegretPerformance()
        log.info(f'Env eval for seed={res["RND_SEED"]}, cumulated regret: {cur_performance}')
        eval_perf.append(cur_performance)
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
    minmax_scaler = env_train.unwrapped.ds.min_max_scaler

    filever = f'{cfg.iterations_name}{trial}S{sid}'
    htmlgraph = agent.saveLearnFig(filepath=cfg.pickles_path, head_title=f"LEARN/{agent.policy.name} TotalSteps:{env_train.unwrapped.total_steps}", filever=filever+"-train")
    htmlfiles.extend(htmlgraph)

    # Show learning graph?
    if renderer != None:
        agent.showLearnFig(head_title=f"LEARN/{agent.policy.name} TotalSteps:{env_train.unwrapped.total_steps}", renderer=renderer)

    total_ep = env_train.unwrapped.cur_episode
    train_regret_cumsum = env_train._regret_per_episode[:total_ep].cumsum()

    htmlgraph = agent.savePredictFig(filepath=cfg.pickles_path, head_title=f"EVAL/{agent.policy.name} TotalSteps:{env_eval.unwrapped.total_steps}", filever=filever+"-eval")
    htmlfiles.extend(htmlgraph)

    total_ep = env_eval.unwrapped.cur_episode
    eval_regret_cumsum = env_eval._regret_per_episode[:total_ep].cumsum()
    eval_iregret_cumsum = env_eval._iregret_per_episode[:total_ep].cumsum()

    graph_evals = env_eval.graphPerWorkload(f"Cumulated Regret/workload. Trained on: {env_train.unwrapped.msgStatus()} {env_train.unwrapped.desc()}")
    graph_evals.addCurve(f"{agent.policy.name} CReg:{round(env_eval.getRegretPerformance(),2)} S:{RND_SEED}", y=env_eval.resultPerWorkload(), perf=env_eval.getRegretPerformance())

    fig = graph_evals.figure()
    htmlgraph = os.path.join(cfg.pickles_path, f'{filever}-eval-creg_perWL-{agent.policy.name}.html')
    fig.write_html(htmlgraph)
    htmlfiles.append(htmlgraph)

    sweep_perf = np.average(eval_perf)

    train_lr=None #cfg.tuner.agent.policy.learning_rate
    agent.save(filepath=cfg.pickles_path, filever=filever, verbose=1, optfiles=htmlfiles, 
                tid=trial, sid=sid, seed=RND_SEED, min_max_scaler=minmax_scaler, train_lr=train_lr,
                train_regrets=train_regret_cumsum, train_reg_perf=env_train.getRegretPerformance(),
                train_reg_results=env_train.resultPerWorkload(), 
                eval_regrets=eval_regret_cumsum, eval_reg_perf=env_eval.getRegretPerformance(), 
                eval_iregrets=eval_iregret_cumsum, eval_ireg_perf=env_eval.getIRegretPerformance(), 
                sweeper_params=sweeper_params, sweep_perf=sweep_perf,
                config=OmegaConf.to_yaml(cfg, resolve=True))    

    log.info(f'Saved {filever} {agent.policy.name} Seed:{RND_SEED} Params:{sweeper_params} EvalRegPerf:{cur_performance}')
    
    if not cfg.graph_keep_html:
        for ff in htmlfiles:
            os.remove(ff)

    if renderer != None:
        fig = graph_evals.figure()
        fig.show(renderer=renderer) # svg is cleaner but not in Gitlab. None is the best in VSCode

    env_train.close()
    env_eval.close()

    log.info(f'Sweep with: {sweep_perf} in {eval_perf}') # (Best:{cur_performance})')

    return sweep_perf


def run_agent_by_seed(cfg, sid, df_train, df_eval) -> dict:
    out = {}
    RND_SEED = cfg.RND_SEED+sid
    htmlfiles = []

    #TRAIN   
    print(1) 
    env_train = instantiate(cfg.tuner.env.wrapper, env={'state_selector': {'df': df_train}})

    print(2) 
    agent = instantiate(cfg.tuner.agent, policy={"seed": RND_SEED}) #, "verbose": cfg.xtraverbosity})
 
    log.info(f'Train with sid:{sid} seed:{RND_SEED} Dataset coverage:{cfg.tuner.TRAINING_COVERAGE}')
    agent.learn(env_train, cfg.tuner.TRAINING_COVERAGE, verbose=cfg.verbosity)
    minmax_scaler = env_train.unwrapped.ds.min_max_scaler

    # /!\ Evaluate with evaluation DataSet !!!
    log.info(f'Evaluate trained model with sid:{sid} seed:{RND_SEED} Ep count: {cfg.tuner.TEST_EPISODES_COUNT}')
    env_eval = instantiate(cfg.tuner.env.wrapper_test, env={'state_selector': {'df': df_eval, 'with_scaler': minmax_scaler}})
    agent.predict(env_eval, episodes_max=cfg.tuner.TEST_EPISODES_COUNT, deterministic=cfg.DETERMINISTIC, verbose=cfg.xtraverbosity)

    out["sid"] = sid
    out["RND_SEED"] = RND_SEED
    out["agent"] = agent
    out["env_train"] = env_train
    out["env_eval"] = env_eval

    return out

import pandas as pd
import hydrauti as hu

@hydra.main(version_base=None, config_path="configs", config_name="agent")
def run(cfg: DictConfig) -> float:
    hydracfg = HydraConfig.get()
    if str(hydracfg.mode) == "RunMode.MULTIRUN":
        link2create = cfg.link2logs+str(hydracfg.job.num)
    else:
        link2create = cfg.link2logs
    log.info(f'link2create:{link2create}')
    log.info(f"Output directory  : {hydracfg.runtime.output_dir}")
    log.info(f"Mode  : {hydracfg.mode}")
    log.info(f"job.name  : {hydracfg.job.name}")
    #print(hydracfg)
    #print("##########")
    #print(hydracfg.runtime)


#    print(f'hydra.run.dir:{hydra.run.dir}')
#    print(f'hydra.sweep.dir:{hydra.sweep.dir}')
#    print(hu.load_baseline(cfg))

    hash = hu.configId(cfg)
    print(hash)

    #trial, sweeper_params = hu.prepare_sweeper_trial(cfg)

    #print(f"Trial:{trial} sweep params:{sweeper_params}")
    # Performs only 1 trial
    #if hu.evict_sweeper_trial():
    #    return np.inf
    
    # TEST  ONLY
    datasets = instantiate(cfg.datasets)
    datasets.load(test_only=False)
    df = datasets.df_test
    ALL_DB_SIZES = df["db_size_mb"].unique()
    log.info(f'All DB sizes: {ALL_DB_SIZES}')
    df = datasets.df_train
    ALL_DB_SIZES = df["db_size_mb"].unique()
    log.info(f'All DB sizes: {ALL_DB_SIZES}')
    df = datasets.df_eval
    ALL_DB_SIZES = df["db_size_mb"].unique()
    log.info(f'All DB sizes: {ALL_DB_SIZES}')

    return 0

    log.info("Evaluate (once for all) Oracle to establish performance baseline on Test dataset...")
    oracle = instantiate(cfg.oracle.agent)
    env_oracle = hu.instantiate_env_wrapper_wa(cfg.oracle.env, to_train=False, dataframe=df_test, perf_meter_args={"name": f'{oracle.policy.shortname}, "stage": "test"'})
    assert oracle.policy.context() == env_oracle.unwrapped.ds.contextElems(), 'Discrepancy between Oracle policy context and env context...'
    oracle.predict(env_oracle, episodes_max=cfg.oracle.TEST_EPISODES_COUNT, verbose=cfg.verbosity)

    agent = instantiate(cfg.tuner.agent)
    #            tid=trial, sid=sid, seed=RND_SEED, min_max_scaler=minmax_scaler,
    #            train_regrets=train_regret_cumsum, train_reg_perf=train_reg_perf, train_reg_results=train_reg_results, 
    #            eval_regrets=eval_regret_cumsum, eval_reg_perf=env_eval.getRegretPerformance(), 
    #            eval_iregrets=eval_iregret_cumsum, eval_ireg_perf=env_eval.getIRegretPerformance(), 
    #            sweeper_params=sweeper_params, sweep_perf=sweep_perf, eval_perf=eval_perf, eval_perf_meter=env_eval.perf_meter, oracle_eval_perf_meter=oracle_perf_meter,
    #            config=OmegaConf.to_yaml(cfg, resolve=True))    
    files_list, optdict = agent.load(filepath=cfg.pickles_path, filever=f'{cfg.iterations_name}*', verbose=0, dftext='-best.pickle')
    for ff in files_list:
        os.remove(ff)
    log.info(f'Agent Filename: {agent.filename}')
    log.info(f'Agent learning params: {optdict["sweeper_params"]}')
    trial = optdict["tid"]
    sid = optdict["sid"]
    scaler = optdict["min_max_scaler"]
    if scaler:
        print(f'MinMaxScaler:{type(scaler)}')

    env_test = hu.instantiate_env_wrapper_wa(cfg.tuner.env, to_train=False, dataframe=df_test, with_scaler=scaler, perf_meter_args={"name": f'{agent.policy.shortname}T{trial}S{sid}', "stage": "test"})
    assert agent.policy.context() == env_test.unwrapped.ds.contextElems(), 'Discrepancy between policy context and env context...'
    log.info(f"Agent {agent.policy.name}...")
    agent.predict(env_test, episodes_max=cfg.tuner.TEST_EPISODES_COUNT, deterministic=cfg.DETERMINISTIC, verbose=cfg.verbosity, baseline_perf_meter=env_oracle.perf_meter)
    filever = f'{cfg.iterations_name}{optdict["tid"]}S{optdict["sid"]}-test'
    htmlgraph = agent.savePredictFig(filepath=cfg.pickles_path, head_title=f"TEST/{agent.policy.name} TotalSteps:{env_test.unwrapped.total_steps}", 
                                     filever=filever, ext='-best.html')
    print(f'Saved html Pred: {htmlgraph}')

    return np.inf


if __name__ == "__main__":
    run()

