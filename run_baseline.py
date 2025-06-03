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
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig

from bandits.policy_linucb_kfoofw import LinUCBPolicy_kfoofw
from bandits.agent import VSAgent
import bandits.datasource.adbms_dataframe as ds
import hydrauti as hu

# Command: python run_best_agent_test.py +experiment=linucb_kfoofw ++hydra.sweeper.n_trials=1 verbosity=1

# A logger for this file
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="baseline")
def run(cfg: DictConfig) -> None:
    trial, sweeper_params = hu.prepare_sweeper_trial(cfg)

    # DATA TEST  
    datasets = instantiate(cfg.datasets)
    datasets.load(test_only=False)
    df_test = datasets.df_test
    #df_test = datasets.df_train

    # ORACLE
    oracle = instantiate(cfg.oracle.agent)
    env_oracle = hu.instantiate_env_wrapper_wa(cfg.oracle.env, to_train=False, dataframe=df_test, perf_meter_args={"name": f'{oracle.policy.shortname}', "stage": "test"})
    assert oracle.policy.context() == env_oracle.unwrapped.ds.contextElems(), 'Discrepancy between Oracle policy context and env context...'
    log.info("Run Oracle...")
    oracle.predict(env_oracle, episodes_max=cfg.oracle.TEST_EPISODES_COUNT, verbose=cfg.verbosity)

    # BASELINE
    agent = instantiate(cfg.tuner.agent)
    env_test = hu.instantiate_env_wrapper_wa(cfg.tuner.env, to_train=False, dataframe=df_test, perf_meter_args={"name": f'{agent.policy.shortname}T{trial}S0', "stage": "test"})

    assert agent.policy.context() == env_test.unwrapped.ds.contextElems(), 'Discrepancy between policy context and env context...'

    log.info(f"Trial:{trial} Run baseline {agent.policy.name} test on {cfg.n_jobs} jobs...")
    agent.predict(env_test, episodes_max=cfg.tuner.TEST_EPISODES_COUNT, deterministic=cfg.DETERMINISTIC, verbose=cfg.verbosity, baseline_perf_meter=env_oracle.perf_meter)
    filever = f'{cfg.iterations_name}{trial}S0-test'
    htmlgraph = agent.savePredictFig(filepath=cfg.pickles_path, head_title=f"TEST/{agent.policy.name} TotalSteps:{env_test.unwrapped.total_steps}", 
                                     filever=filever, ext='-best.html')
    print(f'Saved html Pred: {htmlgraph}')

    graph_tests = env_test.graphPerWorkload(f'Cumulated Regret/workload Sweep:{sweeper_params}')
    perf_name = env_test.perf_meter.name
    perf = env_test.perf_meter.getSessionPerformanceMultiObj()
    perf = tuple(map(lambda x: round(x, 3), perf))
    oracle_perf = env_oracle.perf_meter.getSessionPerformanceMultiObj()
    oracle_perf = tuple(map(lambda x: round(x, 3), oracle_perf))
    graph_tests.addCurve(f"{perf_name} CReg:{round(env_test.getRegretPerformance(),2)} Perf(U-SLA,CRAM):{perf} vs Oracle:{oracle_perf}", y=env_test.resultPerWorkload(), perf=env_test.getRegretPerformance())

    fig = graph_tests.figure()
    htmlgraph = os.path.join(cfg.pickles_path, f'{filever}-creg_perWL-{agent.policy.name}-best.html')
    fig.write_html(htmlgraph)
    print(f'Saved html Cumulated Regret per WL: {htmlgraph}')

    results_kpi = env_test.perf_meter.getSessionPerformanceKPIs( env_oracle.perf_meter)
    log.info(f'Results KPI on TEST, ScalarPerf, (USLA,RAM,SLAV): {results_kpi}')

    graph_perf = agent.graph_perf
    if graph_perf:
        fig = graph_perf.figure()
        htmlgraph = os.path.join(cfg.pickles_path, f'{filever}-perfmetrics_perWL-{agent.policy.name}-best.html')
        fig.write_html(htmlgraph)
        print(f'Saved Performance Metrics Graph: {htmlgraph}')

    return env_test.getRegretPerformance()


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
