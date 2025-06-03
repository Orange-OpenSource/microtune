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
from bandits.graph import GraphPerfMeterComparaison

# Test all agents of all trials on Test Dataset and produce a graph to compare them
# Command: python run_disq_agent_test.py +experiment=<sweepseed_experiment> ++n_trials=N

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="all_agents_test")
def run(cfg: DictConfig) -> None:
    #logging.basicConfig(filename=cfg.logfile)
    # Performs only 1 trial
    if not hu.prepare_run(cfg):
        return np.inf

    test_perf_meter_list = []

    # TEST  ONLY
    datasets = instantiate(cfg.datasets)
    datasets.load(test_only=True)
    df_test = datasets.df_test

    log.info("Test (normaly once) Oracle to establish performance baseline on Test dataset...")
    oracle = instantiate(cfg.oracle.agent)
    env_oracle = hu.instantiate_env_wrapper_wa(cfg.oracle.env, to_train=False, dataframe=df_test, perf_meter_args={"name": f'{oracle.policy.shortname}', "stage": "test"})
    assert oracle.policy.context() == env_oracle.unwrapped.ds.contextElems(), 'Discrepancy between Oracle policy context and env context...'
    oracle.predict(env_oracle, episodes_max=cfg.oracle.TEST_EPISODES_COUNT, verbose=cfg.verbosity, label="Test")

    baseline = instantiate(cfg.baseline.agent)
    env_baseline = hu.instantiate_env_wrapper_wa(cfg.baseline.env, to_train=False, dataframe=df_test, perf_meter_args={"name": f'{baseline.policy.shortname}', "stage": "test"})
    assert baseline.policy.context() == env_baseline.unwrapped.ds.contextElems(), 'Discrepancy between BL policy context and env context...'
    log.info(f"Test Baseline {baseline.policy.name}...")
    baseline.predict(env_baseline, episodes_max=cfg.baseline.TEST_EPISODES_COUNT, verbose=cfg.verbosity, label="Test")
    test_perf_meter_list.append(env_baseline.perf_meter)

    baseline2 = instantiate(cfg.baseline2.agent)
    env_baseline2 = hu.instantiate_env_wrapper_wa(cfg.baseline2.env, to_train=False, dataframe=df_test, perf_meter_args={"name": f'{baseline2.policy.shortname}', "stage": "test"})
    assert baseline2.policy.context() == env_baseline2.unwrapped.ds.contextElems(), 'Discrepancy between BL2 policy context and env context...'
    log.info(f"Test Baseline2 {baseline2.policy.name}...")
    baseline2.predict(env_baseline2, episodes_max=cfg.baseline2.TEST_EPISODES_COUNT, verbose=cfg.verbosity, label="Test")
    test_perf_meter_list.append(env_baseline2.perf_meter)

    baseline3 = instantiate(cfg.baseline3.agent)
    env_baseline3 = hu.instantiate_env_wrapper_wa(cfg.baseline3.env, to_train=False, dataframe=df_test, perf_meter_args={"name": f'{baseline3.policy.shortname}', "stage": "test"})
    assert baseline3.policy.context() == env_baseline3.unwrapped.ds.contextElems(), 'Discrepancy between BL2 policy context and env context...'
    log.info(f"Test Baseline2 {baseline3.policy.name}...")
    baseline3.predict(env_baseline3, episodes_max=cfg.baseline3.TEST_EPISODES_COUNT, verbose=cfg.verbosity, label="Test")
    test_perf_meter_list.append(env_baseline3.perf_meter)

    agent = instantiate(cfg.tuner.agent)
    log.info(f'# Trials:{cfg.n_trials}')
    for trial in range(cfg.n_trials):
        agent = instantiate(cfg.tuner.agent)

        filever = f'{cfg.iterations_name}{trial}S*'
        _, optdict = agent.load(filepath=cfg.pickles_path, filever=filever, verbose=0, dftext='.pickle')

        log.info(f'AgentFilename: {agent.filename}')
        log.info(f'AgentLearningParams: {optdict["sweeper_params"]}')
        scaler = optdict["min_max_scaler"]
        if scaler:
            log.info(f'MinMaxScaler:{type(scaler)}')

        env_test = hu.instantiate_env_wrapper_wa(cfg.tuner.env, to_train=False, dataframe=df_test, with_scaler=scaler, perf_meter_args={"name": f'{agent.policy.shortname}T{trial}S{optdict["sid"]}', "stage": "test"})
        assert agent.policy.context() == env_test.unwrapped.ds.contextElems(), 'Discrepancy between policy context and env context...'
        log.info(f"Trial: {trial} TestAgent: {agent.policy.name} ...")
        agent.predict(env_test, episodes_max=cfg.tuner.TEST_EPISODES_COUNT, deterministic=cfg.DETERMINISTIC, verbose=cfg.verbosity, baseline_perf_meter=env_oracle.perf_meter, label=f'Best one:{optdict["sweeper_params"]}')
        log.info(f'Trial: {trial} TestPerf(ScalarPerf,USLA,RAM,SLAV): {(env_test.perf_meter.getSessionPerformanceKPIs(env_oracle.perf_meter))}')
        test_perf_meter_list.append(env_test.perf_meter)

    log.info("Graph all TEST perf meters...")
    log.info(f'LengthPerfMeterlist: {len(test_perf_meter_list)}')
    graph_perf_eval = GraphPerfMeterComparaison(oracle_perf_meter=env_oracle.perf_meter, perf_meter_list=test_perf_meter_list, title=cfg.run_info, stage="TEST")
    fig = graph_perf_eval.figure()
    htmlgraph = os.path.join(cfg.pickles_path, f'{cfg.iterations_name}all-perfmeters_comp-{agent.policy.name}-disq.html')
    fig.write_html(htmlgraph)
    log.info(f'Saved html Perf Meters in Test: {htmlgraph}')

    return 0


if __name__ == "__main__":
    run()


