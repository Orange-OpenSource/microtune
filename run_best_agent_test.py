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

# Command: python run_best_agent_test.py +experiment=linucb_kfoofw ++hydra.sweeper.n_trials=1 verbosity=1

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="best_agent_test")
def run(cfg: DictConfig) -> None:
    log.info('===== RUN BEST AGENT TEST task =====')
    #logging.basicConfig(filename=cfg.logfile)
    # Performs only 1 trial from any Hydra launcher
    if not hu.prepare_run(cfg):
        return np.inf

    test_perf_meter_list = []
    eval_perf_meter_list = []
    
    # TEST  ONLY
    datasets = instantiate(cfg.datasets)
    datasets.load(test_only=True)
    df_test = datasets.df_test

    log.info("Test (normaly once) Oracle to establish performance baseline on Test dataset...")
    oracle = instantiate(cfg.oracle.agent)
    env_oracle = hu.instantiate_env_wrapper_wa(cfg.oracle.env, to_train=False, dataframe=df_test, perf_meter_args={"name": f'{oracle.policy.shortname}', "stage": "test"})
    assert oracle.policy.context() == env_oracle.unwrapped.ds.contextElems(), 'Discrepancy between Oracle policy context and env context...'
    oracle.predict(env_oracle, episodes_max=cfg.oracle.TEST_EPISODES_COUNT, verbose=cfg.verbosity, label="Test")

    agent = instantiate(cfg.tuner.agent)
    #            tid=trial, sid=sid, seed=RND_SEED, min_max_scaler=minmax_scaler,
    #            train_regrets=train_regret_cumsum, train_reg_perf=train_reg_perf, train_reg_results=train_reg_results, 
    #            eval_regrets=eval_regret_cumsum, eval_reg_perf=env_eval.getRegretPerformance(), 
    #            eval_iregrets=eval_iregret_cumsum, eval_ireg_perf=env_eval.getIRegretPerformance(), 
    #            sweeper_params=sweeper_params, sweep_perf=sweep_perf, eval_perf_meter=env_eval.perf_meter, oracle_eval_perf_meter=oracle_perf_meter,
    #            eval_perf_meter_list=[],
    #            config=OmegaConf.to_yaml(cfg, resolve=True))    
    _, optdict = agent.load(filepath=cfg.pickles_path, filever=f'{cfg.iterations_name}*', verbose=0, dftext='-best.pickle')

    log.info(f'Agent Filename: {agent.filename}')
    log.info(f'Agent learning params: {optdict["sweeper_params"]}')
    trial = optdict["tid"]
    sid = optdict["sid"]
    scaler = optdict["min_max_scaler"]
    if scaler:
        log.info(f'MinMaxScaler:{type(scaler)}')

    # Display Eval perf if available ?
    pm = optdict.get("eval_perf_meter")
    if pm:
        log.info(f'Sweep eval perf: {optdict["sweep_perf"]} Model eval perf(MO): {pm.getSessionPerformanceMultiObj()} #Other Eval perfs to display:{len(optdict["eval_perf_meter_list"])}')
        eval_perf_meter_list.extend(optdict["eval_perf_meter_list"])
        eval_perf_meter_list.append(optdict["bfr_eval_perf_meter"])
        eval_perf_meter_list.append(optdict["basic_eval_perf_meter"])
        eval_perf_meter_list.append(optdict["hpa_eval_perf_meter"])
        eval_perf_meter_list.append(optdict["chr_eval_perf_meter"])

    env_test = hu.instantiate_env_wrapper_wa(cfg.tuner.env, to_train=False, dataframe=df_test, with_scaler=scaler, perf_meter_args={"name": f'{agent.policy.shortname}T{trial}S{sid}', "stage": "test"})
    assert agent.policy.context() == env_test.unwrapped.ds.contextElems(), 'Discrepancy between policy context and env context...'
    log.info(f"TestAgent: {agent.policy.name}")
    agent.predict(env_test, episodes_max=cfg.tuner.TEST_EPISODES_COUNT, deterministic=cfg.DETERMINISTIC, verbose=cfg.verbosity, baseline_perf_meter=env_oracle.perf_meter, label=f'Best one:{optdict["sweeper_params"]}')
    log.info(f'TestPerf(ScalarPerf, USLA, RAM, SLAV): {(env_test.perf_meter.getSessionPerformanceKPIs(env_oracle.perf_meter))}')
    filever = f'{cfg.iterations_name}{optdict["tid"]}S{optdict["sid"]}-test'
    htmlgraph = agent.savePredictFig(filepath=cfg.pickles_path, head_title=f"TEST/{agent.policy.name} TotalSteps:{env_test.unwrapped.total_steps}", 
                                     filever=filever, ext='-best.html')
    log.info(f'Saved html Pred: {htmlgraph}')
    test_perf_meter_list.append(env_test.perf_meter)

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
    log.info(f"Test Baseline3 {baseline3.policy.name}...")
    baseline3.predict(env_baseline3, episodes_max=cfg.baseline3.TEST_EPISODES_COUNT, verbose=cfg.verbosity, label="Test")
    test_perf_meter_list.append(env_baseline3.perf_meter)

    log.info("Graph tests...")
    oracle_perf = env_oracle.perf_meter.getSessionPerformanceMultiObj()
    oracle_perf = tuple(map(lambda x: round(x, 2), oracle_perf))
    graph_tests = env_test.graphPerWorkload(f'Cumulated Regret/workload {optdict["sweeper_params"]}', title_x=f"Oracle:{oracle_perf}")
    # Curve for the agent
    perf_name = env_test.perf_meter.name
    perf = env_test.perf_meter.getSessionPerformanceMultiObj()
    perf = tuple(map(lambda x: round(x, 2), perf))
    graph_tests.addCurve(f"{perf_name} CReg:{round(env_test.getRegretPerformance(),1)} Perf(USLA,CRAM):{perf}", y=env_test.resultPerWorkload(), perf=env_test.getRegretPerformance())
    # Curve for the baseline
    perf_name = env_baseline.perf_meter.name
    perf = env_baseline.perf_meter.getSessionPerformanceMultiObj()
    perf = tuple(map(lambda x: round(x, 2), perf))
    graph_tests.addCurve(f"{perf_name} CReg:{round(env_baseline.getRegretPerformance(),1)} Perf(USLA,CRAM):{perf}", y=env_baseline.resultPerWorkload(), perf=env_baseline.getRegretPerformance())
    # Curve for the baseline2
    perf_name = env_baseline2.perf_meter.name
    perf = env_baseline2.perf_meter.getSessionPerformanceMultiObj()
    perf = tuple(map(lambda x: round(x, 2), perf))
    graph_tests.addCurve(f"{perf_name} CReg:{round(env_baseline2.getRegretPerformance(),1)} Perf(USLA,CRAM):{perf}", y=env_baseline2.resultPerWorkload(), perf=env_baseline2.getRegretPerformance())
    # Curve for the baseline3
    perf_name = env_baseline3.perf_meter.name
    perf = env_baseline3.perf_meter.getSessionPerformanceMultiObj()
    perf = tuple(map(lambda x: round(x, 2), perf))
    graph_tests.addCurve(f"{perf_name} CReg:{round(env_baseline3.getRegretPerformance(),1)} Perf(USLA,CRAM):{perf}", y=env_baseline3.resultPerWorkload(), perf=env_baseline3.getRegretPerformance())

    fig = graph_tests.figure()
    htmlgraph = os.path.join(cfg.pickles_path, f'{filever}-creg_perWL-{agent.policy.name}-best.html')
    fig.write_html(htmlgraph)
    log.info(f'Saved html Cumulated Regret per WL: {htmlgraph}')

    oracle_eval_pm  = optdict.get("oracle_eval_perf_meter")
    if oracle_eval_pm:
        log.info("Graph all EVAL perf meters...")
        log.info(f'Len(EvalPerfMeterList): {len(eval_perf_meter_list)}')
        graph_perf_eval = GraphPerfMeterComparaison(oracle_perf_meter=oracle_eval_pm, perf_meter_list=eval_perf_meter_list, title=cfg.run_info, stage="EVAL")
        fig = graph_perf_eval.figure()
        htmlgraph = os.path.join(cfg.pickles_path, f'{filever}-eval-perfmeters-comp-{agent.policy.name}-best.html')
        fig.write_html(htmlgraph)
        log.info(f'Saved html Perf Meters in Evals: {htmlgraph}')
    else:
        log.info("WARNING: NO EVAL Graph with all perf meters.")

    log.info("Graph one TEST perf meter against Baselines...")
    log.info(f'Leng(TestPerfMeterList): {len(test_perf_meter_list)}')
    graph_perf_eval = GraphPerfMeterComparaison(oracle_perf_meter=env_oracle.perf_meter, perf_meter_list=test_perf_meter_list, title=cfg.run_info, stage="TEST")
    fig = graph_perf_eval.figure()
    htmlgraph = os.path.join(cfg.pickles_path, f'{filever}-perfmeters_comp-{agent.policy.name}-best.html')
    fig.write_html(htmlgraph)
    log.info(f'Saved html Perf Meters in Test: {htmlgraph}')

    log.info("Graph perf...")
    graph_perf = agent.graph_perf
    if graph_perf:
        fig = graph_perf.figure()
        htmlgraph = os.path.join(cfg.pickles_path, f'{filever}-perfmetrics_perWL-{agent.policy.name}-best.html')
        fig.write_html(htmlgraph)
        log.info(f'Saved Performance Metrics Graph: {htmlgraph}')

    return 0


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
