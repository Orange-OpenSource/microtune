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
import numpy as np

import logging
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

import hydrauti as hu
from bandits.graph import GraphPredictionsStatus
from bandits.gym_env import VSEnv, VSMonitorCallback

# Command: python run_best_agent_test.py +experiment=linucb_kfoofw ++hydra.sweeper.n_trials=1 verbosity=1

# A logger for this file
log = logging.getLogger(__name__)

class MonitorSLA(VSMonitorCallback):
    def __init__(self):
        super().__init__()
        self.states = []
        self.latencies = []
        self.buffers_mb = []
        self.step_in_wl = []
    
    def onStart(self, env: VSEnv):
        pass
    def onReset(self, env: VSEnv):
        pass

    def onStep(self, env: VSEnv, info: dict={}):
        self.states.append(env.sla)
        self.latencies.append(env.latency)
        self.buffers_mb.append(env.getBufferSizeMB())
        self.step_in_wl.append(env.cur_step)

@hydra.main(version_base=None, config_path="configs", config_name="simple_agent_test")
def run(cfg: DictConfig) -> None:
    #logging.basicConfig(filename=cfg.logfile)
    # Performs only 1 trial
    if not hu.prepare_run(cfg):
        return np.inf

    # TEST  ONLY
    datasets = instantiate(cfg.datasets)
    if cfg.eval_data:
        datasets.load(test_only=False)
        df_test = datasets.df_eval
        stage="EVAL"
    else:
        datasets.load(test_only=True)
        df_test = datasets.df_test
        stage="TEST"
    trial = cfg.trial
    sid = cfg.seed
    filever=f'{cfg.iterations_name}{trial}S{sid}'

    agent = instantiate(cfg.tuner.agent)
    #            tid=trial, sid=sid, seed=RND_SEED, min_max_scaler=minmax_scaler,
    #            train_regrets=train_regret_cumsum, train_reg_perf=train_reg_perf, train_reg_results=train_reg_results, 
    #            eval_regrets=eval_regret_cumsum, eval_reg_perf=env_eval.getRegretPerformance(), 
    #            eval_iregrets=eval_iregret_cumsum, eval_ireg_perf=env_eval.getIRegretPerformance(), 
    #            sweeper_params=sweeper_params, sweep_perf=sweep_perf, eval_perf_meter=env_eval.perf_meter, oracle_eval_perf_meter=oracle_perf_meter,
    #            eval_perf_meter_list=[],
    #            config=OmegaConf.to_yaml(cfg, resolve=True))    
    files_list, optdict = agent.load(filepath=cfg.pickles_path, filever=filever, verbose=0, dftext='-best.pickle')

#    for ff in files_list:
#        os.remove(ff)
    log.info(f'Agent Filename: {agent.filename}')
    log.info(f'Agent learning params: {optdict["sweeper_params"]}')
    assert cfg.trial == optdict["tid"]
    assert cfg.seed == optdict["sid"]
    scaler = optdict["min_max_scaler"]
    if scaler:
        print(f'MinMaxScaler:{type(scaler)}')

    # Display Eval perf if available ?
    pm = optdict.get("eval_perf_meter")
    if pm:
        log.info(f'Sweep eval perf: {optdict["sweep_perf"]} Model eval perf(MO): {pm.getSessionPerformanceMultiObj()} #Other Eval perfs to display:{len(optdict["eval_perf_meter_list"])}')

    log.info("Graph Best Train Perf against others...")
    env_test = hu.instantiate_env_wrapper_wa(cfg.tuner.env, to_train=False, dataframe=df_test, with_scaler=scaler, perf_meter_args={"name": f'{agent.policy.shortname}T{trial}S{sid}', "stage": stage.lower()})
    assert agent.policy.context() == env_test.unwrapped.ds.contextElems(), 'Discrepancy between policy context and env context...'
    log.info(f"Agent {agent.policy.name}...")

    m_sla = MonitorSLA()
    env_test.addCallback(m_sla)
    agent.predict(env_test, episodes_max=cfg.tuner.TEST_EPISODES_COUNT, deterministic=cfg.DETERMINISTIC, verbose=cfg.verbosity)
    pf = env_test.perf_meter

    graph = GraphPredictionsStatus(title=f"SLA Graph Stage:{stage} Perf:{pf.getSessionPerformanceMultiObj()} {env_test.unwrapped.reward.name}")
    fig = graph.figure(states=m_sla.states, latencies=m_sla.latencies, buffers=m_sla.buffers_mb, step_in_wl=m_sla.step_in_wl)
    htmlgraph = os.path.join(cfg.pickles_path, f'{filever}-{stage.lower()}-sla_perf-{agent.policy.name}-best.html')
    fig.write_html(htmlgraph)
    log.info(f'Saved SLA Graph: {htmlgraph}')

    return 0


if __name__ == "__main__":
    run()


