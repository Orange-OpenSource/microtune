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
import numpy as np
import logging
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
#from hydra.core.hydra_config import HydraConfig
#import joblib

#import pkg.datasource.adbms_dataframe as ds
import hydrauti as hu
#from pkg.datasource.dataframes.obs_samples_dataframes import ObsSamplesDF 


# A logger for this file
log = logging.getLogger(__name__)
#lock = Lock()


#from typing import Tuple

@hydra.main(version_base=None, config_path="configs", config_name="show_models")
def run(cfg: DictConfig) -> float: #Tuple[float, float]:
    if not hu.prepare_run(cfg):
        return (np.inf, np.inf)

    trial = cfg.trial
    sid = cfg.seed
    if trial is None or trial < 0:
        filever=f'{cfg.iterations_name}*' #{trial}S{sid}'
        dftext='-best.pickle'
    else:
        filever=f'{cfg.iterations_name}{trial}S{sid}'
        dftext='.pickle'

    # Agent selected by default by the configuration or experiment
    #            tid=trial, sid=sid, seed=RND_SEED, min_max_scaler=minmax_scaler,
    #            train_regrets=train_regret_cumsum, train_reg_perf=train_reg_perf, train_reg_results=train_reg_results, 
    #            eval_regrets=eval_regret_cumsum, eval_reg_perf=env_eval.getRegretPerformance(), 
    #            eval_iregrets=eval_iregret_cumsum, eval_ireg_perf=env_eval.getIRegretPerformance(), 
    #            sweeper_params=sweeper_params, sweep_perf=sweep_perf, eval_perf_meter=env_eval.perf_meter, oracle_eval_perf_meter=oracle_perf_meter,
    #            eval_perf_meter_list=[],
    #            config=OmegaConf.to_yaml(cfg, resolve=True))    
    #agent = instantiate(cfg.tuner.agent)
    #files_list, optdict = agent.load(filepath=cfg.pickles_path, filever=filever, verbose=1, dftext=dftext) # Load the picckle specified by its Trial and Seed ID

    agent = instantiate(cfg.a2c.agent)
    files_list, optdict = agent.load(filepath=cfg.pickles_path+"_disq", filever=filever, verbose=1, dftext=dftext) # Load the picckle specified by its Trial and Seed ID

    #print(f"Agent options: {optdict}")
    print(f"Agent learning params: {optdict['sweeper_params']}")
    
    # PerfMeter object is in pkg/perf_meter.py
    oracle_perf_meter = optdict['oracle_eval_perf_meter']
    print(f"Agent Oracle performance: 0, (USLA,CRAM,CRAM_MB/step)={oracle_perf_meter.getSessionPerformanceMultiObj(pretty=True)}")
    
    pm = optdict['eval_perf_meter']
    print(f"Agent Scalar performance vs Oracle: {pm.getSessionScalarPerformance(oracle_perf_meter)}, (USLA,CRAM,CRAM_MB/step)={pm.getSessionPerformanceMultiObj(pretty=True)}")
    print(f"Agent loaded with embedded files: {files_list}")
 
    return 0
 
if __name__ == "__main__":
    run()
