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

#qpslat_weights="01"   # QPS is not used in performance computation (weigth is 0%), the objective is to maintain te Latency (weigth is 100%). Other options is "19" (10% for QPS, 90% for Latency)

#from multiprocessing import Lock
import numpy as np
import logging
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
#from hydra.core.hydra_config import HydraConfig
#import joblib

#import bandits.datasource.adbms_dataframe as ds
import hydrauti as hu
#from bandits.datasource.dataframes.obs_samples_dataframes import ObsSamplesDF 


# A logger for this file
log = logging.getLogger(__name__)
#lock = Lock()


#from typing import Tuple

@hydra.main(version_base=None, config_path="configs", config_name="dataset")
def run(cfg: DictConfig) -> float: #Tuple[float, float]:
    if not hu.prepare_run(cfg):
        return (np.inf, np.inf)

    datasets = instantiate(cfg.datasets)
    datasets.load()

    baseline, optfiles, optdict = hu.instanciate_agent_and_save(cfg, datasets=datasets, tuner="oracle", eval=True)
    perf = optdict["best_perf_mo"]
    log.info(f'{baseline.policy.name} Perf: {perf}')

    return perf


if __name__ == "__main__":
    run()

