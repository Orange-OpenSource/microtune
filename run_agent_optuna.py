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

import hydra
from omegaconf import DictConfig

from run_agent import run

from typing import Tuple

@hydra.main(version_base=None, config_path="configs", config_name="agent_optuna")
def run_with_optuna(cfg: DictConfig) -> Tuple[float, float]:
    return run(cfg)


if __name__ == "__main__":
    run_with_optuna()

