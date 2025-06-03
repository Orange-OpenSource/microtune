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
import sys
sys.path.append('.')

from bandits.datasource.dataframes.obs_samples_dataframes import ObsSamplesDF 

picklefiles="./workloads"
obsdf = ObsSamplesDF(version="9")
df = obsdf.mergeVersionsToNew(picklefiles, "10", "11")
df_train, df_test = obsdf.spliDFByClients(df) 

obsdf.saveTrainTests(picklefiles, df_train, df_test) 
