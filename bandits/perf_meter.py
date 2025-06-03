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

#import math
import numpy as np
import re

class PerfMeter():
    def __init__(self, name="", baseline=0, stage="", trial=-1):
        self.name = name
        self.stage = stage
        self.label = ""
        self.trial = trial # Unknown until there
        self.isOracle = False # By default
        self.baseline= baseline
        self.isBaseline = False if baseline == 0 else True # By default

        if "oracle" in name.lower():
            self.isOracle =True
            self.baseline=1
        
        self.startSession()

    # On prediction session start (first episode of a serie of episodes)
    def startSession(self):
        self.episode_ids = []
        self._ep_count = 0
        self.under_sla_tot = 0
        self.ram_quantities_tot = 0
        self.nsteps = 0
        self.nsteps_pos_cur_ep = 0
        self.nsteps_neg_cur_ep = 0
        self.cumul_idelta_pos_per_ep = []
        self.cumul_idelta_neg_per_ep = []
        self.ram_quantities_pos_per_ep = []
        self.ram_quantities_neg_per_ep = []
        self.violations_count_per_ep = []

    def _endEpisode(self):
        cur_ep_count = len(self.episode_ids)
        if self._ep_count == cur_ep_count:
            return
        self._ep_count = cur_ep_count

        # Finalize RAM outflow computation for last episode if it has more than 1 step
        if self.nsteps_pos_cur_ep > 1:
            self.ram_quantities_pos_per_ep[-1] /= self.nsteps_pos_cur_ep
        if self.nsteps_neg_cur_ep > 1:
            self.ram_quantities_neg_per_ep[-1] /= self.nsteps_neg_cur_ep

    # End session and ge indicators
    # Return a tuple of 5 numpy arrays of a length = Episodes count:
    # (IDELTA+, IDELTA-, RAM outflow+, RAM outflow-, SLA VIOLATIONS)
    # IDELTA+, IDELTA-, SLA VIOLATIONS are counted when regret is >0 only
    # RAM outflows + and - are counted whatever the regret value
    def getIndicators(self):
        self._endEpisode()
        assert len(self.cumul_idelta_pos_per_ep) == self._ep_count, f'Res length: {len(self.cumul_idelta_pos_per_ep)} != {self._ep_count}'
        assert len(self.cumul_idelta_neg_per_ep) == self._ep_count, f'Res length: {len(self.cumul_idelta_pos_per_ep)} != {self._ep_count}'
        assert len(self.ram_quantities_pos_per_ep) == self._ep_count, f'Res length: {len(self.ram_quantities_pos_per_ep)} != {self._ep_count}'
        assert len(self.ram_quantities_neg_per_ep) == self._ep_count, f'Res length: {len(self.ram_quantities_neg_per_ep)} != {self._ep_count}'
        assert len(self.violations_count_per_ep) == self._ep_count, f'Res length: {len(self.violations_count_per_ep)} != {self._ep_count}'

        return (np.array(self.cumul_idelta_pos_per_ep), 
                np.array(self.cumul_idelta_neg_per_ep), 
                np.array(self.ram_quantities_pos_per_ep), 
                np.array(self.ram_quantities_neg_per_ep),
                np.array(self.violations_count_per_ep))
            
    def onNextEpisode(self, episode_id="noname"):
        self._endEpisode()
        # Start new episode
        self.episode_ids.append(episode_id)
        self.nsteps_pos_cur_ep = 0
        self.nsteps_neg_cur_ep = 0
        self.cumul_idelta_pos_per_ep.append(0)
        self.cumul_idelta_neg_per_ep.append(0)
        self.ram_quantities_pos_per_ep.append(0)
        self.ram_quantities_neg_per_ep.append(0)
        self.violations_count_per_ep.append(0)

    # New step for the last Episode (append only!!!)        
    def onNextStep(self, regret, idelta, buf_size):
        self.nsteps += 1
        ep_idx = len(self.episode_ids) -1
        if regret > 0:
            if idelta>=0:
                self.cumul_idelta_pos_per_ep[ep_idx] += idelta
            else:
                self.cumul_idelta_neg_per_ep[ep_idx] -= idelta
                self.violations_count_per_ep[ep_idx] += 1

        if idelta>=0:
            self.nsteps_pos_cur_ep += 1
            self.ram_quantities_pos_per_ep[ep_idx] += buf_size
        else:
            self.nsteps_neg_cur_ep += 1
            self.ram_quantities_neg_per_ep[ep_idx] += buf_size
            self.under_sla_tot  += 1
        
        self.ram_quantities_tot += buf_size


    def getSessionEpisodesCount(self):
        return len(self.episode_ids)
    
    def getSessionStepsCount(self):
        return self.nsteps
    
    def getSessionEpisodesIds(self):
        return np.array(self.episode_ids)

    # pretty: RAM quantity in MB per step on all sessions
    def getSessionPerformanceMultiObj(self, pretty=False):
        if pretty:
            return self.under_sla_tot, round((self.ram_quantities_tot/self.nsteps)/1024, 5)
        else:
            return self.under_sla_tot, self.ram_quantities_tot
    
    # Scalar performance value obtained with UnderSLA and RAM quantities
    def getSessionScalarPerformance(self, oracle_perf_meter = None):
        assert oracle_perf_meter is not None, f"Error, The Oracle performance is required. Provide an Eval/Test env monitor with the Oracle performance"
        assert self.nsteps == oracle_perf_meter.nsteps, f"Error Oracle and agent tested on different datasets"

        return abs(self.under_sla_tot - oracle_perf_meter.under_sla_tot)/oracle_perf_meter.under_sla_tot + abs(self.ram_quantities_tot - oracle_perf_meter.ram_quantities_tot)/oracle_perf_meter.ram_quantities_tot

    # Return Scalar_perf, UnderSLA, RAM, SLAViolations
    def getSessionPerformanceKPIs(self, oracle_perf_meter = None):
        scalar_perf = self.getSessionScalarPerformance(oracle_perf_meter)
        return scalar_perf, self.under_sla_tot, self.ram_quantities_tot, sum(self.violations_count_per_ep)

    def getSessionPerformanceMultiObjOLD(self, oracle_perf_meter = None):
        assert oracle_perf_meter is not None, f"Error, The Oracle performance is required. Provide an Eval/Test env monitor with the Oracle performance"
        assert self.nsteps == oracle_perf_meter.nsteps, f"Error Oracle and agent tested on different datasets"
        
        # Compare here the RAW performance (Violations and RAM quantities) againts the Oracle and compute a level of performance...
        #pm.under_sla_tot = 0
        #pm.ram_quantities_tot = 0
        ram_quantities_tot = self.ram_quantities_tot - oracle_perf_meter.ram_quantities_tot 

        return self.under_sla_tot, ram_quantities_tot