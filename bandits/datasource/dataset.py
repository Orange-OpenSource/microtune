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

import abc
import numpy as np


from numpy.random import Generator, PCG64, SeedSequence
from scipy.stats import truncnorm
from sklearn.preprocessing import MinMaxScaler 


class DataSetEntryContextSelector(abc.ABC):
    def __init__(self, group_list=["Default"], entries_per_group=1, seed=1, context_elems=None, normalize=False, with_scaler:MinMaxScaler=None):
        self.seed = seed

        # Determine Groups size and Entries count per group
        self._group_list = group_list
        self._groups_count = len(self._group_list)
        self._entries_count = entries_per_group
        self._total_entries_count = self._groups_count*self._entries_count

        self._reinit(context_elems=context_elems, scaler=with_scaler)
        if context_elems:
            self.prepareContext(context_elems=context_elems, normalize=normalize, with_scaler=with_scaler)

    # Reinit current group and entry indexes. Remove context vectors
    def _reinit(self, context_elems=None, scaler=None):
        # Entry concern reinit
        self.group_idx = 0
        self.entry_idx = 0

        sg = SeedSequence(self.seed)
        rg = [Generator(PCG64(s)) for s in sg.spawn(self._groups_count+1)]
        self.rg_group = rg[0]
        self.rg_entry = np.array(rg[1:])

        # Entry's context reinit
        self._context_elems = context_elems
        self._context_vectors = None
        self.min_max_scaler = scaler


    def _selectGroupFromUnboundIndex(self, gidx:int):
        self.group_idx = int(abs(gidx)%self._groups_count)
        self.entry_idx = 0

    def _selectGroupRandomly(self):
        # Return random integers from low (inclusive) to high (exclusive)
        self.group_idx = self.rg_group.integers(0, self._groups_count)
        self.entry_idx = 0

    def _selectEntryMinMax(self, set_to_min=True):
        self.entry_idx = 0 if set_to_min else self._entries_count-1

    def _selectEntryFromRatio(self, ratio=0.5):
        self.entry_idx = round(ratio*(self._entries_count -1))

    # Gaussian (normal) distribution that is constrainted by the available entries in the group
    # sigma: If 0, only mu is fired. If <0 an automatic computation of sigma is done based on mu value and the total entries count in a group. 
    # The automatic mode has for purpose to cover all the entries with a Gaussian distribution.
    def _selectEntryGRandomly(self, mu, sigma_tipping_ratio):
        mu = int(mu)
        
        if sigma_tipping_ratio == 0:
            self.entry_idx = min(max(0, mu), self._entries_count -1)
            return
        
       # Automatic computation of sigma ?
        if sigma_tipping_ratio < 0:
            sigma = abs(sigma_tipping_ratio)*((max(mu, self._entries_count-mu))//2 -1)
        else:
            sigma = int((self._entries_count//2)*sigma_tipping_ratio)

        # Define the bounds for the truncation relative to the mean and standard deviation
        a, b = -mu/sigma, (self._entries_count -1 - mu)/sigma
        # Generate truncated Gaussian distribution. Each workload has its own generator
        idx = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=1, random_state=self.rg_entry[self.group_idx])
        self.entry_idx = int(np.rint(idx))

    # Uniform distribution
    def _selectEntryURandomly(self):
        # Return random integers from low (inclusive) to high (exclusive)
        self.entry_idx = self.rg_entry.integers(0, self._entries_count)

    # Select an eentry in current group
    # Return the real increment applied, that can be different than the one in argument if the move goes out of range (in group)
    # Change entry but keep it in its range (0 to entries count per group)
    # incr can be <0 or >=0, 
    # Return the real increment applied
    def _select(self, incr=0) -> int:
        newidx = self.entry_idx+incr

        real_incr = incr
        if newidx < 0:
           real_incr = -self.entry_idx                    # idx=2: incr=-5 -idx=2 => -2
        elif newidx >= self._entries_count:
           real_incr = incr -(newidx - self._entries_count +1)   # idx=6: incr=5 - (newidx=11 -ec=10 +1) => 3, idx=9: incr=1 - (newidx=10 -ec=10 +1) => 3

        self.entry_idx += real_incr
        
        return real_incr

    def _applyLambda2entries(self, incrs=[],  func=lambda idx, incr:None):
        for idx in range(len(incrs)):
            real_incr = 0
            incr = incrs[idx] 
            if incr != 0:
                real_incr = self._select(incr)
            
            func(idx, incr) # Do not use real_incr. Thus the incr state is presumed at the same state than re real_incr. This happens at both ends of the entries group
            
            if real_incr != 0:
                self._select(-real_incr)          

    # Index number of the current row in the Dataframe
    def globalIndex(self):
        assert self.group_idx < self._groups_count and self.entry_idx < self._entries_count, f"Error, GlobalIDX out of bound: GID:{self.group_idx} EID:{self.entry_idx}"
        return (self.group_idx*self._entries_count)+self.entry_idx

    def getTotalStatesCount(self):
        return self._total_entries_count

    ##
    ## Context functions
    ##

    def _normalizeNumpyFloatElems(self, vectors):
        if self.min_max_scaler is None:
            min_max_scaler = MinMaxScaler()
            self.min_max_scaler = min_max_scaler.fit(vectors)

        return self.min_max_scaler.transform(vectors)

    def getContextVectors(self):
        return self._context_vectors
    
    # Return current observation from current state
    # Requirement: observation space must be prepared first
    def context(self):
        return self._context_vectors[self.globalIndex()]

    # Return context elements names
    def contextElems(self):
        return self._context_elems

    @abc.abstractmethod
    def prepareContext(self, context_elems=[], check_data=False, normalize=False, with_scaler:MinMaxScaler=None):
        pass

# DO NOT USE IT AS IS: Must be sub-classed to be completed with either a Dataframe or a DB conection (Live)
class ADBMSDataSetEntryContextSelector(DataSetEntryContextSelector):
    def __init__(self, group_list=["Default"], entries_per_group=1, seed=1, context_elems=None, normalize=False, with_scaler:MinMaxScaler=None):
        super().__init__(group_list=group_list, entries_per_group=entries_per_group, seed=seed, context_elems=context_elems, normalize=normalize, with_scaler=with_scaler)

    # Return current state (if bufferidx_increment is 0) or the state at an another buffer index value.
    # Return None if the buffer idx resulting to the current buffer+the increment is out of range.
    def state(self, bufferidx_increment=0):
       pass

    # Apply a function to each buffer's action
    def applyLambda2actions(self, actions: np.array=None,  buffunc=lambda action_idx, real_bufferidx_increment:None):
        incrs = np.array(list(map(lambda x:-x, actions))) # Convert buffer's actions to dataset's increments
        def func(idx, real_incr):
            buffunc(idx, -real_incr)
        
        self._applyLambda2entries(incrs, func)

    # Reset position onto a new workload and a new buffer index in the workload's entries
    def reset(self, workload_idx: int = 0):
        self._selectGroupFromUnboundIndex(workload_idx)

    def next(self):
        self._selectEntryURandomly()

    # Move in current group
    # Here incr is used as an increment on current index (in dataframe)
    # action: -N for DOWN, 0 for STAY, +N for UP
    # Return the real increment applied, that can be different than the one in argument if the move goes out of range (in group)
    def move(self, bufferidx_increment=0):
        return -self._select(incr=-bufferidx_increment)

    @abc.abstractmethod
    def getWorkloadNameId(self):
        assert False, "getWorkloadNameId() must be implemented!"

    def getWorkloadsCount(self):
        return self._groups_count
    
    def getWorkloadsList(self):
        return self._groups_list
    
    def getStatesCountPerWorkload(self):
        return self._entries_count

    ##
    ## Some Metrics
    ##

    @abc.abstractmethod
    def getQPSLatWeigths(self):
        assert False, "getQPSLatWeigths() must be implemented!"
    
    # Return IPERF, IDELTAPERF, IPERFTARGET
    @abc.abstractmethod
    def getIPerfIndicators(self):
        assert False, "getIPerfIndicators() must be implemented!"

    @abc.abstractmethod
    def getLatency(self, filtered=True):
        assert False, "getLatency() must be implemented!"
    
    # Return Latency gap, Latency gap in ms (str), latency target (threshold)
    @abc.abstractmethod
    def getLatencyGapTarget(self):
        assert False, "getLatencyGapTarget() must be implemented!"


