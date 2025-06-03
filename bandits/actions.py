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
import math
import numpy as np

class Actions():
    def __init__(self, minmax=(-1,1)):
        assert len(minmax) == 2, f'Invalid tuple length. 2 is mandatory, passed {minmax}'
        self._minmax = (math.floor(minmax[0]), math.ceil(minmax[1]))
        assert self._minmax[0] <= 0, f'No STAY action possible with {minmax}'
        assert (self._minmax[1]-self._minmax[0]) > 0, f'No action possible with {minmax}'
        self._action_stay_idx = abs(self._minmax[0])
        self._action_count = self._minmax[1]-self._minmax[0] +1
        self._action_names = [] #np.empty(self._action_count, dtype=str)     # All action's Names
        self._action_vals = np.zeros(self._action_count, dtype=int)          # All action's direction values

        for idx in range(0, self._action_count):
            direction = idx-self._action_stay_idx
            self._action_vals[idx] = direction
            if direction < 0:
                self._action_names.append(f'D{abs(direction)}')
            else:
                self._action_names.append(f'U{abs(direction)}')
        self._action_names[self._action_stay_idx] = "S0"

    def min(self):
        return self._minmax[0]
    def max(self):
        return self._minmax[1]

    def minMax(self):
        return self._minmax
    
    def count(self):
        return self._action_count

    def vals(self):
        return self._action_vals
    
    def name(self, action):
        return self._action_names[int(action)+self._action_stay_idx]

    def arm(self, action):
        return action - self._minmax[0]

    def armStay(self):
        return self._action_stay_idx

    def arm2action(self, arm: int=0):
        return arm - self._action_stay_idx




from gymnasium import Env
from stable_baselines3.common.noise import ActionNoise, NormalActionNoise, OrnsteinUhlenbeckActionNoise

class NormalActionNoise(NormalActionNoise):
    def __init__(self, n_actions=1, noise_sigma: float = 0):
        #print("# NOISE ACTIONS:", n_actions)
        super(NormalActionNoise, self).__init__(mean=np.zeros(n_actions), sigma=noise_sigma * np.ones(n_actions))

