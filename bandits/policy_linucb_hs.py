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
import numpy as np

from bandits.policy import CtxPolicy
from bandits.actions import Actions

#Pat ver

# HSLinUCB from  https://github.com/Orange-OpenSource/HSLinUCB/blob/main/roles/HSLinUCB/files/simulation/hslinucb_coldstart.ipynb
class HSLinUCBPolicy(CtxPolicy):
    def __init__(self, actions: Actions | tuple = (-1, 1), ctx=[], learning_rate=0.8, inversion_interval = 5, _lambda = 1, tie_break_mode = "random", seed=None):
        super().__init__(actions, True, ctx=ctx, use_tips=True, seed=seed)
        self._nArms = self._k_arms
        self._alpha = learning_rate
        self._lambda = _lambda
        self._inversion_interval = inversion_interval
        self._tie_break_mode = tie_break_mode
        self.reset()

    def reset(self):
        self.t = 0
        self._A = [self._lambda * np.eye(self._nfeatures) for k in range(self._nArms)]
        self._b = [np.zeros((self._nfeatures, 1)) for k in range(self._nArms)]
        self._invert(display=True)        

    
    def _getPolicyFeatures(self, x_array):
        return  np.asmatrix(super()._getPolicyFeatures(x_array))
    
    def _invert(self, display = False):
        #print([a for a in self._A])
        #self._A_inv = [np.linalg.pinv(a) for a in self._A]   # /!\ WE We had some "Singular Matrix" error. 
        self._A_inv = [np.linalg.inv(a) for a in self._A]   # /!\ Do not use linalg.pinv because it degrades the performance!! 
        self._theta = [np.dot(self._A_inv[k],self._b[k]) for k in range(self._nArms)]

    # Overrides virtual method of ABC CtxPolicy class
    def select_arm(self, x_array, deterministic=False, debug=False):
        features = self._getPolicyFeatures(x_array)
        value = np.zeros(self._nArms)
        confidence = np.zeros(self._nArms)
        for index,i_theta in enumerate(self._theta):
            value[index] = np.dot(features,i_theta)[0]
            
        for k in range(self._nArms):
            confidence[k] = self._alpha * np.sqrt(np.dot(features, np.dot(self._A_inv[k],np.transpose(features))))
        decision_space = [i for i,v in enumerate(np.squeeze(np.asarray(value + confidence)).ravel()) if v == np.max(np.squeeze(np.asarray(value + confidence)))]
        if (self._tie_break_mode == "min"):
            #if more than one possible action choose the first one
            best_action = np.min(decision_space)
        elif (self._tie_break_mode == "max"):
            #if more than one possible action choose the last one
            best_action = np.max(decision_space)
        else:
            #if more than one possible action choose randomly
            best_action = np.random.choice(decision_space)

        return int(best_action) #, value, confidence

    # def observe(self, played_arm, context, next_context, reward, update = False):
    def update(self, arm_index, reward, x_array, next_x_array=None):
        features = self._getPolicyFeatures(x_array)
        self._A[arm_index] = self._A[arm_index] + np.dot(np.transpose(features),features)
        self._b[arm_index] = self._b[arm_index] + np.transpose(features * reward)
        self._invert(display=True)
        self.t += 1
