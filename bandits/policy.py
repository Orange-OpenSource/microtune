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
from bandits.actions import Actions

import gymnasium as gym

# Virtual base class that can NOT be instanciated. See LinUCB_xxx and SB3 implementations....
class CtxPolicy(abc.ABC):
    def __init__(self, actions: Actions | tuple | list, discrete_arms_mode=True, ctx=[], use_tips=True, seed=None):
        if type(actions) is not Actions:
            assert len(actions)==2, f'Invalid actions must be of length=2, Type:{type(actions)}, Val:{actions}'
            actions = Actions((actions[0], actions[1]))
        k_arms = actions.count()
        stay_arm = actions.armStay()
        assert k_arms > 0, f"{type(self)}, invalid arms count: {k_arms}, must be > 0"
        assert stay_arm >= 0, f"{type(self)}, invalid Stay arm, {stay_arm}, must be >=0"
        self._k_arms = k_arms
        self._stay_arm = stay_arm
        self._discrete_arms = discrete_arms_mode
        self._ctx_desc = ctx
        self._ctx_len = len(ctx)
        #self._nfeatures = 2 if ctx_len==1 else ctx_len     # Raphaël's tip! Necessary especially when ctx_len is 1, to have a 2 coef linearity (as y = ax +b)
        self._tips = use_tips # Raphaël's tip! Necessary especially when ctx_len is 1, to have a 2 coef linearity (as y = ax +b)
        self._nfeatures = self._ctx_len+1 if self._tips else self._ctx_len
        self.renameFromType()
        np.random.default_rng(seed=seed)
        self.env = None

    def initWithEnv(self, env: gym.Wrapper):
        if env:
            if type(env.action_space) is gym.spaces.Discrete:
                assert self._discrete_arms == True, "Mismatch between Environnement's discrete action space and SB3 Policy working in continous action mode!"
                assert int(env.action_space.start) == 0, f"Env discrete action space must start at 0 and not {env.action_space.start}"
                assert env.action_space.n == self._k_arms, f"Mismatch between Environnement's discrete action space size n={env.action_space.n} and current Arms count is k_arms={self._k_arms}"
                #self._rescale_func = lambda arm: arm + arm_min # Rescale (normalize) a discrete arm index (int, in 0 to K_ARMS-1 interval) into discrete interval [X,Y] (with X > Y)
                print(f"Will Rescale arms to {env.action_space} [{env.action_space.start}, {env.action_space.n-1}]")
            elif type(env.action_space) is gym.spaces.Box:
                assert self._discrete_arms == False, "Mismatch between Environnement's continous action space (Box) and SB3 Policy working in discrete action mode!"
                assert env.action_space.low[0] == -1 and env.action_space.high[0] == 1, f"Env continous action space must be -1.,1. (because of the required symmetry, see tips SB3). Actual min={env.action_space.low[0]}, max={env.action_space.high[0]}"
                #arm_max = self._k_arms -1
                #self._rescale_func = lambda arm: np.array([(2 * arm / arm_max) - 1], dtype=float) # Rescale (normalize) a discrete arm index (int, in 0 to K_ARMS-1 interval) into continous -1.,1. interval (float)
                print(f"Will Rescale arms to {env.action_space} [{env.action_space.low[0]}, {env.action_space.high[0]}]")
            else:
                assert False, f'Env action_space type not supported {env.action_space}. Should be Box [-1.,1.] or Discrete()'
            self.env = env

    def context(self):
        return self._ctx_desc
    
    def renameFromType(self, obj=None, origin=''):
        if obj is None:
            obj = self
        polname = type(obj).__name__.replace('Policy', '')
        self.shortname = f"{origin}{polname}"
        amode = 'D' if self._discrete_arms else 'C'
        minmax = f'{-self._stay_arm}_{self._k_arms-self._stay_arm-1}'
        self.name = f"{self.shortname}_{minmax}{amode}"

    # Rescale (normalize) a discrete arm index (int, in 0 to K_ARMS-1 interval) into continous -1.,1. interval (float)
    # Return an np.array with action
    def _arm_discrete2continous(self, arm):
        arm_max = self._k_arms -1
        return np.array([(2 * arm / arm_max) - 1], dtype=float) 

    def _arm_rescale(self, arm):
        return int(arm) if self._discrete_arms else self._arm_discrete2continous(arm)


    def _getPolicyFeatures(self, x_array):
        #if self._ctx_len == 1:
        #    return np.append(x_array, [1])
        #else:
        #    return x_array
        #assert len(x_array) == self._ctx_len, f"ERROR, CTX LEN {len(x_array)}"
        return np.append(x_array, [1]) if self._tips else x_array

    # Convert the context vector into a dictionary
    def getCtxDict(self, x_array):
        return { k: x_array[idx] for idx, k in enumerate(self._ctx_desc) }
        
    def getArmsCount(self):
        return self._k_arms   

    # boost has to be >= 0 typiccaly
    # Returned arm starts from 0 to Stay-1
    def getDownArmIndex(self, boost:int=0):
        incr = 1+max(0,boost)
        return self._arm_rescale(max(0, self._stay_arm-incr))
            
    def getStayArmIndex(self):
        return self._arm_rescale(self._stay_arm)
            
    # boost has to be >= 0 typiccaly
    # Returned arm starts from Stay+1 to Max
    def getUpArmIndex(self, boost=0):
        incr = 1+max(0,boost)
        return self._arm_rescale(min(self._k_arms-1, self._stay_arm+incr))

    def getCtxLen(self):
        return self._ctx_len   

    # Return policy's data to save. Do not include Gym env in data. 
    def dataToSave(self):
        saved_env = self.env
        data = [ dict(vars(self)) ]
        self.env = saved_env
        return data 
    
    def restoreData(self, data=[]):
        saved_vars = data.pop(0)
        self.__dict__ |= saved_vars
        return data

    # Like predict_values() in STB3 policies   
    @abc.abstractmethod
    def select_arm(self, x_array, deterministic=False, debug=False):
        pass

    # Like forward() in STB3 policies   
    def update(self, arm_index, reward, x_array, next_x_array=None):
        pass




