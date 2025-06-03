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

from bandits.policy import CtxPolicy
from bandits.actions import Actions

import numpy as np

# TODO: Bug sur index du buffer=0
class HpaPolicy(CtxPolicy):
    def __init__(self, actions: Actions, ctx=[],  ext_threshold=None, seed=None):
        super().__init__(actions, ctx=ctx, use_tips=False, seed=seed)
        self.desiredBufSzIdx = -1
        self.threshold = ext_threshold
           
    # Overrides virtual method off ABC CtxPolicy class
    def select_arm(self, context, deterministic=False, debug=False):
        threshold = context[2] if self.threshold is None else self.threshold
        self.desiredBufSzIdx = math.ceil((context[0]+1) * (context[1] / threshold))-1 # TIPS: Because context[0] is Discrete and can be 0, Add +1 to avoid HPA to stick at bottom value
        boost = int(max(0, abs(context[0] - self.desiredBufSzIdx) -1))
        if debug:
            cd = self.getCtxDict(context)
            print(f"ContextDict:{cd} Desired BufferSize Index:{self.desiredBufSzIdx} Threshold:{threshold} Boost:{boost}")

        #Down arm ?
        if (self.desiredBufSzIdx) < context[0]:
            return self.getDownArmIndex(boost=boost)
            
        #Stay arm ?
        if (self.desiredBufSzIdx) == context[0]:
            return self.getStayArmIndex()
            
        #Up arm ?
        if (self.desiredBufSzIdx) > context[0]:
            return self.getUpArmIndex(boost=boost)


class BasicPolicy(CtxPolicy):
    def __init__(self,actions: Actions, ctx=[], seed=None):
        super().__init__(actions, ctx=ctx, use_tips=False, seed=seed)

    # Overrides virtual method off ABC CtxPolicy class
    def select_arm(self, context, deterministic=False, debug=False):
        if debug:
            cd = self.getCtxDict(context)
            print(f"ContextDict:{cd}")

        latency_perf_range = context[4]-context[3]
        if context[0] <= context[1] - context[2]*latency_perf_range : # SLA OVER CUR_LAT <= TRESHOLD - OBJGAP*(LATMAX-LATMIN) 
            #Down arm
            return self.getDownArmIndex(boost=0)
            
        if context[0] > context[1] : # SLA FAILED
            #Up arm
            return self.getUpArmIndex(boost=0)

        #SLA OK. Stay arm
        return self.getStayArmIndex()   

class OraclePolicy(CtxPolicy):
    def __init__(self,actions: Actions, ctx=[], seed=None):
        super().__init__(actions, ctx=ctx, use_tips=False, seed=seed)

    # Overrides virtual method off ABC CtxPolicy class
    def select_arm(self, context, deterministic=False, debug=False):
        if debug:
            print(self.env.unwrapped.getRewardStates())
        # return np.argmax(self.env.unwrapped.getRewardStates())
        try:
            cur_idelta, down_idelta = self.env.unwrapped.getIDeltaOnAction(action=-1)
        except Exception as ex:
            print("---------------------------")
            print(Exception)
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print("---------------------------")
        if cur_idelta < 0 :
            return self.getUpArmIndex()
        if down_idelta == None or down_idelta < 0:
             return self.getStayArmIndex()
        
        return self.getDownArmIndex()


        
class CacheHitRatioPolicy(CtxPolicy):
    def __init__(self,actions: Actions, ctx=[], threshold=0.96, seed=None):
        super().__init__(actions, ctx=ctx, use_tips=False, seed=seed)
        self.threshold = threshold
        origin = f'TH{round(threshold, 3)}'
        self.renameFromType(self, origin=origin)


    # Overrides virtual method off ABC CtxPolicy class
    def select_arm(self, context, deterministic=False, debug=False):
        if debug:
            cd = self.getCtxDict(context)
            print(f"ContextDict:{cd}")

        if context[0] >= (self.threshold+0.01):
            #Down arm
            return self.getDownArmIndex()            

        if context[0] < self.threshold:
            #Up arm
            return self.getUpArmIndex()            

        #Stay arm
        return self.getStayArmIndex()   


class BufFillingRatePolicy(CtxPolicy):
    def __init__(self,actions: Actions, ctx=[], threshold=0.92, seed=None):
        super().__init__(actions, ctx=ctx, use_tips=False, seed=seed)
        self.threshold = threshold
        origin = f'TH{round(threshold, 3)}'
        self.renameFromType(self, origin=origin)


    # Overrides virtual method off ABC CtxPolicy class
    def select_arm(self, context, deterministic=False, debug=False):
        if debug:
            cd = self.getCtxDict(context)
            print(f"ContextDict:{cd}")

        if context[0] < (self.threshold - self.threshold*0.2):
            #Down arm
            return self.getDownArmIndex()            

        if context[0] >= self.threshold:
            #Up arm
            return self.getUpArmIndex()            

        #Stay arm
        return self.getStayArmIndex()   

# First context field must be the buffer's index from 0 up to ("buf_values_count" -1)
class TopDownPolicy(CtxPolicy):
    def __init__(self, actions: Actions, ctx=[], buf_values_count=64, seed=None):
        super().__init__(actions, ctx=ctx, use_tips=False, seed=seed)
        self.buf_values_count = buf_values_count
        self._topdown = True

    # Overrides virtual method off ABC CtxPolicy class
    def select_arm(self, context, deterministic=False, debug=False):
        if debug:
            cd = self.getCtxDict(context)
            print(f"ContextDict:{cd} BufferSize Count (per workload): {self.buf_values_count}")

        if context[0] > 0:
            if context[0] == self.buf_values_count-1:
                self._topdown = True

            if self._topdown:
                return self.getDownArmIndex() #Down arm
            else:
                return self.getUpArmIndex() # Up arm

        if self._topdown:
            self._topdown = False
            return self.getStayArmIndex()   
        else:
            return self.getUpArmIndex() # Up arm

# First context field must be a value from 0.0 up to 1.0
class TopDownPolicyNormalized(CtxPolicy):
    def __init__(self, actions: Actions, ctx=[], seed=None):
        super().__init__(actions, ctx=ctx, use_tips=False, seed=seed)
        self._topdown = True

    # Overrides virtual method off ABC CtxPolicy class
    def select_arm(self, context, deterministic=False, debug=False):
        if debug:
            cd = self.getCtxDict(context)
            print(f"ContextDict:{cd} BufferSize Count (per workload): {self.buf_values_count}")

        if context[0] > 0:
            if context[0] == 1.:
                self._topdown = True

            if self._topdown:
                return self.getDownArmIndex() #Down arm
            else:
                return self.getUpArmIndex() # Up arm

        if self._topdown:
            self._topdown = False
            return self.getStayArmIndex()   
        else:
            return self.getUpArmIndex() # Up arm

