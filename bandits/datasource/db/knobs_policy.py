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

import random
import numpy as np
from datetime import datetime, timedelta

import bandits.datasource.db.dberrors as dberrors


# knobs_policy: {
#    <knob>_reset_policy: "min", "max", "mean", "default", "rand", "stay"
# }
#    The action_method on the knobs are an incrementation rule based on a direction -1, 0, or +1te the new value from an action. 
#
class DiscreteKnobsPolicy():
    ACTION_METHOD_INCR=0
    #ACTION_METHOD_SPEED=1  # It is useless and without any interest as done here. 

    def __init__(self):
        self._def = {}
        self._cur_knobs = {}
        self._roundrobin_trip_num = 0
        self._roundrobin_trip_list = [ "min", "mean", "max" ]
        self._pop_list = {}

    def get(self):
        out = self._def.copy()
        for k,v in out.items():
            if k.endswith("_last_update"):
                out[k] = v.ctime()

        return out

    def _discrete_array(self, minimum, maximum, increment):
        arr = np.arange(minimum, maximum + increment, increment)
        return arr.tolist()

    def _getMeanPerIncrement(self, min:float, max:float, incr:float):
        count = (max - min)/incr    
        return min + int(count//2)*incr
    
    def getKnobDiscreteValues(self, name: str):
        return self._discrete_array(self._def[name+"_min"], self._def[name+"_max"], self._def[name+"_incr"])

    def getKnobMinMaxValues(self, name: str):
        return self._def[name+"_min"], self._def[name+"_max"]

    # addIntKnob
    # This method adds an integer knob with its policy (range values, behaviour on reset, incrementation steps, and required warm up time)
    # The increment is the Minimal Increment value to add or substract from the current on a "drive" action.
    # The speed_factor multiply the drive direction to increase by a factor the speed of the displacement of the knob value. By this way the drive direction (see drive method)
    # requires a normalized direction in the range of the continous interval [-1.0 - +1.0]
    # Before driving this knobs, for the first time, it MUST be reset (method applyResetPolicy()) through the reset policy of your choice:
    #   stay (default): keep its current value
    #   min: set min at each reset
    #   max: set max at each reset
    #   mean: set the mean value in the parameter's range, at each reset
    #   default: apply default value of this knob, at each reset
    #   rand: apply a random value into the parameter's range, at each reset
    #   roundrobin: At each reset, get cycliquely the value defined by either "min", "mean", "max"
    #   pop: Pop a new value since last reset. It is a cycle, if all possible values have been already popped, restart from the first of the list
    def addIntKnob(self, name, min, max, default:int = None, incr=0, speed_factor: int=1, reset_policy="stay", warmup_time: int=0):
        self._cur_knobs[name] = None
        self._def[name+"_last_update"] = datetime.fromtimestamp(0.)

        self._def[name+"_min"] = min
        self._def[name+"_max"] = max
        if default is None:
            self._def[name+"_default"] = self._getMeanPerIncrement(min, max, incr)
        else:
            self._def[name+"_default"] = default

        diff = max - min
        if incr > diff:
            incr = diff 
        self._def[name+"_incr"] = incr

        self._def[name+"_reset_policy"] = reset_policy
        self._def[name+"_drive_policy"] = self.ACTION_METHOD_INCR  # The only method for the moment
        self._def[name+"_speed_factor"] = speed_factor

        self._def[name+"_warmup_time"] = warmup_time
        self._pop_list[name] = []

    # Get a normalized value in range [0.0-1.0] of the knob
    def getNormalizedValue(self, knob: str, value: int) -> float:
        min = self._def[knob+"_min"]
        max = self._def[knob+"_max"]
        return round((value - min)/(max - min), 3)

    # Get an indexed value in range [0, maxcount] of the knob
    def getIndexedValue(self, knob: str, value: int) -> int:
        min = self._def[knob+"_min"]
        incr = self._def[knob+"_incr"]
        return (value - min)//incr

    # Apply reset policy defined for all knobs added with the addIntKnob() method
    # Return the dictionary of new knobs values (can be the same than the knobs values passed as argument)
    # NOTE!! Whatever the case (a change or not) with set the start time to now (reset time) of the warmup period 
    def applyResetPolicy(self, knobs={}, force_default=False) -> dict:
        knobs_out = {}

        for k in self._cur_knobs.keys():
            cur_val = int(knobs[k])
            self._cur_knobs[k] = cur_val

            if force_default:
                p = "default"
            else:
                p = self._def[k+"_reset_policy"]

            if p == "roundrobin":
                p = self._roundrobin_trip_list[self._roundrobin_trip_num]
                self._roundrobin_trip_num = (self._roundrobin_trip_num + 1)%len(self._roundrobin_trip_list)

            match p:
                case "min":
                    knobs_out[k] = self._def[k+"_min"]
                case "max":
                    knobs_out[k] = self._def[k+"_max"]
                case "mean":  
                    # A mean computation which works with both integer or float values  
                    count = (self._def[k+"_max"] - self._def[k+"_min"])/self._def[k+"_incr"]    
                    knobs_out[k] = self._def[k+"_min"] + int(count//2)*self._def[k+"_incr"]
                case "default":    
                    knobs_out[k] = self._def[k+"_default"]
                case "rand":
                    count = (self._def[k+"_max"] - self._def[k+"_min"])/self._def[k+"_incr"]    
                    knobs_out[k] = self._def[k+"_min"] + random.randint(0, int(count))*self._def[k+"_incr"]
                case "stay":    
                    knobs_out[k] = cur_val
                case "pop":
                    if len(self._pop_list[k]) == 0:
                        self._pop_list[k] = self.getKnobDiscreteValues(k)
                    list = self._pop_list[k]
                    knobs_out[k] = list.pop(0)
                case _:
                    knobs_out[k] = cur_val

            self._def[k+"_warming_up_on_drive"] = True if cur_val != knobs_out[k] else False

        return knobs_out

    def _getLastWarmupSeconds(self, knob: str) -> int:
        now = datetime.now()
        interval = now.timestamp() - self._def[knob+"_last_update"].timestamp()
        return timedelta(seconds=interval).seconds


    # Directions for each knobs: -1=down 0=stay, 1=up
    # Up and down are multiplied by the speed_factor attribute of this knob and the minimal increment
    # IMPORTANT NOTE: With DDPG, that used "noisy" actions, the direction can be a float value
    # KnobDriveError
    def _changeKnobIntValue(self, knob: str, curVal: int, direction: float) -> int:
        increment = self._def[knob+"_incr"]
        speedFactor = self._def[knob+"_speed_factor"]
        newVal = curVal + round(direction*speedFactor)*increment

        minVal = self._def[knob+"_min"]
        maxVal = self._def[knob+"_max"]

        match direction:
            case direction if direction < 0:
                if newVal < minVal:
                    if curVal == minVal:
                        raise dberrors.KnobDriveError(knob, str(curVal), str(newVal))
                    newVal = minVal
            case 0:
                pass
            case direction if direction > 0:
                if newVal > maxVal:
                    if curVal == maxVal:
                        raise dberrors.KnobDriveError(knob, str(curVal), str(newVal))
                    newVal = maxVal

        warmup_duration = self._def[knob+"_warmup_time"]    
        if warmup_duration > 0:           
            deltanow = self._getLastWarmupSeconds(knob)

            if deltanow > warmup_duration:            
                self._def[knob+"_warming_up_on_drive"] = False
            else:
                self._def[knob+"_warming_up_on_drive"] = True
                if newVal != curVal:
                    raise dberrors.KnobDriveWarmupError(knob, str(curVal), str(newVal))
            
        
        return newVal


    # Compute new knobs values from current knobs and directions passed as arguments.
    # In case of error, an exception is raised. 
    # Directions for each knobs: -1,....=down 0=stay, 1,...=up
    # The absolute value specify the speed in the direction.
    # 2 grows twice faster the knob in UP direction. N to grows by a factor of N.
    # -2 reduces twice faster the knob in DOWN direction. -N to reduces by a factor of N.
    # In case of error (see raises) to set ONE knob, a rollback to the previous knobs values is operated.
    # If no exception is rased, we consider that the agent reached its new direction.
    #
    # Raises:
    # dberrors.KnobDriveError (Knob is driven to a wrong value, i.e. over its min or max limit) 
    # dberrors.KnobDriveWarmupError (Knob has been changed too recently, the database didn't end up it warmup time)
    def driveKnobsFromAction(self, knobs={}, directionsVector=[]) -> dict:
        knobs_out={}
        index=0
        for key, v in knobs.items():
            self._cur_knobs[key] = int(v) # Ensure to be synched with the current values
            direction = directionsVector[index]
            #print(f"KNOB: {key} DIRECTION: {direction}" )
            knobs_out[key] = self._changeKnobIntValue(key, int(v), direction)  
            index += 1

        #print("KNOB OUT:", knobs_out)
        return knobs_out

    def getWarmupStatusAtLastDriveAction(self) -> int:
        for knob in self._cur_knobs.keys():
            if self._def[knob+"_warmup_time"] > 0 and self._def[knob+"_warming_up_on_drive"]:
                return 1
        
        return 0         

    def getWarmupStatus(self) -> int:
        for knob in self._cur_knobs.keys():
            warmup_duration = self._def[knob+"_warmup_time"]    
            if warmup_duration > 0:           
                deltanow = self._getLastWarmupSeconds(knob)
                if deltanow <= warmup_duration:
                    return 1 
        
        return 0         


    # Store current knobs value and ensure to set the "warming_up state if the last update time is less than the required warmup_time for each parameter.
    # Return: True if one (ore more) knob(s) value is changed
    def commitNewKnobs(self, knobs: dict) -> bool:
        change = False
        
        for k, v in knobs.items():
            newVal = int(v)
            if newVal != self._cur_knobs[k]:
                if  self._def[k+"_warmup_time"] > 0:           
                    self._def[k+"_last_update"] = datetime.now()
                change = True
                self._cur_knobs[k] = newVal
        
        return change
