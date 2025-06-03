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
# Request Prometheus to pickup Sysbench performance during online trainings.



from datetime import datetime, timedelta
import numpy as np
import ast

 # SysbenchPerfMonitor get Sysbench performance from metrics sent to Prometheus during onlne training
 # LATER: The global config technique used here could be used in others python classes over there. 
 # See suggestions here: https://stackoverflow.com/questions/6198372/most-pythonic-way-to-provide-global-configuration-variables-in-config-py
class SysbenchPerfMonitor():
    __config = {
        "file_adresse": "./test/log.txt",
        "latency_metric": "adbms_autotuning_sysbench__latency",
        "long_latency_metric": "adbms_autotuning_sysbench__long_latency",
        "label_config": {
            'APP_NAME': 'workload_sb', 
            'MESSAGE_SEVERITY': 'informational'},
    }
    __setters = ["file_adresse"]

    @staticmethod
    def config(name):
        return SysbenchPerfMonitor.__config[name]

    @staticmethod
    def set(name, value):
        if name in SysbenchPerfMonitor.__setters:
            SysbenchPerfMonitor.__config[name] = value
        else:
            raise NameError("Name not accepted in set() method") 
    
    def __init__(self):
        self._file = self.__config["file_adresse"]
        self._start_time = datetime.now()
        self._end_time = None
    

    def _getMetricValues(self, metric_name="", time = None , params=None) -> np.ndarray:
        end = self._end_time or datetime.now()
        dicts = []
        
        with open(self._file) as file:
            lines = [line.rstrip() for line in file]
        for line in lines:
            dicky = ast.literal_eval(line)
            #print(dicky)
            # -7 to ignore details after millie seconds
            timestamp = datetime.strptime(dicky["ts"][:23], "%Y-%m-%dT%H:%M:%S.%f")
            if time is None:
                if  timestamp >= self._start_time and timestamp <= end :
                    dicts.append(dicky)
            else:
                if  timestamp >= self._start_time and timestamp <= end and timestamp >= (end - timedelta(seconds=time)) :
                    dicts.append(dicky)
            #print(timestamp >= self._start_time)
            #print(timestamp <= self._start_time)
            # print(type(dicky["ts"]))
            # print((dicky["ts"]))

    
        npvals = np.zeros(len(dicts))
        for idx in range(0, len(dicts)):
            npvals[idx] = float(dicts[idx][metric_name])
        if len(npvals) == 0:
            return None
        return npvals

    def startTime(self, time=None):
        self._start_time = time or datetime.now()
        self._end_time = None
        #print("START TIME: ", self._start_time)

    # Current throughput in Queries per second (client side), on last seconds (usually 5s, as defined in promtail configuration) is the Min observed during this time (pessimmist strategy)
    # Return -1 if no value is currently available.
    def getCurrentMeanQps(self) -> float:
        nplist = self._getMetricValues(metric_name='qps')
        if nplist is None:
            return -1.        
        return np.mean(nplist)

    # Current throughput in Transactions per second (client side) on last seconds (usually 5s, as defined in promtail configuration) is the Min observed during this time (pessimmist strategy)
    # Return -1 if no value is currently available.
    def getCurrentMeanTps(self) -> float:
        nplist = self._getMetricValues(metric_name='tps')
        if nplist is None:
            return -1.        
        return np.mean(nplist)

    # Current latency on last seconds (usually 5s, as defined in promtail configuration) is the Max observed during this time (pessimmist strategy)
    # Return -1 if no value is currently available.
    def getCurrentLatency(self) -> float:
        nplist = self._getMetricValues(metric_name='lat')
        if nplist is None:
            return -1.        
        return np.max(nplist)

    # Mean of Current latency on last 15s (usually 15s, as defined in promtail configuration)
    # Return -1 if no value is currently available.
    def getPast15secondsMeanLatency(self) -> float:
        nplist = self._getMetricValues(metric_name='lat', time = 15)
        if nplist is None:
            return -1
        else:
            return np.mean(nplist)
        

    # Mean of Current latency on last minute (usually 1min, as defined in promtail configuration)
    # Return -1 if no value is currently available.
    def getPast1minMeanLatency(self) -> float:
        nplist = self._getMetricValues(metric_name='lat',time = 60)
        if nplist is None:
            return -1
        else:
            return np.mean(nplist)
        


    # Sigma/Standard Normalized deviation of Current latency on last 5 minutes (as defined in promtail configuration)
    # Return -1 if no value is currently available.
    def _getPastScaledSigmaLatency(self,nplist) -> float:
        if nplist is None:
            return -1.        
        max = np.max(nplist)
        min = np.min(nplist)
        nplist_scaled = np.array([(x -min)/(max - min) for x in nplist])
        return np.std(nplist_scaled)

    def getPast5minScaledSigmaLatency(self) -> float:
        nplist = self._getMetricValues(metric_name='lat', time = 300)
        if nplist is None:
            return -1
        else:
            return self._getPastScaledSigmaLatency(nplist)

    def getPast15secondsScaledSigmaLatency(self) -> float:
        nplist = self._getMetricValues(metric_name='lat', time=15)
        if nplist is None:
            return -1
        else:
            return self._getPastScaledSigmaLatency(nplist)    
        
    def getPast1minScaledSigmaLatency(self) -> float:
        nplist = self._getMetricValues(metric_name='lat', time = 60)
        if nplist is None:
            return -1
        else:
            return self._getPastScaledSigmaLatency(nplist)

    # latency is reported Sysbench latency in milliseconds. -1 if information is not available
    # statements is reported Sysbench queries/s + transactions/s. 0 if information is not available or sum of both is zero.
    def getMetrics(self) -> dict:
        self._end_time = datetime.now()
        itvl = self._end_time.timestamp() - self._start_time.timestamp()

        metrics = { "interval":  timedelta(seconds=itvl).seconds }
        metrics["latency"] = self.getCurrentLatency()
        metrics["latency_mean"] = self.getPast15secondsMeanLatency()  
        metrics["latency_sigma"] = self.getPast1minScaledSigmaLatency()
        qps = self.getCurrentMeanQps()
        stmts = 0 if qps  == -1. else qps
        tps = self.getCurrentMeanTps()
        stmts += 0 if tps  == -1. else tps
        metrics["statements_mean"] = stmts

        self._end_time = None

        return metrics

    

