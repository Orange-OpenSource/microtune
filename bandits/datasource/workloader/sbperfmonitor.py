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

# See https://pypi.org/project/prometheus-api-client/
from prometheus_api_client import PrometheusConnect
#from prometheus_api_client.utils import parse_datetime
import time
from datetime import datetime, timedelta
import numpy as np


 # SysbenchPerfMonitor get Sysbench performance from metrics sent to Prometheus during onlne training
 # LATER: The global config technique used here could be used in others python classes over there. 
 # See suggestions here: https://stackoverflow.com/questions/6198372/most-pythonic-way-to-provide-global-configuration-variables-in-config-py
class SysbenchPerfMonitor():
    __config = {
        "host": "prometheus.local",
        "port": "9090",
        "disable_ssl": True,
        "latency_metric": "adbms_autotuning_sysbench__latency",
        "long_latency_metric": "adbms_autotuning_sysbench__long_latency",
        "label_config": {
            'APP_NAME': 'workload_sb', 
            'MESSAGE_SEVERITY': 'informational'},
    }
    __setters = ["host", "port","disable_ssl","label_selector"]

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
        self._prometheusUrl = "http://"+self.__config["host"]+":"+self.__config["port"]
        self._prom = PrometheusConnect(url =self._prometheusUrl, disable_ssl=self.__config["disable_ssl"])
        self._start_time = datetime.now()
        self._end_time = None

    # NOT USED
    def _getMetricValuesInRange(self, metric_name="", params=None) -> np.ndarray:
        end = self._end_time or datetime.now()
        list = self._prom.get_metric_range_data(metric_name=metric_name, label_config=self.__config["label_config"], start_time=self._start_time, end_time=end, params=params)
        if len(list) == 0:
            return None

        npvals = np.zeros(len(list))
        for idx in range(0, len(list)):
            tsvals = list[idx]['values']
            npvals[idx] = float(tsvals[0][1])

        return npvals

    def _getMetricValues(self, metric_name="", params=None) -> np.ndarray:
        list = self._prom.get_current_metric_value(metric_name=metric_name, label_config=self.__config["label_config"], params=params)
        if len(list) == 0:
            return None

        npvals = np.zeros(len(list))
        for idx in range(0, len(list)):
            tsvals = list[idx]['value']
            npvals[idx] = float(tsvals[1])

        return npvals
    
    # DEPRECATED
    def startTime(self, time=None):
        self._start_time = time or datetime.now()
        self._end_time = None
        print("DEPRECATED CALL sbperfmonitor.startTime(): START TIME: ", self._start_time)

    def _getMean(self, nplist=None) -> float:
        if nplist is None:
            return -1.        
        return np.mean(nplist)

    # Ecart-type (Sigma)
    def _getStd(self, nplist=None) -> float:
        if nplist is None:
            return -1.        
        return np.std(nplist)

    # Current throughput in Queries per second (client side), on last seconds (usually 3s, as defined in promtail configuration) is the Min observed during this time (pessimmist strategy)
    # Return -1 if no value is currently available.
    def getCurrentMeanQps(self) -> float:
        nplist = self._getMetricValues(metric_name='adbms_autotuning_sysbench__qps')
        return self._getMean(nplist)

    # Current throughput in Transactions per second (client side) on last seconds (usually 5s, as defined in promtail configuration) is the Min observed during this time (pessimmist strategy)
    # Return -1 if no value is currently available.
    def getCurrentMeanTps(self) -> float:
        nplist = self._getMetricValues(metric_name='adbms_autotuning_sysbench__tps')
        return self._getMean(nplist)

    # Current latency on last seconds (usually 5s, as defined in promtail configuration) is the Max observed during this time (pessimmist strategy)
    # Return -1 if no value is currently available.
    def getCurrentLatency(self) -> float:
        nplist = self._getMetricValues(metric_name='adbms_autotuning_sysbench__latency')
        return self._getMean(nplist)

    # Mean of Current latency on last 15s (usually 15s, as defined in promtail configuration)
    # Return -1 if no value is currently available.
    def getPast15secondsMeanLatency(self) -> float:
        nplist = self._getMetricValues(metric_name='adbms_autotuning_sysbench__fifteensec_latency')
        return self._getMean(nplist)

    # Mean of Current latency on last minute (usually 1min, as defined in promtail configuration)
    # Return -1 if no value is currently available.
    def getPast1minMeanLatency(self) -> float:
        nplist = self._getMetricValues(metric_name='adbms_autotuning_sysbench__onemin_latency')
        return self._getMean(nplist)

    # Sigma/Standard Normalized (in space 0 to 0.1s) deviation of Current latency on last X s/minutes, as defined in promtail configuration, thus by the list past as argument
    # Return -1 if no value is currently available.
    def _getPastScaledSigmaLatency(self, nplist=None) -> float:
        if nplist is None:
            return -1.        
#        max = np.max(nplist)
#        min = np.min(nplist)
        max = 100
        min = 0
        nplist_scaled = np.array([(x -min)/(max - min) for x in nplist])
        return np.std(nplist_scaled)

    def getPast5minScaledSigmaLatency(self) -> float:
        nplist = self._getMetricValues(metric_name='adbms_autotuning_sysbench__fivemin_latency')
        return self._getPastScaledSigmaLatency(nplist)

    def getPast15secondsScaledSigmaLatency(self) -> float:
        nplist = self._getMetricValues(metric_name='adbms_autotuning_sysbench__fifteensec_latency')
        return self._getPastScaledSigmaLatency(nplist)

    def getPast1minScaledSigmaLatency(self) -> float:
        nplist = self._getMetricValues(metric_name='adbms_autotuning_sysbench__onemin_latency')
        return self._getPastScaledSigmaLatency(nplist)

    # latency is reported Sysbench latency in milliseconds. -1 if information is not available
    # statements is reported Sysbench queries/s + transactions/s. 0 if information is not available or sum of both is zero.
    def getMetrics(self) -> dict:
        #self._end_time = datetime.now() # DEPRECATED
        #itvl = self._end_time.timestamp() - self._start_time.timestamp()

        nplist1m = self._getMetricValues(metric_name='adbms_autotuning_sysbench__onemin_latency')
        nplist15s = self._getMetricValues(metric_name='adbms_autotuning_sysbench__fifteensec_latency')

        #metrics = { "interval":  timedelta(seconds=itvl).seconds }
        metrics = {}
        metrics["latency"] = self.getCurrentLatency()
        metrics["latency_mean"] = self._getMean(nplist15s)  
        metrics["latency_mean_minute"] = self._getMean(nplist1m)  
        metrics["latency_sigma"] = self._getPastScaledSigmaLatency(nplist15s)
        metrics["latency_sigma_minute"] = self._getPastScaledSigmaLatency(nplist1m)
        qps = self.getCurrentMeanQps()
        stmts = 0 if qps  == -1. else qps
        tps = self.getCurrentMeanTps()
        stmts += 0 if tps  == -1. else tps
        metrics["statements_mean"] = stmts

        self._end_time = None

        return metrics

    
class SysbenchMetricsPicker():
    def __init__(self,  
                 warmups: dict = None, # {'on_start': 50.0, 'on_buf_update': 20.0, 'sigma_latency_toleration': None or > 0.}
                 observation_time=3.,
                 monitor: SysbenchPerfMonitor = None):
        self._monitor = monitor if monitor is not None else SysbenchPerfMonitor()
        self._warmups = warmups
        self._metrics_observation_time = observation_time  # Prometheus metrics
        self._max_duration = max(self._warmups["on_buf_update"], self._metrics_observation_time)

    def _startSigmaSession(self):
        self._total_time = 0
        self._sigma_min = np.Inf
        self._best_metrics = None
        self._best_metrics_time = 0

    def _pickup(self): # -> (bool, bool, dict):
        time.sleep(self._metrics_observation_time)
        self._total_time += self._metrics_observation_time
        timeout = True if self._total_time >= self._max_duration else False

        metrics = self._monitor.getMetrics()

        sigma = metrics["latency_sigma"]
        stable = False if sigma < 0. or sigma >= self._warmups["sigma_latency_toleration"] else True
        # Note that if sigma is < 0, best-metrics is the last collected
        if sigma < self._sigma_min:
            self._best_metrics = metrics
            self._best_metrics_time = self._total_time
            self._sigma_min = np.inf if sigma < 0. else sigma 

        return timeout, stable, metrics

    def _mostStableMetrics(self) -> dict:
        return  self._best_metrics
      
    # In case of NO usage of sigma latency value, the best metrics are assumed to be took at the end of a long waiting time (tto lt the database to stabilize).
    # In case of usage of the sigma latency value, a research session is started:
    #   - The best metrics are the one picked up as soon as a low sigma value (one under a threshold) is reached after a change (and during a stable workload and server).
    #   - If the timeout duration is reached, the algo will take the metrics picked up at the minimum sigma value during the session.
    #
    def getBestMetrics(self) -> dict:
        res = {}
        sigma_toleration = self._warmups.get("sigma_latency_toleration")
        if sigma_toleration is None or sigma_toleration == 0:
            headroom_wait = max(0, self._max_duration-self._metrics_observation_time)
            print(datetime.now(), "Wait at step time:", headroom_wait, "s headroom for DB stabilisation...")
            time.sleep(headroom_wait)

            #self._monitor.startTime() # DEPRECATED (managed at Prometheus level): Start now the perf observation during this step
            #self._sbperf_local.startTime()
            print(datetime.now(), "Wait at step time:", self._metrics_observation_time, "s for mean measure...")
            time.sleep(self._metrics_observation_time)
            res = self._monitor.getMetrics()
        else:
            timeout = False
            stable = False
            self._startSigmaSession()
            
            print(datetime.now(), "Get perf metrics with",self._metrics_observation_time,"s. observation time...")
            while not stable and not timeout:
                timeout, stable, metrics = self._pickup()
                print(datetime.now(), "Wait time:", self._total_time, "/", self._max_duration, "Cur Sigma latency:", metrics["latency_sigma"])

            if not stable and timeout:
                res = self._mostStableMetrics()
                print(datetime.now(), "Finally choose the best session metrics with sigma", res["latency_sigma"], "at", self._best_metrics_time, "s.")
            else:            
                res = metrics

        return res
 