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


import time
from datetime import datetime
import sys
#from pandas import DataFrame
import numpy as np
from sklearn.preprocessing import MinMaxScaler 

from bandits.datasource.db.dbadminmysql import DBAdminMySql
from bandits.datasource.db.knobs_policy import DiscreteKnobsPolicy
import bandits.datasource.db.dberrors as dberrors
from bandits.datasource.workloader.sbperfmonitor import SysbenchMetricsPicker

from bandits.datasource.dataset import ADBMSDataSetEntryContextSelector

import logging
import logging_loki

logging_loki.emitter.LokiEmitter.level_tag = "info"
# assign to a variable named handler 
handler = logging_loki.LokiHandler(
   url="http://loki.local:3100/loki/api/v1/push",
   tags={"APP_NAME": "trainer"},
   auth=("admin", "admin"),
   version="1",
)

# create a new logger instance, name it whatever you want
logger = logging.getLogger("adbms-poc-trainer")
#logger.addHandler(handler)


def knobs_prettifier(knobs={}):
    output={}

    for k in knobs:
        match k:
            case "innodb_buffer_pool_size":
                val = int(knobs[k])//1024//1024
                output[k] = str(val)+" MB"
            case "max_connections":
                output[k] = int(knobs[k])
            case _:
                output[k] = str(knobs[k])
    
    return output

from collections.abc import MutableMapping
import pandas as pd

def flatten_dict(d: MutableMapping, sep: str= '.') -> MutableMapping:
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient='records')
    return flat_dict

# A subclass of DataSetEntryContextSelector that redefine Group into Workload and entry as a state for a specific Buffer Cache Size 
# It provides methods to change group and entry in group either randomly or sequentialy.
# The default following methods are implemented to provide a default behaviour:
#   - reset(group_idx) - Go to the group group_ix and set the default entry randomly (with a uniform distribution) 
#   - next() - Go randomly (with an uniform distribution) to a new entry in the current group. 
#   - move(increment) - Change to a new current entry in the current group from a relative increment to the current position (index) 
# By default, at the instanciation, the current default is the first entry in the dataset (index=0)
# The reset(), move() and next() methods can ve overriden by a sub-class in order to implement a specific behaviour
class ADBMSBufferCacheStatesLive(ADBMSDataSetEntryContextSelector):
    def __init__(self, perf_target_level= 0.98, qpslat_w="01",                 
                 ram_limit: int = 8589934592,   # 8GB
                 buf_reset_policy="stay",
                 db_warmup_time=6,
                 metrics_picker: SysbenchMetricsPicker = None,  # {'on_start': 50.0, 'on_buf_update': 20.0, 'use_sigma_metric': True}
                 dba: DBAdminMySql = None,
                 context_elems=None, normalize=False, with_scaler:MinMaxScaler=None):
        self._qpslat_w = qpslat_w
        self._dba = dba
        self._knobs_policy = None
        self._sbperf = metrics_picker
        self.db_warmup_time = db_warmup_time
        self._dbstatus_observation_time = 5.
        #self._metrics_observation_time = 3.  # Prometheus metrics
        #state = self.state()
        #self.latency_range = state["latency_mean_max"] - state["latency_mean_min"]
        self.lat_gap = 0
        self._latgap_ms = "NA"
        self.lat_target = -1
        self._iperf_target = perf_target_level
        #self._action2incr = np.vectorize(lambda x:-x)
        self._ram_limit = ram_limit
        self.innodb_buffer_pool_size_reset_policy = buf_reset_policy
        self._createOnceKnobsPolicy()
        self._cur_state = {}
        self._dba.collectGlobalStatusValues(observation_time=1, complete=True)
        self._initial_usage = self._dba.getDBUsageStatus()
        print("DB USAGE on reset: ", self._initial_usage)

        super().__init__(group_list=["LIVE"], entries_per_group=self._entries_count, context_elems=context_elems, normalize=normalize, with_scaler=with_scaler)
        #self._context_elems = None
        #self.min_max_scaler = None

    def _createOnceKnobsPolicy(self):
        if self._knobs_policy is not None:
            return
        
        self._knobs_policy = DiscreteKnobsPolicy()
        
        min_buf = self._dba.getMinBufferPoolSize()
        max_buf = max(min_buf*2, self._ram_limit)
        buf_incr = min_buf
        speed_factor = 1

        dft_buf = None # Rely on knob policy to compute the mean value as default

        # Knobs MySQL 8.0.30 doc:
        #  innodb_buffer_pool_chunk_size: 5242880 (5MB), 9223372036854775807, a multiple of (chunck_size*pool_instances), should not exceed 80% of available RAM
        #  max_connections: 10, 1000000
        # 536870912=512MB, 1073741824=1GB, 2147483648=2GB, 4294967296=4GB, 5368709120=5GB
        self._knobs_policy.addIntKnob("innodb_buffer_pool_size", 
                                      min=min_buf, max=max_buf, incr=buf_incr, speed_factor=speed_factor, default=dft_buf, 
                                      reset_policy=self.innodb_buffer_pool_size_reset_policy,
                                      warmup_time=0)

        d_values = self._knobs_policy.getKnobDiscreteValues("innodb_buffer_pool_size")
        self._entries_count = len(d_values)
        print("** BUFFER VALUES:", d_values, self._entries_count, " values")

    def _agent_location(self, dbstatus_observation_time=1, complete: bool = False):
        obs = {}
        obs["innodb_buffer_pool_size"] = int(self._cur_knobs["innodb_buffer_pool_size"])
        obs["normalized_buf_size"] = self._knobs_policy.getNormalizedValue("innodb_buffer_pool_size", obs["innodb_buffer_pool_size"])

        # Manage to get OBS's Status parameters (calculated from DB Global variables)
        self._dba.collectGlobalStatusValues(observation_time=dbstatus_observation_time, complete=complete)
        status = self._dba.getDBStatus()

        pool_wait_free = int(status["innodb_buffer_pool_wait_free_diff"])
        pool_write_requests = int(status["innodb_buffer_pool_write_requests_diff"])
        if pool_write_requests == 0:
            obs["write_wait_ratio"] = 0
        else:
            obs["write_wait_ratio"] = float(pool_wait_free)/pool_write_requests
        print("TRACE:", "pool_wait_free", pool_wait_free, "pool_write_requests", pool_write_requests, "write_wait_ratio", obs["write_wait_ratio"])

        pool_pages_total = int(status["innodb_buffer_pool_pages_total_end"])
        pool_pages_free = int(status["innodb_buffer_pool_pages_free_end"])
        obs["cache_free_pages_ratio"] = round(float(pool_pages_free)/pool_pages_total, 4)
#        print("TRACE:", "pool_pages_free", pool_pages_free, "pool_pages_total", pool_pages_total, "cache_free_pages_ratio", obs["cache_free_pages_ratio"])
        pool_pages_used = int(status["innodb_buffer_pool_pages_data_end"])
        obs["cache_used_pages_ratio"] = round(float(pool_pages_used)/pool_pages_total, 4)
        print("TRACE:", "pool_pages_used", pool_pages_used, "pool_pages_total", pool_pages_total, "cache_used_pages_ratio", obs["cache_used_pages_ratio"])

        reads_on_disk = int(status["innodb_buffer_pool_reads_diff"])
        read_requests = int(status["innodb_buffer_pool_read_requests_diff"])
        if read_requests == 0:
            obs["cache_hit_ratio"] = 0
        else:
            obs["cache_hit_ratio"] = 1. - (float(reads_on_disk)/read_requests)
        print("TRACE:", "reads_on_disk", reads_on_disk, "read_requests", read_requests, "cache_hit_ratio", obs["cache_hit_ratio"])

        obs["wl_clients"] = int(status["threads_connected_end"]) -1 # Remove 1 for the agent's connection

        return obs

    def _get_info(self):
        info = {}
        
        k = self._dba.getKnobsToDrive(sync_from_db=False)
        info["bufsize"] = int(k["innodb_buffer_pool_size"])
        
        info["usage"] = self._dba.getDBUsageStatus()
        info["usage"]["db_size_initial"] = self._initial_usage["db_size"]
        info["global_status"] = self._dba.getDBStatus()
        info["other_knobs"] = self._dba.getOtherKnobs()
        #print(f'Info usage: {info["usage"]} glog status: {info["global_status"]}')

        nosb = {"latency": -1, "latency_mean": -1, "latency_mean_minute": -1, "latency_sigma": -1, "latency_sigma_minute": -1}
        info["sysbench"] = nosb if self._sbperf is None else self._sbperf.getBestMetrics()
        info["sysbench_local"] = nosb

        return info

    def getWorkloadNameId(self):
        return self._group_list[0]

    def getQPSLatWeigths(self):
        return self._qpslat_w
    
    # Return IPERF, IDELTAPERF, IPERFTARGET
    def getIPerfIndicators(self):
        return (0, 0, 0)

    def getLatency(self, filtered=True):
        return 0
    
    # Return Latency gap, Latency gap in ms (str), latency target (threshold)
    def getLatencyGapTarget(self):
        return self.lat_gap, self._latgap_ms, self.lat_target    

    def getTotalStatesCount(self):
        return self._entries_count
    
    def getWorkloadsCount(self):
        return len(self._group_list)
    
    def getWorkloadsList(self):
        return self._group_list
    
    def getStatesCountPerWorkload(self):
        return self._entries_count

    # Update current State and Context vector
    def _set_cur_state(self):
        print(datetime.now(), "Collect all observed data...")
        obs = self._agent_location(dbstatus_observation_time=self._dbstatus_observation_time, complete=True)
        print("## TUNING OBSERVATION: ", knobs_prettifier(obs))
        print(datetime.now(), "End of waiting time.")

        info = self._get_info()
        
        buf_min = self._dba.getMinBufferPoolSize()
        bmin, _ = self._knobs_policy.getKnobMinMaxValues("innodb_buffer_pool_size")
        assert buf_min == bmin, "Discrepancy. An update on Database configuration impacted its Min buffer value size"

        dbsize = int(info["usage"]["db_size"]["TableSizeMB"])+int(info["usage"]["db_size"]["IndexSizeMB"]) 
        state = {
            "wl_clients" : obs["wl_clients"],
            "buf_size" : obs["innodb_buffer_pool_size"],
            "buf_size_min_mb": buf_min//1024//1024,
            "db_size_mb" : dbsize,
            "buf_size_idx": self._knobs_policy.getIndexedValue("innodb_buffer_pool_size", obs["innodb_buffer_pool_size"]),
            "observation": obs,
            "extra_info" : info,
        }
        if info.get("sysbench"):
            state["sysbench"] = info["sysbench"]

        self._cur_state = flatten_dict(state)
        #print(f'CURSTATE:{self._cur_state}')
        self.entry_idx = self.getStatesCountPerWorkload() - self._cur_state["buf_size_idx"] -1

        # Update context vectors by using a single entry vector to normalize or not
        vectors = np.zeros((1, len(self._context_elems)), dtype=float)   

        for idx, name in enumerate(self._context_elems):
            vectors[0][idx] = float(self._cur_state[name])

        if self.min_max_scaler:
            vectors = self._normalizeNumpyFloatElems(vectors)
        
        # Update current context at buffer idx with Normalized (or not!) values
        self._context_vectors[self.globalIndex()] = vectors[0]

        return self._cur_state

    # Return current state (if bufferidx_increment is 0) or the state at an another buffer index value.
    # Return None if the buffer idx resulting to the current buffer+the increment is out of range.
    def state(self, bufferidx_increment=0):
        assert bufferidx_increment == 0, f"WARNING: on a live case with a DB, requiering the state at another buffer value than the current value is not supperted. Current state is always returned. You should use move(buf_incr) instead."

        return self._cur_state
    
    
    # DUMMY Action function as it is not possible to compute Reward for other buf_inf than 0 (current state). So Regret can not be obtained too!
    # Apply a function to each buffer's action
    def applyLambda2actions(self, actions: np.array=None,  buffunc=lambda action_idx, real_bufferidx_increment:None):
        for idx in range(len(actions)):
            incr = actions[idx] 
            buffunc(idx, incr) 

    # Reset position onto a new workload and a new buffer index in the workload's entries
    def reset(self, workload_idx: int = 0):
        try:
            self._dba.sanity()           

            self._createOnceKnobsPolicy() # Creates it only once!!
            
            self._cur_knobs = self._dba.getKnobsToDrive(sync_from_db=True)           
            print("CUR KNOBS on reset: "+str(knobs_prettifier(self._cur_knobs)))
            new_knobs = self._knobs_policy.applyResetPolicy(self._cur_knobs)

            self._dba.driveDynamicKnobs(new_knobs) # Set DB's knobs to initial value fixed by our reset policy           
            updated = self._knobs_policy.commitNewKnobs(new_knobs) 
            print("NEW KNOBS on reset: "+str(knobs_prettifier(new_knobs))+" UPDATED: "+str(updated))
            self._cur_knobs = new_knobs

            # Apply always the warmup time, whatever a change on knobs has been made or not....
            print(datetime.now(), "Wait at Episode time:", self.db_warmup_time, "s ...")
            time.sleep(self.db_warmup_time)

            self._set_cur_state()
            self._initial_usage = self._dba.getDBUsageStatus()

            print("DB USAGE on reset: ", self._initial_usage)

        except (dberrors.KnobGetError,dberrors.KnobSetError,dberrors.KnobUpdateInProgress,dberrors.DBStatusError, dberrors.StatusVarGetError)  as err:
            print(err)
            extra = {"tags": {"ctx": "reset", "err": err.msg, "wid": workload_idx}}
            logger.error("Exception",
                        extra=extra,
            )
                                
        info = {}
        info["knobs_policy"] = self._knobs_policy.get()
        info["usage"] = self._dba.getDBUsageStatus()
        obs = [v for k, v, in self._cur_state.items() if k.startswith("observation.")]

        extra = {"tags": {"ctx": "reset", "wid": workload_idx}, "observation": obs, "info": info }
        logger.info("Reset done",
                    extra=extra,
        )
        


    def next(self):
        pass

    # Move in current group
    # Here incr is used as an increment on current index (in dataframe)
    # action: -N for DOWN, 0 for STAY, +N for UP
    # Return the real increment applied, that can be different than the one in argument if the move goes out of range (in group) or in case of database error
    def move(self, bufferidx_increment=0):
        real_incr = 0
        sleeptime_err = 0 
        try:
            self._cur_knobs = self._dba.getKnobsToDrive(sync_from_db=True)
            print("CUR KNOBS: ", knobs_prettifier(self._cur_knobs))
            new_knobs = self._knobs_policy.driveKnobsFromAction(knobs=self._cur_knobs, directionsVector=[bufferidx_increment])

            self._dba.driveDynamicKnobs(new_knobs)
            updated = self._knobs_policy.commitNewKnobs(new_knobs)
            print("NEW KNOBS: ", knobs_prettifier(new_knobs))
            if updated:
                self._cur_knobs = new_knobs

            real_incr = bufferidx_increment

            # HOT PATCH!!! TO BE FIXED Apply always the warmup time, whatever a change on knobs has been made or not....
            if self._sbperf is None:
                print(datetime.now(), "No SbPerfMonitor enabled -> Wait at Step time:", self.db_warmup_time, "s ...")
                time.sleep(self.db_warmup_time)

        except dberrors.KnobDriveError as err:
            sleeptime_err = self.db_warmup_time
            #print("ERROR", err)
            #self._cur_knobs["innodb_buffer_pool_size"] = int(err.newVal)
            #self._set_cur_state()
            real_incr = 0
        except dberrors.KnobUpdateInProgress as err:
            sleeptime_err = self.db_warmup_time
            print("ERROR", err)
        # Useless exception, warmup is managed now by the configuration and no longer KnobPolicy on knob update
        except dberrors.KnobDriveWarmupError as err:
            sleeptime_err = self.db_warmup_time
            print("!!!!!!!!!!!!!!!!! KNOB WARMUP EXCEPTION ?!")
            print(err)
        except (dberrors.KnobGetError, dberrors.KnobRollbackError, dberrors.KnobSetError) as err:
            sleeptime_err = self.db_warmup_time
            print("!!!!!!!!!!!!!!!!! DATABASE ACCESS TO KNOBS EXCEPTION ?!")
            print(err)
        except:
            sleeptime_err = self.db_warmup_time
            print("!!!!!!!!!!!!!!!!! UNKNOWN EXCEPTION ?!")
            print("ERROR", sys.exc_info()[0])

        try:
            self._set_cur_state()
        except (dberrors.DBStatusError, dberrors.StatusVarGetError) as err:
            sleeptime_err = self.db_warmup_time
            print("!!!!!!!!!!!!!!!!! DATABASE ACCESS TO GET OBSERVATION PARAMS ?!")
            print("ERROR", err)
        except:
            sleeptime_err = self.db_warmup_time
            print("!!!!!!!!!!!!!!!!! UNKNOWN EXCEPTION ?!")
            print("ERROR", sys.exc_info()[0])

        time.sleep(sleeptime_err)

        return real_incr

    def prepareContext(self, context_elems=[], check_data=False, normalize=False, with_scaler:MinMaxScaler=None):
        self._reinit(context_elems, with_scaler)
        assert normalize is False or with_scaler is not None, "On a live DB, a normalized context can only be obtained with a provided MinMax scaler."

        print(f'Prepare Context ({self._total_entries_count}, {len(context_elems)})')
        #vectors = np.array([[float(row[x]) for x in self._context_elems] for index, row in self._df.iterrows()])
        vectors = np.zeros((self._total_entries_count, len(context_elems)), dtype=float)    

        if normalize:
            vectors = self._normalizeNumpyFloatElems(vectors)

        self._context_vectors = vectors

"""     
global_status = [
        "Bytes_received",
        "Connections",
        "threads_created",
        "threads_running",
        "threads_connected",
        "max_used_connections",
        "aborted_clients",
        "aborted_connects", 
        "innodb_buffer_pool_bytes_data", # The total number of bytes in the InnoDB buffer pool containing data. The number includes both dirty and clean pages
        "innodb_buffer_pool_bytes_dirty", # The total current number of bytes held in dirty pages in the InnoDB buffer pool
        "innodb_buffer_pool_pages_data", # The total number of pages in the InnoDB buffer pool containing data. The number includes both dirty and clean pages
        "innodb_buffer_pool_pages_total", # Total count of pages in the memory buffer
        "innodb_buffer_pool_pages_free",  # Free pages count in the memory buffer 
        "innodb_buffer_pool_pages_misc", # Is = innodb_buffer_pool_pages_total-innodb_buffer_pool_pages_free.The number of pages that are busy because they have been allocated for administrative overhead such as row locks or the adaptive hash index
        "innodb_buffer_pool_reads",         # The number of physical reads of a page from disk, ie, cache misses because not satisfied from the buffer pool
        "innodb_buffer_pool_read_requests", # NA in MariaDB. The number of read requests of a page from the Cache that came either from disk or from the buffer pool
        "innodb_buffer_pool_wait_free", # Total number of waits for pages to be flushed first from the InnoDB buffer pool
        "innodb_buffer_pool_write_requests", # The number of writes done to the InnoDB buffer pool
    ]
 """
# Converted from config/db/variables/mariadb.yaml with GPT3.5
global_status = [
    'Bytes_received',  # Bytes received
    'Connections',  # Connections
    'threads_created',  # Number of threads created
    'threads_running',  # Number of threads running
    'threads_connected',  # Number of threads connected
    'max_used_connections',  # Maximum number of used connections
    'aborted_clients',  # Number of aborted clients
    'aborted_connects',  # Number of aborted connects
    'created_tmp_files',  # Number of temporary files created
    'created_tmp_tables',  # Number of internal temporary tables created
    'created_tmp_disk_tables',  # Number of on-disk temporary tables created
    'binlog_cache_disk_use',  # Number of transactions that used the temporary binary log cache and exceeded binlog_cache_size
    'binlog_stmt_cache_disk_use',  # Number of transactions that used the temporary binary log statement cache and exceeded binlog_cache_size
    'handler_delete',  # Number of row deletions
    'handler_update',  # Number of row updates
    'handler_write',  # Number of row inserts
    'innodb_pages_created',  # Number of pages created by operations on InnoDB tables
    'innodb_pages_read',  # Number of pages read from the InnoDB buffer pool by operations on InnoDB tables
    'innodb_pages_written',  # Number of pages written by operations on InnoDB tables
    'innodb_rows_deleted',  # Number of rows deleted from InnoDB tables
    'innodb_rows_inserted',  # Number of rows inserted into InnoDB tables
    'innodb_rows_read',  # Number of rows read from InnoDB tables
    'innodb_rows_updated',  # Number of rows updated in InnoDB tables
    'innodb_row_lock_current_waits',  # Number of current row lock waits in InnoDB
    'innodb_row_lock_time',  # Total time spent in row lock waits in InnoDB
    'innodb_row_lock_time_avg',  # Average time spent in row lock waits in InnoDB
    'innodb_row_lock_time_max',  # Maximum time spent in row lock waits in InnoDB
    'innodb_row_lock_waits',  # Number of row lock waits in InnoDB
    'innodb_buffer_pool_bytes_data',  # Total number of bytes in the InnoDB buffer pool containing data
    'innodb_buffer_pool_bytes_dirty',  # Total number of bytes held in dirty pages in the InnoDB buffer pool
    'innodb_buffer_pool_pages_data',  # Total number of pages in the InnoDB buffer pool containing data
    'innodb_buffer_pool_pages_total',  # Total count of pages in the InnoDB buffer pool
    'innodb_buffer_pool_pages_free',  # Free pages count in the InnoDB buffer pool
    'innodb_buffer_pool_pages_misc',  # Number of pages that are busy due to administrative overhead in the InnoDB buffer pool
    'innodb_buffer_pool_reads',  # Number of physical reads of a page from disk in the InnoDB buffer pool
    'innodb_buffer_pool_read_requests',  # Number of read requests of a page from the cache in the InnoDB buffer pool
    'innodb_buffer_pool_wait_free',  # Total number of waits for pages to be flushed from the InnoDB buffer pool
    'innodb_buffer_pool_write_requests',  # Number of writes done to the InnoDB buffer pool
    'innodb_data_read',  # Number of InnoDB bytes read since server startup
    'innodb_data_reads',  # Number of InnoDB read operations
    'innodb_data_writes',  # Number of InnoDB write operations
    'innodb_data_written'  # Number of InnoDB bytes written since server startup
]


information_schemas= [
  "innodb_metrics"
#  "innodb_buffer_pool_stats"
#  innodb_buffer_page
#  innodb_lock_waits
#  innodb_locks
]

# Mapping of names to do with innodb_metrics schema
information_schemas_mapping = {
    "innodb_buffer_pool_read_requests": { "schema": "innodb_metrics", "name": "buffer_pool_read_requests"}
#    "innodb_buffer_pool_read_requests": { "schema": "", "name": ""}
}
# innodb_buffer_pool_read_requests is not well updated in MariaDB (bug resolved with the conclusion that the Glob Status Var can NOT be updated as expected!!).https://jira.mariadb.org/browse/MDEV-31309
""" innodb_buffer_pool_read_requests:
    schema: innodb_metrics
    name: buffer_pool_read_requests
innodb_buffer_pool_reads:
    schema: innodb_metrics
    name: buffer_pool_reads
innodb_buffer_pool_write_requests:
    schema: innodb_metrics
    name: buffer_pool_write_requests
innodb_buffer_pool_wait_free:
    schema: innodb_metrics
    name: buffer_pool_wait_free
 """    

# ONLY For test without usage of Hydra configuration but with a hard-coded config
class ADBMSBufferCacheStates(ADBMSBufferCacheStatesLive):
    def __init__(self, perf_target_level=0.98, qpslat_w="01",                 
                 dbhost="db.local", ram_limit: int=8589934592,   # 8589934592=8GB 6442450944=6GB 4294967296=4GB 2147483648=2GB
                 buf_reset_policy="stay",
                 warmups = {'on_start': 10.0, 'on_buf_update': 30.0, 'sigma_latency_toleration': 0.03 }):
        dba = DBAdminMySql(dbhost=dbhost, servername="mariadb", serverversion="11.1.3", dynamicKnobsToDrive=["innodb_buffer_pool_size"],
                           global_status=global_status,
                           information_schemas=information_schemas,
                           information_schemas_mapping=information_schemas_mapping)
        super().__init__(perf_target_level=perf_target_level, 
                         qpslat_w=qpslat_w,                 
                         ram_limit=ram_limit,
                         buf_reset_policy=buf_reset_policy,
                         db_warmup_time=warmups["on_start"],
                         metrics_picker=SysbenchMetricsPicker(warmups=warmups), 
                         dba=dba)
