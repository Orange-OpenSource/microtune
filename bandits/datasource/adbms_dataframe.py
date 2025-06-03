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


import os
import pandas as pd
from pandas import DataFrame
import numpy as np
import xxhash

from bandits.datasource.dataframes.obs_samples_dataframes import ObsSamplesDF 
from bandits.datasource.dataset import ADBMSDataSetEntryContextSelector


class TrainTestDataSets():
    # perf_level: Perf level(s) choosen from ALL_PERF_OBJS = [0.9, 0.95, 0.965, 0.98, 0.99, 0.995, 0.997] # i.e 100ms, 50ms, 35ms, 20ms, 10ms, 5ms, 3ms with MIN=0, and MAX=1000
    # randtypes: Full list is [ "special", "gaussian", "uniform"]. v9 and + have "pareto" in more
    # clients: In v9 [1,2,3,4,5,6,7,8,10,12]. In v8 [ 3,4,6,8,12 ]
    def __init__(self, version=8, testastrain=False, ratio=80, ratio_eval_test=50, 
                 perf_level=0.98, randtypes=[ "gaussian", "uniform" ], clients=[4,6], 
                 pickles_path="./", pickles_prefix="workloads", seed=42, verbose=1):
        self.verbose = verbose
        self.seed = seed
        self.version = version
        self.testastrain = testastrain
        self.ratio = ratio
        self.ratio_eval_test = ratio_eval_test
        assert os.path.isdir(pickles_path), f"Error, invalid path for Dataset pickles: {pickles_path}"
        self.perf_level = perf_level
        self.randtypes = randtypes
        self.clients = clients

        self.filespath = pickles_path
        self.filesprefix = pickles_prefix
        
        obssamples = ObsSamplesDF(version=version)
        self.LATENCY_MIN = obssamples.LATENCY_MIN  # ms
        self.LATENCY_MAX = obssamples.LATENCY_MAX  # ms
        self.QPS_MIN = obssamples.QPS_MIN          # Throughput Queries/s
        self.QPS_MAX = obssamples.QPS_MAX          # Throughput Queries/s
        self.df_train = self.df_eval = self.df_test = None

        # For a right hash signature, take care to put this line at the end of __init__
        self._id_hash = xxhash.xxh3_64_hexdigest(str(dict(vars(self)))) #, seed=self.seed)
        self.obssamples = obssamples
    
    def hashName(self):
        return self.filesprefix+self._id_hash
    
    def load(self, test_only=False, force_reload=False):
        # Try to load prepared DF
        pathprefix = os.path.join(self.filespath, self.hashName())
        self.train_file = f'{pathprefix}_train_v{self.version}-tmp.pickle'
        self.eval_file = f'{pathprefix}_eval_v{self.version}-tmp.pickle'
        self.test_file = f'{pathprefix}_test_v{self.version}-tmp.pickle'
        if not force_reload and os.path.isfile(self.train_file) and os.path.isfile(self.eval_file) and os.path.isfile(self.test_file):
            print("Loading dedicated DF...")
            if not test_only:
                self.df_train = pd.read_pickle(self.train_file)
                self.df_eval = pd.read_pickle(self.eval_file)
            self.df_test = pd.read_pickle(self.test_file)
        else:
            self._loadCompleteDF(testastrain=self.testastrain)
            # Save dedicated DataFrames
            self.df_eval, self.df_test = self.obssamples.spliDF(self.df_test, ratio=self.ratio_eval_test, rnd_seed=self.seed)
            self.df_train.to_pickle(self.train_file)
            self.df_eval.to_pickle(self.eval_file)
            self.df_test.to_pickle(self.test_file)

    def _loadCompleteDF(self, testastrain=False):
        pathprefix = os.path.join(self.filespath, self.filesprefix)
        self._log(f"Load {pathprefix}...")
        df = self.obssamples.mergeTrainTest(pathprefix)

        # Workloads with bad quality data in version 8
        # 22 1.0 3 Un: Un pb sur le CHR (entre autre?) à mi-parcours, valeur 0
        # 22 1.0 3 Sp: Rien de visible dans nos courbes
        # 30 1.0 4 Ga: Wrong QPS @ Buf > 7680MB
        #bad_quality_data = ["22 1.0 3 Un", "22 1.0 3 Sp", "30 1.0 4 Ga"] # v9 [] # v8 ["22 1.0 3 Un", "22 1.0 3 Sp", "30 1.0 4 Ga"]
        #df = df.drop(df[df["combined_column"].isin(bad_quality_data)].index)
        # 5 tables experiences have a very bad cumulated regret ^^ (don't know why...) For example _pool_stats.FREE_BUFFERS is 0...
        #df = df.drop(df[df["tables"] == 5].index)

        ALL_PERF_OBJS = df["perf_target_level"].unique()
        self._log(f'All possible perf levels: {ALL_PERF_OBJS} Choose:{self.perf_level}')

        ALL_RANDTYPES = df["randtype"].unique()
        self._log(f'Workloads data access. All possible rand types distribution: {ALL_RANDTYPES} Choose:{self.randtypes}')

        ALL_CON_CLIENTS = df["wl_clients"].unique()
        self._log(f"All possible client connections count: {ALL_CON_CLIENTS} Choose:{self.clients}")

        # Filter to get a smaller dataset
        df = df[(df["perf_target_level"].isin([ self.perf_level ])) & (df["randtype"].isin(self.randtypes)) & (df["wl_clients"].isin(self.clients))] # & (df["tables"] == 22) & (df["tables_rows"] == 1000000)]

        # Tolerancy over the SLA's max latency
        # i.e. GAP objective margin to compute SLA toleration around a performance objective
        objgap = df["objective_gap"].unique()
        assert len(objgap) == 1, "ERROR: Only 1 objectif gap per dataset can exists. Dataset is corrupted!"
        OBJECTIVE_GAP = objgap[0] 
        self._log(f'Tolerancy over SLA max latency: {OBJECTIVE_GAP}')

        BUF_VALUES_COUNT = df.iloc[0]["buf_values_count"]
        self._log(f'Buffer values count per workload: {BUF_VALUES_COUNT}')

        assert len(df)//BUF_VALUES_COUNT == len(df)/BUF_VALUES_COUNT, f"Error, Buffer values count is {BUF_VALUES_COUNT} and data frame len ({len(df)}) is not a multiple of buffer values count."

        # Train and Test dataframes
        if testastrain:
            self.df_train = self.df_test = df
            wl_list = self.df_train["combined_column"].unique()
            self._log(f'WILL TRAIN and TEST ON {len(wl_list)*len([ self.perf_level ])} WORKLOADS:\n{wl_list}'[:1600]+"...")
        else:
            if self.ratio > 0:
                self.df_train, self.df_test = self.obssamples.spliDF(df, ratio=self.ratio, rnd_seed=self.seed)
            else:
                self.df_train, self.df_test = self.obssamples.spliDFByClients(df)
            wl_list = self.df_train["combined_column"].unique()
            self._log(f'WILL TRAIN ON {len(wl_list)*len([ self.perf_level ])} WORKLOADS:\n{wl_list}'[:1600]+"...")
            self._log(f'All Client Cnx: {self.df_train["wl_clients"].unique()}')

            wl_list = self.df_test["combined_column"].unique()
            self._log(f'MAY TEST ON {len(wl_list)*len([ self.perf_level ])} WORKLOADS:\n{wl_list}'[:800])
            self._log(f'All Client Cnx: {self.df_test["wl_clients"].unique()}')

        self._log(f'Dataset Total Workloads count={len(df)//BUF_VALUES_COUNT}')

        # TODO: save armN and PROBA_ARMN in DF. Right now, information is lost except when a complete load is
        # performed. Dedicated dataframe have'nt the information.
        # Probabilities of arm on the whole dataset, Train AND Test
        self.df_arm0 = df.loc[df["buf_size_idx"] == 0]["ARM0_01"].sum()
        self.df_arm1 = df.loc[df["buf_size_idx"] == 0]["ARM1_01"].sum()
        self.df_arm2 = df.loc[df["buf_size_idx"] == 0]["ARM2_01"].sum()
        df_arm_tot= self.df_arm0 + self.df_arm1 + self.df_arm2

        self.PROBA_ARM0=self.df_arm0/df_arm_tot
        self.PROBA_ARM1=self.df_arm1/df_arm_tot
        self.PROBA_ARM2=self.df_arm2/df_arm_tot
        self._log(f'#All Downs (Train&Test):{self.df_arm0} Prob0:{self.PROBA_ARM0}')
        self._log(f'#All Stay  (Train&Test):{self.df_arm1} Prob1:{self.PROBA_ARM1}')
        self._log(f'#All Up    (Train&Test):{self.df_arm2} Prob2:{self.PROBA_ARM2}')

    #def splitTestsForEval(self, ratio=50):
    #    self.df_eval, self.df_test = self.obssamples.spliDF(self.df_test, ratio=ratio, rnd_seed=self.seed)


    def _log(self, msg):
        if self.verbose > 0:
            print(msg)

    def close(self):
        self.df_train = self.df_eval = self.df_test = None

# A cache allowing to reduce states access in panda's Dataframe
from functools import lru_cache
from sklearn.preprocessing import MinMaxScaler 


# A Dataset is a dataframe composed of groups of data having each the same entries count.
# The Dataset length is equal to the Groups_count*Entries_count_per_group
# This Dataset entry selector has for purpose to allow navigation in DataSet either sequentialy, randomly on by increment
# DataSetEntrySelector determines Group count and Entries_count_per_group thanks to the 2 fields "group_id_field" and "entry_id_field" sorted uniquely and counted.
# The _entry(incr) method allows to get the current dataframe's entry in the dataset or the entry at the current index+incr. It returns None if out of range.
# The entries (i.e. dataframe rows) are cached with a LRU management thanks to the Python's functools.
class ADBMSDataFrameEntrySelector(ADBMSDataSetEntryContextSelector):
    def __init__(self, df: DataFrame, group_id_field="", entry_id_field="", seed=1, context_elems=None, normalize=False, with_scaler:MinMaxScaler=None):
        #self._df = df.reset_index(drop=True) # (RE)Ensure a suitable dataframe's indexing
        self._df = df
        # Determine Groups size and Entries count per group
        group_list = self._df[group_id_field].unique()
        e_list = self._df[entry_id_field].unique()

        super().__init__(group_list=group_list, entries_per_group=len(e_list), seed=seed, context_elems=context_elems, normalize=normalize, with_scaler=with_scaler)

        assert self._total_entries_count == len(self._df), f"Error, Discrepancy in data set. {self._groups_count} group(s) of {group_id_field} with {self._entries_count} entries {entry_id_field} per group do not match dataset length {len(df)}"

    @lru_cache(maxsize=64) # typed=False
    def _cached_entry(self, idx):
        return self._df.iloc[idx]
    
    # Return the entry at the current position+incr or None if the position is out of bounds (out of the current group in dataset)
    def _entry(self, incr=0):
        if incr != 0:
            real_incr = self._select(incr)
            if real_incr != incr:
                return None
        e = self._cached_entry(self.globalIndex())
        if incr != 0:
            self._select(-real_incr)
        return e

    # Return current state (if bufferidx_increment is 0) or the state at an another buffer index value.
    # Return None if the buffer idx resulting to the current buffer+the increment is out of range.
    def state(self, bufferidx_increment=0):
        return  self._entry(incr=-bufferidx_increment) # Buffer indexes are in reverse order   

    def prepareContext(self, context_elems=[], check_data=False, normalize=False, with_scaler:MinMaxScaler=None):
        self._reinit(context_elems, with_scaler)
        
        vectors = np.array([[float(row[x]) for x in self._context_elems] for index, row in self._df.iterrows()])
        assert len(vectors) == self._total_entries_count, f"Samples of OBS and dataframe length mismatched! #Vectors={len(vectors)} / total_entries_count={self._total_entries_count}"

        if normalize:
            vectors = self._normalizeNumpyFloatElems(vectors)

        self._context_vectors = vectors



# A subclass of ADBMSDataFrameEntrySelector that redefine Group into Workload and entry as a state for a specific Buffer Cache Size 
# It provides methods to change group and entry in group either randomly or sequentialy.
# The default following methods are implemented to provide a default behaviour:
#   - reset(group_idx) - Go to the group group_ix and set the default entry randomly (with a uniform distribution) 
#   - next() - Go randomly (with an uniform distribution) to a new entry in the current group. 
#   - move(increment) - Change to a new current entry in the current group from a relative increment to the current position (index) 
# By default, at the instanciation, the current default is the first entry in the dataset (index=0)
# The reset(), move() and next() methods can ve overriden by a sub-class in order to implement a specific behaviour
class ADBMSBufferCacheStates(ADBMSDataFrameEntrySelector):
    def __init__(self, df: DataFrame, group_id_field="", entry_id_field="", qpslat_w="01", seed=1, context_elems=None, normalize=False, with_scaler:MinMaxScaler=None):
        super().__init__(df, group_id_field, entry_id_field, seed, context_elems=context_elems, normalize=normalize, with_scaler=with_scaler)
        self._qpslat_w = qpslat_w
        state = self.state()
        self.latency_range = state["latency_mean_max"] - state["latency_mean_min"]
        if self._qpslat_w == "01":
            self.lat_gap = state["objective_gap"]*self.latency_range # objective_gap, only true for "ilat" or "iperf01"
            self._latgap_ms = str(round(self.lat_gap, 3))+"ms"
            self.lat_target = (1.-state["perf_target_level"])*self.latency_range # True only with "ilat" and "iperf01" AND only 1 performance target per environement
        else:
            self.lat_gap = 0
            self._latgap_ms = "NA"
            self.lat_target = -1
        self._iperf_target = state["perf_target_level"]
        #self._action2incr = np.vectorize(lambda x:-x)    

    def getWorkloadNameId(self):
        state = self.state()
        return state['combined_column']

    #def getEpisodeArmsCount(self):
    #    state = self.state()
    #    return state["ARM0_"+self._qpslat_w], state["ARM1_"+self._qpslat_w], state["ARM2_"+self._qpslat_w]

    def getQPSLatWeigths(self):
        return self._qpslat_w
    
    # Return IPERF, IDELTAPERF, IPERFTARGET
    def getIPerfIndicators(self):
        est = self.state()
        return (est["iperf"+self._qpslat_w], est["delta_perf_target"+self._qpslat_w], self._iperf_target)

    def getLatency(self, filtered=True):
        state = self.state()
        return state["sysbench_filtered.latency_mean"]
    
    # Return Latency gap, Latency gap in ms (str), latency target (threshold)
    def getLatencyGapTarget(self):
        return self.lat_gap, self._latgap_ms, self.lat_target    



##
# Specific implementation of selectors for ADBMS dataset from ObsSamples referential
##

# Gaussian (normal) distribution implemented here are constrainted by the available buffer values around the tipping point (the point where the SLA switch from OK to the VIOLATION state)
# sigma: If 0, only mu is fired. If <0 an automatic computation of sigma is done based on mu value and the total buffer values count in a workload. 
# The automatic mode has for purpose to cover all the entries of the workload with a Gaussian distribution.

# Workload: Random normal selection
# Buff value: Gaussian distribution for both reset method. Next() method use an uniform random distribution.
# sigma_tippping_ratio is typically in [0., 1.0], but If <0, an automatic computation is done.
class ADBMSBufferCacheStatesFullRandomSelector(ADBMSBufferCacheStates):
    def __init__(self, df, seed=42, sla_tipping_field="sla_tipping01", sigma_tipping_ratio=-0.50, qpslat_w="01", context_elems=None, normalize=False, with_scaler:MinMaxScaler=None):
        super().__init__(df, group_id_field="combined_column", entry_id_field="buf_size_idx", qpslat_w=qpslat_w, seed=seed, context_elems=context_elems, normalize=normalize, with_scaler=with_scaler)
        self._sla_tipping_field = sla_tipping_field
        self._sla_tipping = self._entries_count//2
        self.sigma_tr = sigma_tipping_ratio

    # Reset to new current workload
    def reset(self, workload_idx: int = 0):
        self._selectGroupRandomly()

    def next(self):
        state = self._entry()
        self._sla_tipping = state[self._sla_tipping_field] # /!\ Index in term of Entry and NOT Buffer IDX!!
        self._selectEntryGRandomly(self._sla_tipping, self.sigma_tr)
        return self.entry_idx

# Workload: Sequential (from index) round-robin selection
# Buff value: Gaussian distribution for both reset method. Next() method use an uniform random distribution.
# sigma_tippping_ratio is typically in [0., 1.0], but If <0, an automatic computation is done.
class ADBMSBufferCacheStatesRandomBufferSelector(ADBMSBufferCacheStates):
    def __init__(self, df, seed=42, sla_tipping_field="sla_tipping01", sigma_tipping_ratio=0.5, qpslat_w="01", context_elems=None, normalize=False, with_scaler:MinMaxScaler=None):
        super().__init__(df, group_id_field="combined_column", entry_id_field="buf_size_idx", qpslat_w=qpslat_w, seed=seed, context_elems=context_elems, normalize=normalize, with_scaler=with_scaler)
        self._sla_tipping_field = sla_tipping_field
        self._sla_tipping = self._entries_count//2 # Default
        self.sigma_tr = sigma_tipping_ratio

    # Reset to new current workload
    def reset(self, workload_idx: int = 0):
        self._selectGroupFromUnboundIndex(workload_idx)

    def next(self):
        state = self._entry()
        self._sla_tipping = state[self._sla_tipping_field] # /!\ Index in term of Entry and NOT Buffer IDX!!
        self._selectEntryGRandomly(self._sla_tipping, self.sigma_tr)
        return self.entry_idx


class ADBMSBufferCacheStatesSequentialSelector(ADBMSBufferCacheStates):
    def __init__(self, df, qpslat_w="01", topdown=True, context_elems=None, normalize=False, with_scaler:MinMaxScaler=None):
        super().__init__(df, group_id_field="combined_column", entry_id_field="buf_size_idx", qpslat_w=qpslat_w, context_elems=context_elems, normalize=normalize, with_scaler=with_scaler)
        if topdown:
            self._from_max=True
            self._incr = 1
        else:
            self._from_max=False
            self._incr = -1

    # Reset to new current workload
    def reset(self, workload_idx: int = 0):
        self._selectGroupFromUnboundIndex(workload_idx)

    def next(self):
        self._selectEntryMinMax(set_to_min=self._from_max)
        return self.entry_idx

class ADBMSBufferCacheStatesFullSequentialSelector(ADBMSBufferCacheStates):
    def __init__(self, df, qpslat_w="01", context_elems=None, normalize=False, with_scaler:MinMaxScaler=None):
        super().__init__(df, group_id_field="combined_column", entry_id_field="buf_size_idx", qpslat_w=qpslat_w, context_elems=context_elems, normalize=normalize, with_scaler=with_scaler)
        self._selectGroupFromUnboundIndex(0)
        self.idx = self.globalIndex() -1

    # Reset to new current workload
    def reset(self, workload_idx: int = 0):
        self.idx = (self.idx+1)%self.getTotalStatesCount()
        self.group_idx = self.idx//self.getStatesCountPerWorkload()
        self.entry_idx = self.idx - (self.group_idx * self.getStatesCountPerWorkload())

    def next(self):
        return self.entry_idx


class ADBMSBufferCacheStatesRandomTopDownSelector(ADBMSBufferCacheStates):
    def __init__(self, df, seed=42, qpslat_w="01", context_elems=None, normalize=False, with_scaler:MinMaxScaler=None):
        super().__init__(df, group_id_field="combined_column", entry_id_field="buf_size_idx", qpslat_w=qpslat_w, seed=seed, context_elems=context_elems, normalize=normalize, with_scaler=with_scaler)
        self._from_max = False

    # Reset to new current workload
    def reset(self, workload_idx: int = 0):
        topdown = self.rg_group.choice(a=2, size=1)
        if topdown == 0:
            self._from_max=True
        else:
            self._from_max=False

        self._selectGroupFromUnboundIndex(workload_idx)

    def next(self):
        self._selectEntryMinMax(set_to_min=self._from_max)
        return self.entry_idx

class ADBMSBufferCacheStatesTopDownRockerSelector(ADBMSBufferCacheStates):
    def __init__(self, df, qpslat_w="01", context_elems=None, normalize=False, with_scaler:MinMaxScaler=None):
        super().__init__(df, group_id_field="combined_column", entry_id_field="buf_size_idx", qpslat_w=qpslat_w, context_elems=context_elems, normalize=normalize, with_scaler=with_scaler)
        self._incr = 0
        self._from_max = False

    # Reset to a new current workload every 2 workload_idx
    def reset(self, workload_idx: int = 0):
        self._from_max = not self._from_max
        self._selectGroupFromUnboundIndex(workload_idx//2)

    # Alternatively go to the first (min) or last (max) entry index (i.e. Buffer Max or Min)
    def next(self):
        self._selectEntryMinMax(set_to_min=self._from_max)
        return self.entry_idx
