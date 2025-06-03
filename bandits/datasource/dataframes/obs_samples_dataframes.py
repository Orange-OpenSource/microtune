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
from pymongo import MongoClient
import pandas as pd
import numpy as np
import tensorflow as tf
import random



class ObsSamplesDF():
    def __init__(self, version="9"):
        self._version = str(version)
        self.LATENCY_MIN = 0.0001    #ms
        self.LATENCY_MAX = 1000    #ms
        self.QPS_MIN = 1        # Throughput Queries/s
        self.QPS_MAX = 100000        # Throughput Queries/s
        self.PERF_OBJS = [0.9, 0.95, 0.965, 0.98, 0.99, 0.995, 0.997] # i.e 100ms, 50ms, 35ms, 20ms, 10ms, 5ms, 3ms with MIN=0, and MAX=1000
        self.MAX_STAY_PER_WORKLOAD = 3

    def getPossiblePerfObjectives(self):
        return self.PERF_OBJS
    
    def retrieve_all_documents(self):
        # Connect to MongoDB
        client = MongoClient("mongodb://192.168.0.206:27017/")

        # Access the database and collection
        db = client["adbms-obs-ref-mariadb-11_1_3"]
        collection = db["sysbench__oltp_read_write__"+self._version]  # Replace "your_collection_name" with the actual collection name

        # Retrieve all documents in the collection
        documents = collection.find()
        list_documents = []
        list_documents0 = []
        
        # Print the documents
        for document in documents:
            if document["buf_size"] == 0:
                list_documents0.append(document)
            else:
                list_documents.append(document)

        # Close the connection
        client.close()
        return list_documents, list_documents0

    def docs2flat_list(self, list_documents):
        flat_list = pd.json_normalize(list_documents)

        df = pd.DataFrame(flat_list)
        null_rates = df.isnull().mean()
        # Set the threshold for null rate (e.g., 80%)
        threshold = 0.8
        # Filter columns based on null rates, in this case, remove all the colums that are produced by null meme indicators
        columns_to_drop = null_rates[null_rates > threshold].index
        # Drop columns with null rates higher than the threshold
        df.drop('_id', axis=1, inplace=True)
        df.drop('reward', axis=1, inplace=True)
        df.drop('now', axis=1, inplace=True)
        df_filtered = df.drop(columns=columns_to_drop)
        # Drop all the rows with null innodb_buffer_pool_size
        df_filtered_drop_0_mem = df_filtered.dropna(subset=["observation.innodb_buffer_pool_size"])

        return df_filtered_drop_0_mem

    # Concatenates N times the dataframe with the added column 'perf_target_level' at different values between 0 to 100%
    def add_perf_target_level(self, df, objective_margin, buf_values_count):
        weigths_desc = [[0, 1, "01"], [0.1, 0.9, "19"]] # Weight WPS, Weigth LAT, Weigth name
        workloads =  df['combined_column'].unique().tolist()

        with pd.option_context("mode.copy_on_write", True):
            df_out = pd.DataFrame()
            
            for lvl in self.PERF_OBJS:
                df['perf_target_level'] = lvl
                objective_gap = abs((lvl-1)*objective_margin)
                df["objective_gap"] = objective_gap

                df["latency_threshold"] = df.apply(lambda row: (1.-row["perf_target_level"])*(row["latency_mean_max"] - row["latency_mean_min"]), axis=1)

                for weigth in weigths_desc:
                    wqps = weigth[0]
                    wlat = weigth[1]
                    wname = weigth[2]

                    df["iperf"+wname] = df.apply(lambda row: (row["iqps"]*wqps+row["ilat"]*wlat), axis=1)
                    df["delta_perf_target"+wname] = df.apply(lambda row: (row["iperf"+wname] - row["perf_target_level"]), axis=1)

                    for wl in workloads:
                        wldf = df[df['combined_column'] == wl]
                        arm0_count = wldf[wldf["delta_perf_target"+wname] > objective_gap]["buf_size"].count()     # Down
                        arm1_count = wldf[(wldf["delta_perf_target"+wname] >= 0.) & (wldf["delta_perf_target"+wname] <= objective_gap)]["buf_size"].count()     # Stay
                        arm2_count = wldf[wldf["delta_perf_target"+wname] < 0.]["buf_size"].count()     # Up
                        if arm0_count> 0 and arm1_count == 0 and arm2_count >= 0:
                            arm1_count = 1  # Stay
                            arm0_count -= 1 # Down
                        elif arm1_count > self.MAX_STAY_PER_WORKLOAD:
                            arm0_count += arm1_count -self.MAX_STAY_PER_WORKLOAD
                            arm1_count = self.MAX_STAY_PER_WORKLOAD
                        
                        idx0 = wldf.index[wldf['combined_column'] == wl][0] # Dummy selection on column but necessary...
                        idxp = wldf.index[wldf["delta_perf_target"+wname] <0].min()
                        # No tipping point (always OVER)? Take last index (smallest buffer size) 
                        if idxp is np.NaN:
                            idxp=idx0 + buf_values_count -1
                        df.loc[df['combined_column'] == wl, 'sla_tipping'+wname] = int(idxp-idx0)


                        df.loc[df['combined_column'] == wl, 'ARM0_'+wname] = arm0_count
                        df.loc[df['combined_column'] == wl, 'ARM1_'+wname] = arm1_count
                        df.loc[df['combined_column'] == wl, 'ARM2_'+wname] = arm2_count
                
                df_out = pd.concat([df_out, df.copy()], ignore_index=True)
        
        return df_out

    def getSimuData(self, objective_margin=0.3):
        list_documents, list_documents0 = self.retrieve_all_documents()
        df_filtered = self.docs2flat_list(list_documents)
        # Fixes ad some add columns with constants
        df_filtered["db_size_mb"] = df_filtered["db_size_mb"].astype(int)
        buf_sizes_list = df_filtered["buf_size"].unique().tolist()
        df_filtered["buf_size_min_mb"] = buf_sizes_list[-1]//1024//1024
        buf_values_count = len(buf_sizes_list)
        df_filtered["buf_values_count"] = buf_values_count
        df_filtered["buf_size_idx"] = df_filtered.apply(lambda row: int((row["buf_size"]/buf_sizes_list[0])*buf_values_count)-1, axis=1)
        df_filtered["tables_rows_M"] = df_filtered.apply(lambda row: round(row["tables_rows"]/1000000,1), axis=1)
        df_filtered["Rtype"] = df_filtered.apply(lambda row: row["randtype"][:2].capitalize(), axis=1)
        #bidx = df_filtered["buf_size_idx"].unique().tolist()
        #print("BUFS IDX", bidx, len(bidx))

        # get all the workloads
        columns_to_combine = ["tables", "tables_rows_M", "wl_clients", "Rtype"]

        # Create a new column with the combined values as strings
        df_filtered['combined_column'] = df_filtered[columns_to_combine].astype(str).agg(' '.join, axis=1)

        # Print the DataFrame with the new combined column
        workloads =  df_filtered['combined_column'].unique().tolist()
        print("ALL workloads type:", workloads)

        # Add performance's min, max columns
        df_filtered["latency_mean_min"] = self.LATENCY_MIN #5.  
        df_filtered["latency_mean_max"] = self.LATENCY_MAX #900.
        #df_filtered["ilat"] = 0.
        df_filtered["qps_mean_min"] = self.QPS_MIN #50.
        df_filtered["qps_mean_max"] = self.QPS_MAX #40000.
        #df_filtered["iqps"] = 0.    
        df_filtered["iqps"] = df_filtered.apply(lambda row: (row["extra_info.sysbench.statements_mean"]-row["qps_mean_min"])/(row["qps_mean_max"]-row["qps_mean_min"]), axis=1)
        df_filtered["ilat"] = df_filtered.apply(lambda row: (min(row["sysbench_filtered.latency_mean"], self.LATENCY_MAX)-row["latency_mean_max"])/(row["latency_mean_min"]-row["latency_mean_max"]), axis=1) # ms. /!\ MIN MAX INVERTED to reflect the improvement when the latency decreases
        
        df_filtered = self.add_perf_target_level(df_filtered, objective_margin=objective_margin, buf_values_count=buf_values_count)

        return df_filtered.reset_index(drop=True, inplace=False)

    def saveToPickle(self, fullname, df):
        df.to_pickle(fullname+'.pickle', compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1})


    def loadFromPickle(self, fullname):
        try:
            return pd.read_pickle(fullname+'.pickle', compression={'method': 'gzip'})
        except Exception:
            return pd.read_pickle(fullname+'.pickle')
        

    
    def spliDFByClients(self, df):
        clients =  df['wl_clients'].unique().tolist()
        #clients.remove(0)

        odd_clients = [num for num in clients if num % 2 == 1]
        even_clients = [num for num in clients if num % 2 == 0]

        if len(odd_clients) > len(even_clients):
            train_wl = odd_clients
            test_wl = even_clients
        else:
            train_wl = even_clients
            test_wl = odd_clients

        print(f"Separate train/test datasets by clients Train:{train_wl} test:{test_wl}")
        df1 = df[df['wl_clients'].isin(train_wl)]
        df2 = df[~df['wl_clients'].isin(train_wl)]

        assert len(df) == (len(df1)+len(df2)), f"Missing Train (len={len(df1)}) or Test (len={len(df2)}) workloads after attribution. Total len={len(df)}..."

        # VERY IMPORTANT: Will help to retrieve previous and next buf values from a current state
        return df1.reset_index(drop=True, inplace=False), df2.reset_index(drop=True, inplace=False)
    
    def spliDF(self, df, ratio=80, rnd_seed=42):
        workloads =  df['combined_column'].unique().tolist()
        # trainning set percentage, seprated by workloads
        num_elements_to_select = int(len(workloads) * (ratio / 100))

        random.seed(rnd_seed)
        selected_elements = random.sample(workloads, num_elements_to_select)

        df1 = df[df['combined_column'].isin(selected_elements)]
        df2 = df[~df['combined_column'].isin(selected_elements)]

        assert len(df) == (len(df1)+len(df2)), f"Missing Train (len={len(df1)}) or Test (len={len(df2)}) workloads after attribution. Total len={len(df)}..."

        # VERY IMPORTANT: Will help to retrieve previous and next buf values from a current state
        return df1.reset_index(drop=True, inplace=False), df2.reset_index(drop=True, inplace=False)
    
    def saveTrainTests(self, name, df_train, df_test):
        self.saveToPickle(name+"_train_"+self._version, df_train)
        self.saveToPickle(name+"_test_"+self._version, df_test)

    def loadTrainTests(self, name):
        df_train = self.loadFromPickle(name+'_train_'+self._version)
        df_test = self.loadFromPickle(name+'_test_'+self._version)
        return df_train, df_test
    
    def mergeTrainTest(self, name):
        df_train, df_test = self.loadTrainTests(name)
        return pd.concat([df_train, df_test], ignore_index=True)

    def mergeVersionsToNew(self, name, vother, vnew):
        pdvcur = self.mergeTrainTest(name)
        pdvcur["origin"] = self._version
        
        self._version = vother
        pdvother = self.mergeTrainTest(name)
        pdvother["origin"] = self._version
        self._version = vnew

        pdnew =  pd.concat([pdvcur, pdvother], ignore_index=True)
        pdnew["combined_column"] = pdnew.apply(lambda row: f'V{row["origin"]} {row["combined_column"]}', axis=1)
        return pdnew

