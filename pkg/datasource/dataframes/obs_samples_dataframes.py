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
    
    def retrieve_all_documents(self, mongohost="localhost", mongoport=27017, dbname="adbms-obs-ref-mariadb-11_1_3", colname="sysbench__oltp_read_write__"):
        # Connect to MongoDB (i.e. 192.168.0.206)
        client = MongoClient(f"mongodb://{mongohost}:{mongoport}/")
        print(f'Connected to {mongohost}:{mongoport}')

        # Access the database and collection
        db = client[dbname]
        collection = db[colname+self._version]  # Replace "your_collection_name" with the actual collection name

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

    # By default, Add new columns ["OP_BUDGET01", "OP_USLA01", "OP_CRAM01", "OP_OB_USLA01", "OP_OB_CRAM01"] to a performance based on 0% QPS objective and 100% Latency objective
    # With weigths_name="19" Add new columns ["OP_BUDGET19", "OP_USLA19", "OP_CRAM19", "OP_OB_USLA19", "OP_OB_CRAM19"] to a performance based on 10% QPS objective and 90% Latency objective
    def _add_optimal_policy_colums(self, df, buf_sizes_list, weigths_name="01"):
        buf_sizes_list_len = len(buf_sizes_list)
        buf_mb = np.array(buf_sizes_list, int)//1024//1024

        def compute_OP_BUDGET(sla_tipping: int, buf_size_idx: int):
            sla_tipping = max(1, sla_tipping)  # Ensure at least 1, sla_tipping is set to 0 when there is no tipping point because RAM max is never enough to reach the SLA
            buf_idx = buf_sizes_list_len - buf_size_idx
            start = min(sla_tipping, buf_idx) -1
            end = max(sla_tipping, buf_idx)
            res = end - start -1
            return res, start, end

        def compute_OP_CRAM(start: int, end: int):
            #print(f"compute_OP_CRAM: sla_tipping={sla_tipping}, buf_size_idx={buf_size_idx} => buf_idx={buf_idx}, idx1={idx1}, idx2={idx2}")
            if end == (start+1):
                cram = buf_mb[start] *2  # Case of STAY arm, thus cumulate RAM at T-1 + RAM at T
            else:
                cram = buf_mb[start:end].sum()
            return cram

        def compute_OP_USLA(sla_tipping: int, buf_size_idx: int):
            res = buf_sizes_list_len - sla_tipping - buf_size_idx
            return max(0, res)

        def compute_OP_OVER_BUDGET_CRAM(sla_tipping):
            buf_idx = sla_tipping -1 if sla_tipping>0 else 0
            return buf_mb[buf_idx]

        def compute_OP_OVER_BUDGET_USLA(sla_tipping):
            # sla_tipping is set to 0 when there is no tipping point because RAM max is never enough to reach the SLA
            usla4overbudget = 1 if sla_tipping == 0 else 0
            return usla4overbudget

        def computeOptimalPolicyPerformance(row, columns):
            sla_tipping = int(row["sla_tipping"+weigths_name])
            buf_size_idx = int(row["buf_size_idx"])

            op_budget, start, end = compute_OP_BUDGET(sla_tipping, buf_size_idx)
            op_usla = compute_OP_USLA(sla_tipping, buf_size_idx)
            op_cram = compute_OP_CRAM(start, end)
            cram4overbudget = compute_OP_OVER_BUDGET_CRAM(sla_tipping)
            usla4overbudget = compute_OP_OVER_BUDGET_USLA(sla_tipping)
            res = (op_budget, op_usla, op_cram, usla4overbudget, cram4overbudget)
            assert len(columns) == len(res), f"computeOptimalPolicyPerformance: columns len {len(columns)} != res len {len(res)}"
            
            return pd.Series(res, index=columns)

        new_columns = ["OP_BUDGET"+weigths_name, "OP_USLA"+weigths_name, "OP_CRAM"+weigths_name, "OP_OB_USLA"+weigths_name, "OP_OB_CRAM"+weigths_name]
        print(f"Adding Optimal Policy Performance columns {new_columns} for ALL workloads with {buf_sizes_list_len} buffer sizes...")
        df[new_columns] = df.apply(lambda row: computeOptimalPolicyPerformance(row, new_columns), axis=1) #, result_type='expand' )


    # Concatenates N times the dataframe with the added column 'perf_target_level' at different values between 0 to 100%
    # Weigth Desc is [Weight QPS, Weigth LAT, Weigth name]. Examples: [0, 1, "01"], or [0.1, 0.9, "19"]
    def add_perf_target_level(self, df, objective_margin, weigths_desc=[0, 1, "01"]):
        workloads =  df['combined_column'].unique().tolist()

        buf_sizes_list = df["buf_size"].unique().tolist()
        buf_values_count = len(buf_sizes_list)

        with pd.option_context("mode.copy_on_write", True):
            df_out = pd.DataFrame()
            
            print(f"Adding performance target levels columns for {len(workloads)} workloads and {len(self.PERF_OBJS)} performance objectives...")
            for lvl in self.PERF_OBJS:
                df['perf_target_level'] = lvl
                objective_gap = abs((lvl-1)*objective_margin)
                df["objective_gap"] = round(objective_gap,4)

                df["latency_threshold"] = df.apply(lambda row: (1.-row["perf_target_level"])*(row["latency_mean_max"] - row["latency_mean_min"]), axis=1)

                weigth_qps = weigths_desc[0]
                weigth_lat = weigths_desc[1]
                weigths = weigths_desc[2]

                df["iperf"+weigths] = df.apply(lambda row: (row["iqps"]*weigth_qps+row["ilat"]*weigth_lat), axis=1)
                df["delta_perf_target"+weigths] = df.apply(lambda row: (row["iperf"+weigths] - row["perf_target_level"]), axis=1)

                for wl in workloads:
                    wldf = df[df['combined_column'] == wl]
                    arm0_count = wldf[wldf["delta_perf_target"+weigths] > objective_gap]["buf_size"].count()     # Down
                    arm1_count = wldf[(wldf["delta_perf_target"+weigths] >= 0.) & (wldf["delta_perf_target"+weigths] <= objective_gap)]["buf_size"].count()     # Stay
                    arm2_count = wldf[wldf["delta_perf_target"+weigths] < 0.]["buf_size"].count()     # Up
                    if arm0_count> 0 and arm1_count == 0 and arm2_count >= 0:
                        arm1_count = 1  # Stay
                        arm0_count -= 1 # Down
                    elif arm1_count > self.MAX_STAY_PER_WORKLOAD:
                        arm0_count += arm1_count -self.MAX_STAY_PER_WORKLOAD
                        arm1_count = self.MAX_STAY_PER_WORKLOAD
                    
                    idx0 = wldf.index[wldf['combined_column'] == wl][0] # Dummy selection on column 'combined_column', but necessary...
                    idxp = wldf.index[wldf["delta_perf_target"+weigths] <0].min() # Use min() in case where there are multiple tipping points, choose the one with the higher buffer value
                    # No tipping point (always OVER)? Take last index (smallest buffer size) 
                    if idxp is np.NaN:
                        idxp=idx0 + buf_values_count
                    df.loc[df['combined_column'] == wl, 'sla_tipping'+weigths] = int(idxp-idx0)
                    #wldf['sla_tipping'+weigths] = int(idxp-idx0)

                    #wldf['ARM0_'+weigths] = arm0_count
                    #wldf['ARM1_'+weigths] = arm1_count
                    #wldf['ARM2_'+weigths] = arm2_count
                    # Slower version without copy_on_write mode
                    df.loc[df['combined_column'] == wl, 'ARM0_'+weigths] = arm0_count
                    df.loc[df['combined_column'] == wl, 'ARM1_'+weigths] = arm1_count
                    df.loc[df['combined_column'] == wl, 'ARM2_'+weigths] = arm2_count

                self._add_optimal_policy_colums(df, buf_sizes_list, weigths) # Apply to all workloads
                
                df_out = pd.concat([df_out, df.copy()], ignore_index=True)
        
        return df_out

    def additionalColumns(self, df, combined_col=True, combine_with_origin=False):
        df["db_size_mb"] = df["db_size_mb"].astype(int)
        buf_sizes_list = df["buf_size"].unique().tolist()
        df["buf_size_min_mb"] = buf_sizes_list[-1]//1024//1024
        buf_values_count = len(buf_sizes_list)
        df["buf_values_count"] = buf_values_count
        df["buf_size_idx"] = df.apply(lambda row: int((row["buf_size"]/buf_sizes_list[0])*buf_values_count)-1, axis=1)
        df["tables_rows_M"] = df.apply(lambda row: round(row["tables_rows"]/1000000,1), axis=1)
        df["Rtype"] = df.apply(lambda row: row["randtype"][:2].capitalize(), axis=1)

        # Label all the workloads types
        if combined_col:
            columns_to_combine = ["tables", "tables_rows_M", "wl_clients", "Rtype"]
            if combine_with_origin:
                columns_to_combine = ["origin"]+columns_to_combine
            # Create a new column with the combined values as strings
            df['combined_column'] = df[columns_to_combine].astype(str).agg(' '.join, axis=1)
            df["combined_column"] = df.apply(lambda row: f'V{row["combined_column"]}', axis=1)

        return df

    def fixColumns(self, df, objective_margin=0.3, combined_col=True):
        if combined_col:
            # Retrieve version from combined_column if present to assign origin column else keep value in place if 'origin' exists else assign with current version of this class
            version = df['combined_column'].str.split().str[0]
            condition = version.str.startswith('V')
            df['origin'] = np.where(condition, version.str[1:], df.get('origin', self._version))

        df = self.additionalColumns(df, combined_col=combined_col, combine_with_origin=True)

        # Add performance's min, max columns
        df["latency_mean_min"] = self.LATENCY_MIN #0.0001  
        df["latency_mean_max"] = self.LATENCY_MAX #1000.
        df["qps_mean_min"] = self.QPS_MIN #1.
        df["qps_mean_max"] = self.QPS_MAX #100000.

        df["iqps"] = df.apply(lambda row: (row["extra_info.sysbench.statements_mean"]-row["qps_mean_min"])/(row["qps_mean_max"]-row["qps_mean_min"]), axis=1)
        # Disable capped value of Max latency value
        #df["ilat"] = df.apply(lambda row: (min(row["sysbench_filtered.latency_mean"], self.LATENCY_MAX)-row["latency_mean_max"])/(row["latency_mean_min"]-row["latency_mean_max"]), axis=1) # ms. /!\ MIN MAX INVERTED to reflect the improvement when the latency decreases
        df["ilat"] = df.apply(lambda row: (row["sysbench_filtered.latency_mean"]-row["latency_mean_max"])/(row["latency_mean_min"]-row["latency_mean_max"]), axis=1) # ms. /!\ MIN MAX INVERTED to reflect the improvement when the latency decreases
        
        df = self.add_perf_target_level(df, objective_margin=objective_margin)

        return df.reset_index(drop=True, inplace=False)

    # Import data from MongoDB (DataRef Obs Samples), fix/add some columns and return the full indexed dataframe
    def getMongoDataRef(self,  mongohost="localhost", mongoport=27017, dbname="adbms-obs-ref-mariadb-11_1_3", colname="sysbench__oltp_read_write__" ,objective_margin=0.1):
        list_documents, list_documents0 = self.retrieve_all_documents(mongohost=mongohost, mongoport=mongoport, dbname=dbname, colname=colname)
        df_filtered = self.docs2flat_list(list_documents)
        # Fixes and add some columns with constants
        df = self.additionalColumns(df_filtered)
        return self.fixColumns(df, objective_margin=objective_margin, combined_col=False)

    def saveToPickle(self, fullname, df):
        df.to_pickle(fullname+'.pickle', compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1})
        return fullname+'.pickle'


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
    
    def saveFullPickle(self, prefix, df):
        return self.saveToPickle(prefix+"_full_"+self._version, df)

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

