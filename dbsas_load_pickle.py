
# DBSAS preparation data tools
# Use original dataset to add missing columns to DBSAS generated dataset

import os
from bandits.datasource.dataframes.obs_samples_dataframes import ObsSamplesDF

def run(version="13", ratio_traintest=80., dbsas_dataset, orig_dataset):
    obss = ObsSamplesDF(version=version)

    dfdbsas = obss.loadFromPickle(dbsas_dataset)

    clist = dfdbsas.columns.values.tolist()
    print(f"DFDBSAS #col:{len(clist)} #lines:{dfdbsas.shape[0]}, {clist}")

    dforig = obss.loadFromPickle(orig_dataset)
    clist = dforig.columns.values.tolist()
    print(f"DFORIG #col:{len(clist)} #lines:{dforig.shape[0]}, {clist}")
    #df_train, df_test = obss.spliDF(dforig)
    #obss.saveTrainTests("workloads_orig", df_train, df_test)

    print(f"DFORIG pef_target_level: {dforig['perf_target_level'].unique().tolist()}")

    cols2add = ['combined_column', 'tables', 'tables_rows', 'randtype', 'observation.innodb_buffer_pool_size', 'observation.normalized_buf_size', 'extra_info.sysbench.statements_mean']
    dfdbsas[cols2add] = dforig[cols2add]
    #dfdbsas['objective_margin'] = 0
    #print(dfdbsas['observation.normalized_buf_size'].unique().tolist())

    obss.PERF_OBJS = [0.98]
    dfdbsas = obss.fixColumns(dfdbsas, combined_col=False)
    print(f"DFDBSAS pef_target_level: {dfdbsas['perf_target_level'].unique().tolist()}")
    objgap = dfdbsas["objective_gap"].unique().tolist()
    print(f"DFDBSAS objectif_gap: {objgap}")

    df_train, df_test = obss.spliDF(dfdbsas, ratio=ratio_traintest)
    workloads =  df_train['combined_column'].unique().tolist()
    print(f"DFDBSAS TRAIN #workloads:{len(workloads)}")
    workloads =  df_test['combined_column'].unique().tolist()
    print(f"DFDBSAS TEST NOT SAVED /!\ #workloads:{len(workloads)}")
    obss.saveToPickle("./workloads_dbsas_train_"+version, df_train)

    df_train, df_test = obss.spliDF(dforig, ratio=ratio_traintest)
    workloads =  df_train['combined_column'].unique().tolist()
    print(f"DFORIG TRAIN #workloads:{len(workloads)}")
    workloads =  df_test['combined_column'].unique().tolist()
    print(f"DFDBSAS&ORIG TEST #workloads:{len(workloads)}")
    obss.saveTrainTests("workloads_orig", df_train, df_test)

    try:
        os.symlink("./workloads_orig_test_"+version+".pickle", "./workloads_dbsas_test.pickle")
        print("Link created successfully")
    except FileExistsError:
        print("Symlink already exists.")
    except PermissionError:
        print("Permission denied: You might need admin rights.")

run(dbsas_dataset="./db_fullnetwork_generated_full_dataset_discrete", orig_dataset="./original_dataset")
exit(0)

