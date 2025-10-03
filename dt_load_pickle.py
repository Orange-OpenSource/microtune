
# Digital Twin data tools

from bandits.datasource.dataframes.obs_samples_dataframes import ObsSamplesDF

obss = ObsSamplesDF(version="13")

df = obss.loadFromPickle("./db_fullnetwork_generated_full_dataset_discrete")

clist = df.columns.values.tolist()
print(f"DF #col:{len(clist)} #lines:{df.shape[0]}, {clist}")

dforig = obss.loadFromPickle("./original_dataset")
clist = dforig.columns.values.tolist()
print(f"DFORIG #col:{len(clist)} #lines:{dforig.shape[0]}, {clist}")
df_train, df_test = obss.spliDF(dforig)
obss.saveTrainTests("workloads", df_train, df_test)

cols2add = ['combined_column', 'tables', 'tables_rows', 'randtype', 'observation.innodb_buffer_pool_size', 'observation.normalized_buf_size', 'extra_info.sysbench.statements_mean']
#print(df['observation.normalized_buf_size'].unique().tolist())

df[cols2add] = dforig[cols2add]
#df['objective_margin'] = 0
#print(df['observation.normalized_buf_size'].unique().tolist())

df = obss.fixColumns(df, combined_col=False)

workloads =  dforig['combined_column'].unique().tolist()
print(f"DFORIG #workloads:{len(workloads)} {workloads}")

workloads =  df['combined_column'].unique().tolist()
print(f"DF #workloads:{len(workloads)} {workloads}")

df_train, df_test = obss.spliDF(df)

obss.saveTrainTests("workloads_dbsas", df_train, df_test)

exit(0)

