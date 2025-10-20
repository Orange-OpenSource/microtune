from pkg.datasource.dataframes.obs_samples_dataframes import ObsSamplesDF

version="13"
orig_hash="1665f184bd78f0d3"
dbsas_hash="d9ca26ff28490f6e"

obss = ObsSamplesDF(version=version)

for phase in ["train", "eval", "test"]:

    dfo = obss.loadFromPickle(f"./workloads_orig{orig_hash}_{phase}_v{version}-tmp")
    dfd = obss.loadFromPickle(f"./workloads_dbsas{dbsas_hash}_{phase}_v{version}-tmp")

    print(f'Compare {phase} datasets ORIG vs DBSAS v{version}. DF equals: {dfo.equals(dfd)}')
    print(f'Compare {phase} datasets ORIG vs DBSAS v{version}. combined_colum equals: {dfo["combined_column"].equals(dfd["combined_column"])}')


