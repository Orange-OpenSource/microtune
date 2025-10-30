from pkg.datasource.dataframes.obs_samples_dataframes import ObsSamplesDF

version="13sg"
orig_hash="9afa1340f5049dfa"
dbsas_hash="c459cde4f8d8cca9"
cvae_hash="c505c0cf3dec3be4"

obss = ObsSamplesDF(version=version)

for phase in ["train", "eval", "test"]:

    dfo = obss.loadFromPickle(f"./workloads_orig{orig_hash}_{phase}_v{version}-tmp")
    dfd = obss.loadFromPickle(f"./workloads_dbsas{dbsas_hash}_{phase}_v{version}-tmp")
    dfc = obss.loadFromPickle(f"./workloads_cvae{cvae_hash}_{phase}_v{version}-tmp")

    ALL_PERF_OBJS_GAP = dfo["objective_gap"].unique()
    print(f"DFO Phase {phase} has performance objectives GAPs: {ALL_PERF_OBJS_GAP}")
    ALL_PERF_OBJS_GAP = dfd["objective_gap"].unique()
    print(f"DFD Phase {phase} has performance objectives GAPs: {ALL_PERF_OBJS_GAP}")
    ALL_PERF_OBJS_GAP = dfc["objective_gap"].unique()
    print(f"DFC Phase {phase} has performance objectives GAPs: {ALL_PERF_OBJS_GAP}")

    print(f'Compare {phase} datasets ORIG vs DBSAS v{version}. DF equals: {dfo.equals(dfd)}')
    print(f'Compare {phase} datasets ORIG vs DBSAS v{version}. combined_colum equals: {dfo["combined_column"].equals(dfd["combined_column"])}')

    print(f'Compare {phase} datasets ORIG vs CVAE v{version}. DF equals: {dfo.equals(dfc)}')
    print(f'Compare {phase} datasets ORIG vs CVAE v{version}. combined_colum equals: {dfo["combined_column"].equals(dfc["combined_column"])}')

