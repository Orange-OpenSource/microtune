
# DBSAS preparation data tools
# Use original dataset to add missing columns to DBSAS generated dataset

import os
import logging
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import hydrauti as hu
from pkg.datasource.dataframes.obs_samples_dataframes import ObsSamplesDF


def gen_full_datasets(version="13", dbsas_dataset="./db_fullnetwork_generated_full_dataset_discrete", orig_dataset="./original_dataset"):
    obss = ObsSamplesDF(version=version)

    print(f"Loading original dataset from {orig_dataset}...")
    dforig = obss.loadFromPickle(orig_dataset)
    dforig_perf_target_level = dforig['perf_target_level'].unique().tolist()
    print(f"DFORIG pef_target_level: {dforig_perf_target_level}")
    print(f"DFORIG #col:{len(dforig.columns.values.tolist())} #lines:{dforig.shape[0]}")
    workloads = dforig['combined_column'].unique().tolist()
    print(f"DFORIG #workloads:{len(workloads)}")
    obss.saveToPickle("./workloads_orig_full_"+version, dforig)
    log.info(f"DBORIG FULL saved.")

    dfdbsas = obss.loadFromPickle(dbsas_dataset)
    cols2add = ['combined_column', 'tables', 'tables_rows', 'randtype', 'observation.innodb_buffer_pool_size', 'observation.normalized_buf_size', 'extra_info.sysbench.statements_mean']
    dfdbsas[cols2add] = dforig[cols2add]
    #dfdbsas['objective_margin'] = 0
    #print(dfdbsas['observation.normalized_buf_size'].unique().tolist())

    obss.PERF_OBJS = dforig_perf_target_level
    log.info(f"DBSAS fix columns...")
    dfdbsas = obss.fixColumns(dfdbsas, combined_col=False)

    print(f"DFDBSAS pef_target_level: {dfdbsas['perf_target_level'].unique().tolist()}")
    print(f"DFDBSAS #col:{len(dfdbsas.columns.values.tolist())} #lines:{dfdbsas.shape[0]}")
    workloads = dfdbsas['combined_column'].unique().tolist()
    print(f"DFDBSAS #workloads:{len(workloads)}")
    objgap = dfdbsas["objective_gap"].unique().tolist()
    print(f"DFDBSAS objectif_gap: {objgap}")
    obss.saveToPickle("./workloads_dbsas_full_"+version, dfdbsas)
    log.info(f"DFDBSAS FULL saved.")

    cache_files = f'workloads_*v{version}-tmp.pickle workloads_*v{version}-tmp.pickle.bak'
    log.info(f"Cleaning up {cache_files}...")
    os.system(f'rm {cache_files}')





# A logger for this file
log = logging.getLogger(__name__)
#lock = Lock()


#from typing import Tuple

@hydra.main(version_base=None, config_path="configs", config_name="dbsas")
def run(cfg: DictConfig) -> float: #Tuple[float, float]:
    # Performs only 1 trial
    if not hu.prepare_run(cfg):
        return np.inf

    print(cfg.version)
    log.info(cfg.info)
    log.info(f"Run DBSAS data preparation version {cfg.version}.{cfg.version_minor}...")
    gen_full_datasets(version=str(cfg.version), dbsas_dataset=f"./{cfg.dataset_dbsas_import_file}", orig_dataset=f"./{cfg.dataset_orig_import_file}")
    
    datasets_orig = instantiate(cfg.orig)
    _, _, orig_test_file = datasets_orig.load()
    log.info(f"ORIG test file: {orig_test_file}")

    datasets_dbsas = instantiate(cfg.dbsas)
    _, _, dbsas_test_file = datasets_dbsas.load()
    log.info(f"Discard DBSAS test file: {dbsas_test_file}...")
    os.rename(dbsas_test_file, dbsas_test_file+".bak")

    try:
        os.symlink(orig_test_file, dbsas_test_file)
        log.info(f"Link from {orig_test_file} to {dbsas_test_file} created successfully")
    except (FileExistsError, PermissionError) as e:
        log.info(e)


if __name__ == "__main__":
    run()

