
# SIMU preparation data tools
# Use original dataset to add missing columns to SIMU generated dataset

import os
import logging
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import hydrauti as hu
from pkg.datasource.dataframes.obs_samples_dataframes import ObsSamplesDF


def gen_full_datasets(cfg: dict = None):
    version=str(cfg.version)
    orig_dataset=f"./{cfg.dataset_orig_import_file}"
    simu_dataset=f"./{cfg.dataset_simu_import_file}"
    
    obss = ObsSamplesDF(version=version)

    print(f"Loading original dataset from {orig_dataset}...")
    dforig = obss.loadFromPickle(orig_dataset)
    dforig_perf_target_level = dforig['perf_target_level'].unique().tolist()
    print(f"DFORIG pef_target_level: {dforig_perf_target_level}")
    print(f"DFORIG #col:{len(dforig.columns.values.tolist())} #lines:{dforig.shape[0]}")
    workloads = dforig['combined_column'].unique().tolist()
    print(f"DFORIG #workloads:{len(workloads)}")
    obss.saveToPickle("./"+cfg.orig.pickles_prefix+"_full_"+version, dforig)
    log.info(f"DBORIG FULL saved.")

    dfsimu = obss.loadFromPickle(simu_dataset)
    cols2add = ['combined_column', 'tables', 'tables_rows', 'randtype', 'observation.innodb_buffer_pool_size', 'observation.normalized_buf_size', 'extra_info.sysbench.statements_mean']
    dfsimu[cols2add] = dforig[cols2add]
    #dfsimu['objective_margin'] = 0
    #print(dfsimu['observation.normalized_buf_size'].unique().tolist())

    obss.PERF_OBJS = dforig_perf_target_level
    log.info(f"SIMU fix columns...")
    dfsimu = obss.fixColumns(dfsimu, combined_col=False)

    print(f"DFSIMU pef_target_level: {dfsimu['perf_target_level'].unique().tolist()}")
    print(f"DFSIMU #col:{len(dfsimu.columns.values.tolist())} #lines:{dfsimu.shape[0]}")
    workloads = dfsimu['combined_column'].unique().tolist()
    print(f"DFSIMU #workloads:{len(workloads)}")
    objgap = dfsimu["objective_gap"].unique().tolist()
    print(f"DFSIMU objectif_gap: {objgap}")
    obss.saveToPickle("./"+cfg.simu.pickles_prefix+"_full_"+version, dfsimu)
    log.info(f"DFSIMU FULL saved.")

    cache_files = f'{cfg.orig.pickles_prefix}*v{version}-tmp.pickle {cfg.simu.pickles_prefix}*v{version}-tmp.pickle {cfg.simu.pickles_prefix}*v{version}-tmp.pickle.bak'
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
    log.info(f"Run SIMU data preparation version {cfg.version}.{cfg.version_minor}...")
    gen_full_datasets(cfg)
    
    datasets_orig = instantiate(cfg.orig)
    _, _, orig_test_file = datasets_orig.load()
    log.info(f"ORIG test file: {orig_test_file}")

    datasets_simu = instantiate(cfg.simu)
    _, _, simu_test_file = datasets_simu.load()
    log.info(f"Discard SIMU test file: {simu_test_file}...")
    os.rename(simu_test_file, simu_test_file+".bak")

    try:
        os.symlink(orig_test_file, simu_test_file)
        log.info(f"Link from {orig_test_file} to {simu_test_file} created successfully")
    except (FileExistsError, PermissionError) as e:
        log.info(e)


if __name__ == "__main__":
    run()

