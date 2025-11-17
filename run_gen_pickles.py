
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
# Essentially DBSAS, CVAE, TABDDPM SIMUs preparation data tool, but can be used even when a single dataset has to be prepared for a long run of training/tests.
# Thus this command can be lauched at each run to prepare Train/Eval/Test data whatever the kind of experiment.
# Prepare from an Original (not simulated) dataset a SIMU (simulation generated) dataset by fixing some missing collumns or anything else
# and save the results into pickles files for further usage.
# Note that eval and test files are symlinked to original dataset files to ensure exact same data for eval and test.

import os
import logging
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import hydrauti as hu
from pkg.datasource.dataframes.obs_samples_dataframes import ObsSamplesDF

# A logger for this file
log = logging.getLogger(__name__)
#lock = Lock()

# set_iperf indicates whether to compute the 'iperf' column or not and then allows computation of DOWN, STAY, UP arms counts
def save_full_dataset(name, import_file, cfg_ds, set_iperf=True, dforig = None):
    version=cfg_ds.version
    pickles_prefix=cfg_ds.pickles_prefix

    obss = ObsSamplesDF(version=version)
    df = obss.loadFromPickle(import_file)

    # Add some fixes from original dataset if provided
#    if dforig is not None:
#       #cols2add = ['combined_column', 'tables', 'tables_rows', 'randtype', 'observation.innodb_buffer_pool_size', 'observation.normalized_buf_size']
#        cols2add = [ "db_size_mb" ]
#        df[cols2add] = dforig[cols2add]

    log.info(f"DF{name} {pickles_prefix} V:{version} fix perf...")
    obss.PERF_OBJS = [ float(cfg_ds.perf_level) ]
    df = obss.fixColumns(df, combined_col=False, compute_iperf=set_iperf) 

    log.info(f"DF{name} perf_target_level: {df['perf_target_level'].unique().tolist()}")
    log.info(f"DF{name} #col:{len(df.columns.values.tolist())} #lines:{df.shape[0]}")
    workloads = df['combined_column'].unique().tolist()
    log.info(f"DF{name} #workloads:{len(workloads)}")

    objgap = df["objective_gap"].unique().tolist()
    log.info(f"DF{name} objective_gap: {objgap}")

    obss.saveFullPickle(pickles_prefix, df)
    log.info(f"DF{name} {pickles_prefix} FULL saved.")

    cache_files = f'{pickles_prefix}*v{version}-tmp.pickle {pickles_prefix}*v{version}-tmp.pickle {pickles_prefix}*v{version}-tmp.pickle.bak'
    log.info(f"Cleaning up {cache_files}...")
    os.system(f'rm -f {cache_files}')

    return df, workloads


def prepare_train_eval_test_data(name, cfg_ds, orig_eval_file: str = None, orig_test_file: str = None):
    dataset_obj = instantiate(cfg_ds)

    # Load full dataset file and split into train/eval/test files
    dataset_train_file, dataset_eval_file, dataset_test_file = dataset_obj.load()
    log.info(f"DF{name} Eval,Test file: {(dataset_train_file, dataset_eval_file, dataset_test_file)}")

    if orig_eval_file is not None or orig_test_file is not None:
        simus_to_fix = [ {'simu_file': dataset_eval_file, 'orig_file': orig_eval_file},
                        {'simu_file': dataset_test_file, 'orig_file': orig_test_file} ]

        for fix in simus_to_fix:
            simu_file = fix['simu_file']
            orig_file = fix['orig_file']
            if orig_file is not None:
                log.info(f"Discard SIMU file: {simu_file}...")
                name, _ = os.path.splitext(simu_file)
                os.rename(simu_file, name+".bak")
                try:
                    os.symlink(orig_file, simu_file)
                    log.info(f"Link from {orig_file} to {simu_file} created successfully")
                except (FileExistsError, PermissionError) as e:
                    log.info(e)
            else:
                log.info(f"No ORIG file provided to discard SIMU file: {simu_file}...")

    return dataset_train_file, dataset_eval_file, dataset_test_file


#from typing import Tuple

@hydra.main(version_base=None, config_path="configs", config_name="gen_pickles")
def run(cfg: DictConfig) -> float: #Tuple[float, float]:
    # Performs only 1 trial (because trials are not applicable here)
    if not hu.prepare_run(cfg):
        return 0.

    log.info(cfg.info)
    log.info(f"Run data preparation version {cfg.version}...")
    
    # Gen ORIG full dataset file ? Else assume a full ORIG dataset pickle file is already present
    if "dataset_orig_import_file" in cfg and cfg.dataset_orig_import_file is not None:
        log.info(f'Loading DFORIG dataset from {cfg.dataset_orig_import_file} and save full version...')
        dforig, workloads_orig = save_full_dataset("ORIG", cfg.dataset_orig_import_file, cfg.orig)
        PERF_OBJS = dforig['perf_target_level'].unique().tolist()
        log.info(f"DFORIG pef_target_level: {PERF_OBJS}")
        log.info(f"DFORIG #workloads:{len(workloads_orig)} {workloads_orig}")
        db_size_mb_lst = dforig["db_size_mb"].unique().tolist()
        print("DFORIG DB SIZE MB", db_size_mb_lst, len(db_size_mb_lst))
    else:
        dforig = None

    # Prepare train/eval/test files for original dataset from full dataset file
    _, orig_eval_file, orig_test_file = prepare_train_eval_test_data("ORIG", cfg.orig)

    if dforig is not None:
        # Gen SIMU full datasets files and ensures same eval/test files as original dataset
        list_datasets = [
            {'import_file': cfg.dataset_dbsas_import_file, 'cfg_ds': cfg.dbsas if "dbsas" in cfg else None, 'name': 'DBSAS'},
            {'import_file': cfg.dataset_cvae_import_file, 'cfg_ds': cfg.cvae if "cvae" in cfg else None, 'name': 'CVAE'},
            {'import_file': cfg.dataset_tabddpm_import_file, 'cfg_ds': cfg.tabddpm if "tabddpm" in cfg else None, 'name': 'TABDDPM'},
        ]

        for simu in list_datasets:
            cfg_ds = simu["cfg_ds"]

            if cfg_ds is not None:
                log.info(f'Loading DF{simu["name"]} dataset from {simu["import_file"]} and save full version...')
                _, workloads = save_full_dataset(simu["name"], simu["import_file"], cfg_ds)
                assert workloads == workloads_orig, f'Workloads in DF{simu["name"]} dataset differ from original dataset!'
                prepare_train_eval_test_data(simu["name"], cfg_ds, orig_eval_file, orig_test_file)
            else:
                log.info(f'Skipping DF{simu["name"]} dataset preparation because not configured.')


if __name__ == "__main__":
    run()

