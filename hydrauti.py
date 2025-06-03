"""
/*
 * Software Name : Microtune
 * SPDX-FileCopyrightText: Copyright (c) Orange SA
 * SPDX-License-Identifier: MIT
 *
 * This software is distributed under the MIT license,
 * see the "LICENSE" file for more details
 *
 * Authors: see CONTRIBUTORS.md
 * Software description: MicroTune is a RL-based DBMS Buffer Pool Auto-Tuning for Optimal and Economical Memory Utilization. Consumed RAM is continously and optimally adjusted in conformance of a SLA constraint (maximum mean latency).
 */
"""

import os
import logging
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig

from bandits.gym_env import VSMonitor
import bandits.datasource.adbms_dataframe as ds

# A logger for this file
log = logging.getLogger(__name__)

# Get dictionary val in a context where missing key can be ignored
def dgv(optdict={}, key=""):
    val = optdict.get(key)
    return "NA" if val is None else val

# Env Monitor instanciation (train mode or eval/test mode)
# Workaround of an instanciation problem with Hydra and resolving variables. This doesn't work:
# env_train = instantiate(cfg.tuner.env.wrapper, env={'state_selector': {'df': df_train}})
# It happens when the config contains relative variables like ${..observations.elems}. We don't want to use absolute variables paths ${tuner.env.observations.elems}
# because train and eval env are contextualized retgarding the type of tuner/agent we want.
def instantiate_env_wrapper_wa(cfg_env, to_train:bool, dataframe,  with_scaler=None, perf_meter_args={}):
    if to_train:
        state_selector_p = instantiate(cfg_env.state_selector_train)
        state_selector = state_selector_p(df=dataframe)
        env_p =  instantiate(cfg_env.wrapper.env)
        perf_meter=None
    else:
        state_selector_p = instantiate(cfg_env.state_selector_test)
        if with_scaler is None:
            state_selector = state_selector_p(df=dataframe)
        else:
            state_selector = state_selector_p(df=dataframe, with_scaler=with_scaler)

        env_p =  instantiate(cfg_env.wrapper_test.env)
        perf_meter = instantiate(cfg_env.wrapper_test.perf_meter, **perf_meter_args)

    env = env_p(state_selector=state_selector)

    return VSMonitor(env, perf_meter)

def instantiate_env_wrapper_live_wa(cfg_env, with_scaler=None):
    state_selector_p = instantiate(cfg_env.state_selector_test)
    if with_scaler is None:
        state_selector = state_selector_p()
    else:
        state_selector = state_selector_p(with_scaler=with_scaler)

    env_p =  instantiate(cfg_env.wrapper_test.env)

    env = env_p(state_selector=state_selector)

    return VSMonitor(env, None)

# Too tricky ops to manage output path...
# This must work in RUN and MULTIRUN modes whatever the Hydra output paths configurations and so on. 
# VERY Hard to solve simply..... :-()
def _mk_outputs_path(cfg: DictConfig):
    hydracfg = HydraConfig.get()
    hydra_logs=hydracfg.runtime.output_dir
    log.info(f"Hydra Output directory (logs): {hydra_logs}")

    if str(hydracfg.mode) == "RunMode.MULTIRUN":
        link2create = cfg.link2logs+str(hydracfg.job.num)
    else:
        link2create = cfg.link2logs

    logs_dir = os.path.dirname(link2create)
    if not os.path.exists(logs_dir): # not os.path.isdir(cfg.pickles_path):
        if os.path.islink(logs_dir):
            os.unlink(logs_dir)
        os.makedirs(logs_dir)
    log.info(f"Root logs dir for all tasks: {logs_dir}")

    if not os.path.exists(link2create) or not os.path.samefile(hydra_logs, link2create):
        if os.path.islink(link2create):
            os.unlink(link2create)
        os.symlink(hydra_logs, link2create)
        log.info(f'Job logs symlink {link2create} -> {hydra_logs}')
    
    os.makedirs(cfg.model_cache_path, exist_ok=True)


def prepare_sweeper_trial(cfg: DictConfig):
    hydracfg = HydraConfig.get()
    trial = hydracfg.job.num

    log.info(f'iterations_name: {cfg.iterations_name}{trial}')
    sweeper_params = list(hydracfg.overrides.task)
    log.info(f'Sweeper params: {sweeper_params}')

    _mk_outputs_path(cfg)

    return trial, sweeper_params

# Performs only N (=max) trial when called with a sweeper config
def prepare_run(cfg: DictConfig, max=1) -> bool:
    hydracfg = HydraConfig.get()
    if hydracfg.sweeper.params or str(hydracfg.mode) == "RunMode.MULTIRUN":
        try:
            job = hydracfg.job.num
        except Exception:
            job=0
        if job >= max:
            log.info(f"Sweeper trial evicted:  {job+1}/{max}")
            return False

    _mk_outputs_path(cfg)

    return True

import xxhash

def configId(cfg: DictConfig, combine="") -> str:
    yaml = OmegaConf.to_yaml(cfg, resolve=True)
    return xxhash.xxh3_64_hexdigest(str(yaml)+combine, seed=1)

# eval: True for Eval or False for Test
def instanciate_agent_and_save(cfg: DictConfig, tuner="tuner", datasets: ds.TrainTestDataSets=None, eval=True):
    cfg_baseline = cfg[tuner]
    ctx = "EVAL" if eval else "TEST"

    baseline = instantiate(cfg_baseline.agent)
    log.info(f"Evaluate {baseline.policy.name} to establish performance baseline on {ctx} dataset...")
    hash_code=configId(cfg_baseline, datasets.hashName())
    filever = f'{cfg.datasets_prefix}_{hash_code}_{cfg.iterations_name}0S0' # f'{cfg.datasets_prefix}{hash_code}'
    ext = f"-{ctx.lower()}.pickle"
    exists, optfiles, optdict = baseline.exists(filepath=cfg.model_cache_path, filever=filever, verbose=cfg.verbosity, dftext=ext)
    if exists:
        log.info(f'{baseline.filename} already exists. Reuse pickle in cache...')
    else:
        df = datasets.df_eval if eval else datasets.df_test
        env_baseline = instantiate_env_wrapper_wa(cfg_baseline.env, to_train=False, dataframe=df, perf_meter_args={"name": f'{baseline.policy.shortname}', "stage": ctx.lower()})
        assert baseline.policy.context() == env_baseline.unwrapped.ds.contextElems(), f'Discrepancy between {baseline.policy.name} policy context and env context...'
        baseline.predict(env_baseline, episodes_max=cfg_baseline.TEST_EPISODES_COUNT, verbose=cfg.verbosity)
        baseline_reg_perf = env_baseline.getRegretPerformance()

        optfiles = []
        optdict = dict(reg_perf=baseline_reg_perf, perf_meter=env_baseline.perf_meter)
        baseline.save(filepath=cfg.model_cache_path, filever=filever, ext=ext, verbose=cfg.verbosity, optfiles=optfiles, **optdict)
        log.info(f'{baseline.filename} saved with performance')

    return baseline, optfiles, optdict
