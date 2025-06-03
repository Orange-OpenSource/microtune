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

#qpslat_weights="01"   # QPS is not used in performance computation (weigth is 0%), the objective is to maintain te Latency (weigth is 100%). Other options is "19" (10% for QPS, 90% for Latency)

import os
import logging
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig

import bandits.datasource.adbms_dataframe as ds
#from bandits.datasource.dataframes.obs_samples_dataframes import ObsSamplesDF 

#from minio import Minio

#mm = Minio()

# A logger for this file
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="agent")
def run(cfg: DictConfig) -> float:
    #cfg.env.observation_space.elems.remove("sysbench_filtered.latency_mean") # Ensure latency is not used in context
    trial = HydraConfig.get().job.num

    #print(OmegaConf.to_yaml(cfg, resolve=True))
    #exit(0)
    print(f'iterations_name: {cfg.iterations_name}{trial}')
    sweeper_params = list(HydraConfig.get().overrides.task)
    print(f'Sweeper params: {sweeper_params}')

    if not os.path.isdir(cfg.pickles_path):
      os.makedirs(cfg.pickles_path)

    src = os.getcwd()+"/"+HydraConfig.get().sweep.dir
    dst = cfg.pickles_path+"/logs"
    if os.path.islink(dst):
      os.unlink(dst)
    os.symlink(src, dst)
    print(f'Created symlink {src} {dst}')

    # Choose ratio=80 for 80% train and 20% eval/test. Choose ratio=0 for a separation by clients. Ex Train on all Odd wl_clients num and eval on even wl_clients num
    datasets = ds.TrainTestDataSets(testastrain=False, ratio=0, version=cfg.version, picklefiles=cfg.datasets_prefix, 
                                    perf_level=cfg.IPERF_LEVEL, randtypes=[ "special", "gaussian", "uniform", "pareto" ], clients=[1,2,3,4,5,6,7,8,9,10,11,12], 
                                    seed=42, verbose=1)
    datasets.splitTestsForEval(ratio=50)
    df_train = datasets.df_train
    df_eval = datasets.df_eval
    datasets.df_test = None

    train_regret = []
    eval_regret = []
    eval_perf = []
    graph_evals = None
    renderer = cfg.graph_renderer if cfg.graph_renderer != "None" else None

    for sid in range(cfg.n_seeds): 
        RND_SEED = cfg.RND_SEED+sid
        htmlfiles = []
        filever = f'{cfg.iterations_name}{trial}S{sid}'

        #TRAIN    
        env_train = instantiate(cfg.tuner.env.wrapper, env={'state_selector': {'df': df_train}})

        agent = instantiate(cfg.tuner.agent, policy={"seed": RND_SEED}) #, "verbose": cfg.xtraverbosity})
        agent.learn(env_train, cfg.tuner.TRAINING_COVERAGE, verbose=cfg.verbosity)
        minmax_scaler = env_train.unwrapped.ds.min_max_scaler

        # Show learning graph!
        if renderer != None:
            agent.showLearnFig(head_title=f"LEARN/{agent.policy.name} TotalSteps:{env_train.unwrapped.total_steps}", renderer=renderer)

        htmlgraph = agent.saveLearnFig(filepath=cfg.pickles_path, head_title=f"LEARN/{agent.policy.name} TotalSteps:{env_train.unwrapped.total_steps}", filever=filever+"-train")
        htmlfiles.extend(htmlgraph)

        total_ep = env_train.unwrapped.cur_episode
        cur_train_regret = env_train._regret_per_episode[:total_ep].cumsum()
        train_regret.append(cur_train_regret)
        
        # /!\ Evaluate with Evaluation DataSet !!!
        env_eval = instantiate(cfg.tuner.env.wrapper_test, env={'state_selector': {'df': df_eval, 'with_scaler': minmax_scaler}})
        agent.predict(env_eval, episodes_max=cfg.tuner.TEST_EPISODES_COUNT, deterministic=cfg.DETERMINISTIC, verbose=cfg.xtraverbosity)

        htmlgraph = agent.savePredictFig(filepath=cfg.pickles_path, head_title=f"EVAL/{agent.policy.name} TotalSteps:{env_eval.unwrapped.total_steps}", filever=filever+"-eval")
        htmlfiles.extend(htmlgraph)

        total_ep = env_eval.unwrapped.cur_episode
        cur_eval_regret = env_eval._regret_per_episode[:total_ep].cumsum()
        eval_regret.append(cur_eval_regret)

        cur_performance = env_eval.getRegretPerformance()
        print(f'Env eval cumulated regret: {cur_performance}')
        eval_perf.append(cur_performance)

        if sid == 0:
            graph_evals = env_eval.graphPerWorkload(f"Cumulated Regret/workload. Trained on: {env_train.unwrapped.msgStatus()} {env_train.unwrapped.desc()}")
        graph_evals.addCurve(f"{agent.policy.name} CReg:{round(env_eval.getRegretPerformance(),2)} S:{RND_SEED}", y=env_eval.resultPerWorkload(), perf=env_eval.getRegretPerformance())

        fig = graph_evals.figure()
        htmlgraph = os.path.join(cfg.pickles_path, f'{filever}-eval-creg_perWL-{agent.policy.name}.html')
        fig.write_html(htmlgraph)
        htmlfiles.append(htmlgraph)

        agent.save(filepath=cfg.pickles_path, filever=filever, verbose=1, optfiles=htmlfiles, 
                    tid=trial, sid=sid, seed=RND_SEED, min_max_scaler=minmax_scaler, train_lr=cfg.tuner.agent.policy.learning_rate,
                    train_regret=cur_train_regret, train_reg_perf=env_train.getRegretPerformance(),
                    train_reg_results=env_train.resultPerWorkload(), 
                    eval_regret=cur_eval_regret, eval_reg_perf=cur_performance,
                    sweeper_params=sweeper_params, config=OmegaConf.to_yaml(cfg, resolve=True))    

        log.info(f'Saved {filever} {agent.policy.name} Seed:{RND_SEED} Params:{sweeper_params} EvalRegPerf:{cur_performance}')
        
        if not cfg.graph_keep_html:
            for ff in htmlfiles:
                os.remove(ff)

        env_train.close()
        env_eval.close()

    if renderer != None:
        fig = graph_evals.figure()
        fig.show(renderer=renderer) # svg is cleaner but not in Gitlab. None is the best in VSCode

    perf = np.average(eval_perf)
    log.info(f'Sweep with: {perf} in {eval_perf}')
    return perf


if __name__ == "__main__":
    run()

