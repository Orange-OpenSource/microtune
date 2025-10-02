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

import numpy as np
import os
import glob
import pickle

from bandits.policy import CtxPolicy
from bandits.gym_env import VSMonitor

from tqdm  import trange, tqdm

import gymnasium as gym

from stable_baselines3.common.callbacks import BaseCallback


# VSBanditAgent: Vertical Scaling agent with N bandits arms
class VSAgent():
    def __init__(self, policy: CtxPolicy):
        #self.verbose = 0
        self.graph_learn = None
        self.graph_predict = None
        self.graph_perf = None
        self.log_interval = 1
        self._arm_min = self._arm_max = 0

        self.policy = policy
        self.filename = None

    #LATER: callback, see https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/callbacks.py on_step(), on_rollout_xxx(), ...
    def _train(self, env: gym.Wrapper, episodes_max, update, deterministic=False, callback: BaseCallback=None, verbose=1):
        self.policy.initWithEnv(env)

        itr = tqdm(range(episodes_max)) if verbose > 0 and episodes_max < np.inf else range(episodes_max)
        debug_arm = bool(max(0,verbose-2))
        
        for episode in itr:
            obs, info = env.reset()
            terminated = truncated = False

            while not (terminated or truncated):
                react = info.get("react")

                if react>0:
                    action = self.policy.getUpArmIndex(boost=react-1)
                else:
                    # Find policy's chosen arm based on input covariates at current time step
                    action = self.policy.select_arm(obs, deterministic=deterministic, debug=debug_arm)

                newobs, rew, terminated, truncated, info = env.step(action)

                # Use reward information for the chosen arm to update
                if update: # and not protect ?
                    self.policy.update(action, rew, obs, newobs)
                
                obs = newobs

    def learn(self, env: VSMonitor, dataset_coverage=1., verbose=0):
        episodes_max = round((env.unwrapped.ds.getTotalStatesCount() * dataset_coverage) // env.unwrapped.max_steps_per_episode) # total * coverage = episodes * steps_per_episode
        status_at_episodes = 0 if verbose<2 else max(1, episodes_max//(verbose*5))
        env.setEpisodesMax(episodes_max, status_at_episodes)
        self.log_interval = 1 #max(1, episodes_max//200)

        self._train(env, episodes_max, update=True, deterministic=False, verbose=verbose)

        if episodes_max > 0:
            self.graph_learn = env.getGraph(f' {env.unwrapped.desc()}')
        if verbose > 1:
            env.showResults()

    # predict: run the policy on the environment, i.e. predict the best arm to choose at each time step
    # episodes_max:
    #  -1: use all workloads in the dataset, i.e. episodes_max = ds.getWorkloadsCount() * abs(episodes_max), thus -N will run N workloads
    #  None: Never stop to run new episodes, i.e. episodes_max = numpy.inf
    #  >0: use episodes_max as the number of episodes to run
    # deterministic: if True, the policy should always choose the same action for the same state    
    # verbose: 0: no display, 1: display every episode, 2: display every 10 episodes, 3: display every 100 episodes, etc.
    def predict(self, env: VSMonitor, episodes_max=-1, deterministic=False, verbose=0, baseline_perf_meter=None, label: str="na"):
        if episodes_max is None:
            episodes_max = np.inf
            status_at_episodes = 1
        else:
            if episodes_max<0:    
                episodes_max = env.unwrapped.ds.getWorkloadsCount()*abs(episodes_max)
            status_at_episodes = 0 if verbose==0 else max(1, episodes_max//(verbose*5))
        env.setEpisodesMax(episodes_max, status_at_episodes)
        self.log_interval = 1
        #verbose = verbose*episodes_max//50 # Determines frequency of the Episode display
        
        self._train(env, episodes_max, update=False, deterministic=deterministic, verbose=verbose)

        self.graph_predict = env.getGraph(f' {env.unwrapped.desc()}')
        if verbose > 0:
            env.showResults()
        
        if env.perf_meter:
            env.perf_meter.label = label
            self.graph_perf = env.getPerfMeterGraph(name=self.policy.name, baseline_perf_meter=baseline_perf_meter)


    def showLearnFig(self, head_title="", renderer=None):
        if self.graph_learn is None:
            return
        
        s_title = self.graph_learn.title
        self.graph_learn.title = head_title+" "+self.graph_learn.title
        fig = self.graph_learn.figure()
        if type(fig) == tuple:
            for i in fig:
                if i != None:
                    i.show(renderer=renderer)
        else:
            fig.show(renderer=renderer)
        
        self.graph_learn.title = s_title

    def saveLearnFig(self, head_title="", filepath=".", filever="", ext=".html"):
        files = []
        if self.graph_learn is None:
            return files

        s_title = self.graph_learn.title
        self.graph_learn.title = head_title+" "+self.graph_learn.title
        filename = self._build_filename(filepath, prefix=str(filever)+'-creg_perEp', dftext=ext)
        fig = self.graph_learn.figure()
        if type(fig) == tuple:
            for i in fig:
                if i != None:
                    i.write_html(filename)
                    files.append(filename)
        else:
            fig.write_html(filename)
            files.append(filename)

        self.graph_learn.title = s_title

        return files

    def showPredictFig(self, head_title="", renderer=None):
        s_title = self.graph_predict.title
        self.graph_predict.title = head_title+" "+self.graph_predict.title
        fig = self.graph_predict.figure()
        if type(fig) == tuple:
            for i in fig:
                i.show(renderer=renderer)
        else:
            fig.show(renderer=renderer)
        
        self.graph_predict.title = s_title

    def savePredictFig(self, head_title="", filepath=".", filever="", ext=".html"):
        s_title = self.graph_predict.title
        self.graph_predict.title = head_title+" "+self.graph_predict.title
        filename = self._build_filename(filepath, prefix=str(filever)+'-creg_perEp', dftext=ext)
        files = []
        fig = self.graph_predict.figure()
        if type(fig) == tuple:
            for i in fig:
                i.write_html(filename)
                files.append(filename)
        else:
            fig.write_html(filename)
            files.append(filename)
        
        self.graph_predict.title = s_title

        return files

    def _delete_files_objects(self, obj_list=[]):
        for object in obj_list:
            if isinstance(object, str) and os.path.isfile(object):
                os.remove(object)

    def _save_obj_list(self, filename="undef.pickle", obj_list=[]):
        with open(filename, 'wb') as pklf:
            for object in obj_list:
                if isinstance(object, str) and os.path.isfile(object):
                    pickle.dump(obj=object, file=pklf)
                    with open(object, 'rb') as ff:
                        data = ff.read()
                else:
                    data = object
                pickle.dump(obj=data, file=pklf, protocol=pickle.HIGHEST_PROTOCOL)
        

    # load objects from pickle file. When the pickle contains a file identified by a known extension (.zip, ...), it is extracted and written on disk
    def _load_obj_list(self, filename="undef.pickle", verbose=0):
        #Return a "generator" on pickle file...
        def load_all():
            with open(filename, "rb") as ff:
                while True:
                    try:
                        yield pickle.load(ff)
                    except EOFError:
                        break            

        pkl_items = load_all()

        obj_list = []
        file_to_extract = None

        for object in pkl_items:
            if file_to_extract is None:
                if isinstance(object, str):
                    file_to_extract = object
                    if verbose:
                        print(f'Will extract: {file_to_extract}')
                obj_list.append(object)
            else:
                dname = os.path.dirname(file_to_extract)
                if dname != '' and not os.path.exists(dname):
                    os.makedirs(dname)
                with open(file_to_extract, 'wb') as f:
                    f.write(object)
                    if verbose:
                        print(f'Extracted: {file_to_extract}')
                    file_to_extract = None

        self.filename = filename

        return obj_list

    def _build_filename(self, filepath:str, prefix:str, filename:str="", dftext=".pickle"):
        assert self.policy is not None, "Cannot save a data without a known policy."

        if filename:
            splitted = os.path.splitext(filename)
            ext = "."+splitted[1] if splitted[1] else "."+dftext
            filename = f'{prefix}{splitted[0]}{ext}'
        else:
            filename = f'{prefix}-{self.policy.name}{dftext}'

        assert os.path.isdir(filepath), f"Invalid file path directory {filepath}"
        if filepath == ".":
            filepath = ""

        return os.path.join(filepath, filename)
    
    # Save agent's policy, a list of files to archive (optfiles list), and an optional dictionary specified as kwargs parameters: p1=param1, p2=param2, ...
    # It is mandatory that filenames in optfiles list are in current folder.
    def save(self, filepath=".", filever="", ext=".pickle", verbose=0, optfiles=[], **optdict):
        filename = self._build_filename(filepath=filepath, prefix=filever, dftext=ext)
        if verbose:
            print(f"Save {filename}...")

        for ff in optfiles:
            assert os.path.isfile(ff), f"Invalid file path: {ff}"

        objlist = []
        policy_data = self.policy.dataToSave()
        objlist.extend(policy_data)
        objlist.extend(optfiles)
        objlist.append(optdict)
        self._save_obj_list(filename, objlist) 
        if verbose:
            print(f"{filename} is saved.")

        self._delete_files_objects(policy_data)

        self.filename = filename

    def _loadFile(self, filename, verbose=0, strict=True):
        if os.path.isfile(filename) is False:
            if strict:
                print(f'cwd:{os.getcwd()} filename not found: {filename}')
            else:
                self.filename = None
                return ([],{})
            filename = None
        elif verbose > 1:
            print(f"Load {filename}...")
        
        assert filename, "No filename specified nor built by default (no policy)"
        self.filename = filename

        objlist = self._load_obj_list(filename) 
        if verbose:
            ol = "" if verbose < 2 else f' with {objlist}' 
            print(f"{filename} is loaded{ol}.")

        objlist = self.policy.restoreData(objlist)

        optdict = objlist.pop(-1) # Pop last elem => it remains optfiles in objlist  
        if verbose > 2:
            print(f"Loaded optdict: {optdict}") 

        return (objlist, optdict) # Return list of archived files and saved optdict (i.e. kwargs) dictionary !!

    # Load agent's policy 
    # strict=true, file must be found absolutely, else this is an error
    def load(self, filepath=".", filever="", verbose=0, strict=True, dftext=".pickle"):
        filename = self._build_filename(filepath=filepath, prefix=filever, dftext=dftext)

        if filename.find('*') > -1:
            if verbose > 1:
                print(f"Search for matching: {filename}")
            flist = glob.glob(filename)
            filename = flist[0] if len(flist)>0 else None

        if verbose > 1:
            print(f"File to load: {filename}")

        return self._loadFile(filename, verbose, strict)

    def exists(self, filepath=".", filever="", verbose=0, dftext=".pickle"):
        objlist, optdict = self.load(filepath=filepath, filever=filever, verbose=verbose, strict=False, dftext=dftext)
        return (False, None, None) if self.filename is None else (True, objlist, optdict)




from bandits.policy_sb3 import SB3Policy


class SB3VSAgent(VSAgent):
    def __init__(self, policy: SB3Policy):
        super().__init__(policy)

    def _train(self, env: gym.Wrapper, episodes_max: int, update: bool, deterministic=False, callback: BaseCallback=None, verbose=1):
        # Learn ??
        if update:
            self.policy.initWithEnv(env)
        
            total_timesteps = episodes_max * env.unwrapped.max_steps_per_episode
            self.log_interval = 100//verbose if verbose>0 else None
            #callback.init_callback(model=policy.model)
            #callback.on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any])
            self.policy.model.learn(total_timesteps=total_timesteps, log_interval=self.log_interval, progress_bar=False, callback=callback)
            #callback.on_training_end()
        else:
            super()._train(env, episodes_max, update=False, deterministic=deterministic, callback=callback, verbose=verbose)



import copy

class AgentsLoaderLoop():
    def __init__(self, agent: VSAgent):
        self.agent = agent

    def load(self, count=0, filepath=".", filever=None, performance_field="", minimimal_performance=True, verbose=0, count_char='S'):
        all_optfiles = []
        all_optdict = []
        best_agent = None
        best_expe = None

        best_performance = np.inf if minimimal_performance else -np.inf

        for loop in range(count):
            optfiles, optdict = self.agent.load(filepath=filepath, filever=f'{filever}{count_char}{loop}', verbose=verbose, strict=False)
            if len(optdict)==0:
                if verbose > 0:
                    print(f"Skip file {filever}L{loop}")
                continue
            all_optfiles.append(optfiles)
            if optdict:
                perf = optdict.get(performance_field)
                if perf:
                    if minimimal_performance:
                        if perf < best_performance:
                            best_performance = perf
                    else:
                        if perf > best_performance:
                            best_performance = perf

                    if perf == best_performance:
                        best_agent = copy.deepcopy(self.agent)
                        best_expe = loop
            else:
                optdict = {}

            all_optdict.append(optdict)  

        return all_optfiles, all_optdict, best_agent, best_expe