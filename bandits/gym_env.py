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
import xxhash

from bandits.datasource.dataset import ADBMSDataSetEntryContextSelector
from bandits.reward import RewardDownStayUp
from bandits.perf_meter import PerfMeter

import gymnasium as gym 
from gymnasium import spaces
from stable_baselines3.common.type_aliases import GymStepReturn, GymResetReturn
from gymnasium.core import ActType, ObsType, RenderFrame
from typing import Any
from bandits.graph import GraphPx

import pickle
from datetime import datetime

# A logger for this file
import logging
log = logging.getLogger(__name__)

# VecEnv and Normalisation: https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/vec_env/vec_normalize.html#VecNormalize
# Check if observation is vectorized or not:  https://stable-baselines3.readthedocs.io/en/master/common/utils.html#stable_baselines3.common.utils.is_vectorized_observation

# Env with a Discrete action space from Arm0=0 up to ArmN=N
# Actions corresponding to each arms are mapped from the Reward's action space ('action_minmax' parameter). With Action min-max being typically (-1, 1) for 3 arms.
# on_terminate: if >=0 return a terminated step if regret==0 and action==STAY. The reward receive a bonus of +on_terminate when the step is terminated. If -1, do not manage the terminated state. 
class VSEnv(gym.Env):
    #metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    metadata = {"render_modes": ["human"], "render_fps": 4}

    # The dataframe comes with a perfomance objective. For example, if it is at 0.99 with a gap of 4% (0.04) that means SLA is OK if perf indicator is between 0.99 and 0.99*(1+0.04).
    # qpslat_w: The weight of QPS (queries/s) and latency. "19" stands for 0.1 for QPS and 0.9 for Latency. "01" stands for 0% QPS, 100% latency.
    #           Admited values "01", "19"
    # slaprotect: If True, try to reduce SLA VIOLATIONs by choosing, at step time, the UP direction whatever the arm choosen by the policy
    # Note that the performance Objective Gap in percent defining the admitted gap around the performance objective (a value in 0.5, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999) is picked up from the dataset. 
    def __init__(self, state_selector: ADBMSDataSetEntryContextSelector, reward: RewardDownStayUp = None, notify_react=False, max_steps_per_episode=64, on_terminate=-1, verbose=0):    
        self.ds = state_selector
        if max_steps_per_episode is None:
            max_steps_per_episode = np.inf
        elif max_steps_per_episode<0:
            max_steps_per_episode = self.ds.getStatesCountPerWorkload()*abs(max_steps_per_episode)

        self.max_steps_per_episode = max_steps_per_episode 
        self.reward = reward
        self.verbose = verbose
        self._notify_react = notify_react
        self.render_mode = "human"
        self._on_terminate = on_terminate
        self._on_terminate_count = 0

        elems = self.ds.contextElems()
        if self.ds.min_max_scaler is None:
            min_obs, max_obs = (-np.inf, np.inf)
        else:
            min_obs, max_obs = self.ds.min_max_scaler.feature_range

        self.observation_space = spaces.Box(low= min_obs, high=max_obs, shape=(len(elems),), dtype=float)
        self.action_space = spaces.Discrete(n=reward.actions.count(), seed=self.ds.seed, start=0) # Keep Start=0 to be in conformance with VSAgent arms Index!!!
        
        self.reinit()

    def desc(self, complete=False):
        addon=f'LatGap:{self._latgap_ms} ACTIONS:{self.action_space}' if complete else ''
        return f'WL:{self.ds.getWorkloadsCount()} States:{self.ds.getTotalStatesCount()} LatTgt:{round(self.lat_target,1)} {addon}'

    def msgStatus(self):
        return f'Ep:{self.cur_episode +1} Steps:{self.total_steps +1}' # CRew:{self._rew_cumul} CReg:{self._reg_cumul}'

    def getRewardStates(self):
        return self.reward.getStates()

    def getIDeltaOnAction(self, action):
        cur_state = self.ds.state()
        act_state = self.ds.state(bufferidx_increment=action)
        if act_state is None:
            return cur_state["delta_perf_target01"], None
        return cur_state["delta_perf_target01"], act_state["delta_perf_target01"]


    def _init_env(self):
        self.cur_episode = self.cur_step = -1
        self.total_steps = 0
        self.bufsize_t0 = 0
        self.terminated_count = 0
        self._bufsize_dba = 0
        self._rew_cumul = 0     # Cumulatice reward per episode
        self._reg_cumul = 0     # Cumulative Regret per episode
        self._obs_digest = 0
        self._obs_digest_dsp = 'NA'
        #add idelta list
        self._idelta_list = []

    # Called at each state change (i.e. either a workload or buffer size change)
    # Use unscaled action in [action_min, action_max] interval, where action_min is typically negative for a DOWN action, and positiv for a UP action. STAY action is 0.
    def _new_state(self, action=1, real_action=1):
        if real_action != 0:
            self.workload_name_id = self.ds.getWorkloadNameId()
            self.lat_gap, self._latgap_ms, self.lat_target = self.ds.getLatencyGapTarget()
            self.iperf, self.delta_perf, self.perf_target = self.ds.getIPerfIndicators()
            self.latency = self.ds.getLatency()
            self._wl_context = f'{self.workload_name_id} {self.perf_target} {round(self.lat_target, 0)}ms'
            ## fanfan here
            # self._idelta_list.append(self.delta_perf)
            self.reward.setState(self.ds)

        def under_or_violation():
            return "VIOLATION" if action <1 else "UNDER"

        self.sla = self.reward.actionMax2Lambda(under=under_or_violation)

    def reinit(self):
        self._init_env()
        self._new_state(0)
        self._obs_digest_update()
        self._logmsg(msgfun=self._msg_init)


    def _reset_state(self):
        self.ds.reset(self.cur_episode) # Select new workload group from dataset selector, according to the selector type
        self.ds.next()                  # Select next entry state in current workload group from dataset selector, according to the selector type
        self._new_state(0)
        self._obs_digest_update()
        self._on_terminate_count = 0

        state = self.ds.state()        
        buf_size_mb = state["buf_size_min_mb"]
        self._bufsize_dba = max(buf_size_mb*2, (int(state["db_size_mb"]*0.8)//buf_size_mb)*buf_size_mb)
        self.bufsize_t0 =  state["buf_size"]//1024//1024

        assert self.ds.entry_idx == (self.ds.getStatesCountPerWorkload() -1 - state["buf_size_idx"]), f'Incoherence {self.ds.entry_idx} != {(self.ds.getStatesCountPerWorkload() -1 - state["buf_size_idx"])}'
        assert self.ds.globalIndex() >=0, f"Error: GIDX:{self.ds.globalIndex()}"

    # Use unscaled action in [action_min, action_max] interval, where action_min is typically negative for a DOWN action, and positiv for a UP action. STAY action is 0.
    def _next_state(self, action):
        real_action = self.ds.move(action) # Move in current workload group
        self._new_state(action, real_action)
        self._obs_digest_update()
    
        return self.ds.context(), real_action

    def _obs_digest_update(self):
        if self.verbose>1:
            new_digest = xxhash.xxh3_64_hexdigest(self.ds.context())

            name='==='
            if new_digest != self._obs_digest:
                self._obs_digest = new_digest
                name = 'CTX' 

            self._obs_digest_dsp = f'{name}:{new_digest}'

    def _info(self, regret=0, pact=1, terminated=False, truncated=False):
        cur_buf_size_mb = self.getBufferSizeMB()
        return {
            "react": 1 if self._notify_react and self.sla == "VIOLATION" else 0,
            "pact": pact,
            "regret": regret,
            "rew_cumul": self._rew_cumul,
            "regret_cumul": self._reg_cumul,
            "iregret_cumul": self._ireg_cumul,
            "delta_perf": self.delta_perf,
            "buf_size_mb": cur_buf_size_mb,
            "mem_gain": self._bufsize_dba - cur_buf_size_mb,
            "TimeLimit.truncated": truncated and not terminated,
        }
    
    def _msg_init(self, *args):
        return f"Sample_states len: {self.ds.getTotalStatesCount()} #Workloads:{self.ds.getWorkloadsCount()}"
    
    def _logmsg(self, msglevel=0, msgfun=lambda: "", *args):
        if msglevel <= self.verbose:
            log.info(msgfun(*args))

    # Return Normalized buf_size and size in MB
    def getBufferSizeMB(self):
        return self.ds.state()["buf_size"]//1024//1024
            
    def getBufferSizeNormalized(self):
        return self.ds.state()["observation.normalized_buf_size"]

    def workloadsList(self):
        return self.ds.getWorkloadsList()

    def getWorkloadsCount(self):
        return self.ds.getWorkloadsCount()
    
    def getKeyValue(self, key=""):
        state = self.ds.state()
        return state[key]
    
    #def getEpisodeArmsCount(self):
    #    state = self.ds.state()
    #    return state["ARM0_"+self.qpslat_w], state["ARM1_"+self.qpslat_w], state["ARM2_"+self.qpslat_w]

    def getEpisodeWorkloadContext(self):
        return self._wl_context
    
    def _msg_reset(self, *args):
        desc = ""
        #    totarm0, totarm1, totarm2 = self.getEpisodeArmsCount()
        #    desc = f'LatGap:{self._latgap_ms}ms #Arm0:{totarm0} #Arm1:{totarm1} #Arm2:{totarm2} for this workload'
        return f'E{self.cur_episode} {self._wl_context} DBSize:{self.ds.state()["db_size_mb"]}MB Dft DBA:{self._bufsize_dba}MB {desc}'
    
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> GymResetReturn:
        super().reset(seed=seed)
        #if self.cur_episode == -1:
        #    np.random.seed(self.seed)

        #if self.cur_step == -1 and self.cur_episode == 0: 
        #    print(f"WARNING: Empty episodes at start are not counted...")
        #else:
        self.cur_episode += 1
        self.cur_step = -1

        self._reset_state()

        self._rew_cumul = 0     # Cumulatice reward per episode
        self._reg_cumul = 0     # Cumulative Regret per episode
        self._ireg_cumul = abs(self.delta_perf)     # Cumulative IPERF Regret per episode

        self._logmsg(1, self._msg_reset)

        return self.ds.context(), self._info()

    def _msg_step(self, cur_buf_size, arm_name, rew, desc=""):
        iperf, idelta, threshold = self.unwrapped.ds.getIPerfIndicators()  # Return IPERF, IDELTAPERF, IPERFTARGET
        state = self.unwrapped.ds.state()        
        lat = state.get("sysbench_filtered.latency_mean")
        lat = "NA" if lat is None else round(lat, 2)

        msg = f'S{str(self.unwrapped.cur_step).ljust(2)} Buf:{str(cur_buf_size).ljust(6)}MB Arm-{str(arm_name).ljust(3)}'
        msg += f'Rew:{str(round(rew,3)).ljust(6)} RegCumul:{str(round(self._reg_cumul,3)).ljust(7)} IRegCumul:{str(round(self._ireg_cumul,3)).ljust(7)}'
        msg += f' SLA-{self.sla.ljust(10)} IPERF{self.unwrapped.ds.getQPSLatWeigths()}/OBJ:{round(iperf,3)}/{threshold} IDELTA:{round(idelta,3)} LAT:{lat}ms'.ljust(66)
        if self.verbose>1:
            msg += self._obs_digest_dsp
        if self.verbose>2:
            msg += f' REW:{self.reward.getStates()}'
        msg += f' {desc}'
        return msg

    # Here action is Arm's action, so in range [0,k_arms-1]
    def step(self, action: ActType) -> GymStepReturn:
        assert self.action_space.contains(action), f"Error, invalid action {action} type:{type(action)} in action_space: {self.action_space}"
        _, idelta, _ = self.unwrapped.ds.getIPerfIndicators() 
        self._idelta_list.append(idelta)
        u_action = self.reward.actions.arm2action(action) # Unscaled Action. Shift action in the reward's action_minmax interval

        return self._step_unscaled_action(u_action)

    # Use unscaled action in [action_min, action_max] interval, where action_min is typically negative for a DOWN action, and positiv for a UP action. STAY action is 0.
    def _step_unscaled_action(self, action: int):
        #assert self.ds.globalIndex() >=0, f"Error: GIDX:{self.ds.globalIndex()}"
        self.cur_step += 1; self.total_steps += 1

        rew, regret, pact = self.reward.get(action)
        self._rew_cumul += rew      # Cumulatice reward per episode
        self._reg_cumul += regret   # Cumulative Regret per episode
       
        prev_buf_size = self.getBufferSizeMB()

        self._logmsg(1, self._msg_step, prev_buf_size, self.reward.action2Name(action), rew)

        # **NEW STATE**
        # Now we can go Next step! => change current state
        obs, real_action = self._next_state(action)

        self._ireg_cumul += abs(self.delta_perf) # Cumulative IPERF Regret per episode

        truncated = True if self.cur_step+1 >= self.max_steps_per_episode else False
        terminated = False
        # Manage Episode termination ?
        if self._on_terminate >= 0:
            # On count condition, Stop the episode, if STAY action is done without any regret!
            if real_action == 0 and regret == 0:
                self._on_terminate_count += 1
                rew += (self._on_terminate_count/(self._on_terminate+1)) #self._on_terminate
                if self._on_terminate_count >= self._on_terminate:
                    terminated=True
                    self.terminated_count += 1

        return obs, rew, terminated, truncated, self._info(regret=regret, pact=pact, terminated=terminated, truncated=truncated) #obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            log.info(f'Total Ep:{self.cur_episode} Cur WL:{self.getEpisodeWorkloadContext()} Cur Step:{self.cur_step}')
        return None
        
    def close(self):
        #self.observations = None
        pass

# Continous actions in a symmetrical space, interval [-1., 1]
# Action 0 is STAY ONLY if STAY action is centered. 
# Action -1 is Arm0, Action +1 is ArmN. Action STAY is between -1., 1. depending of Reward's action_minmax parameter.
class VSEnvContinousSpace(VSEnv):
    def __init__(self, state_selector: ADBMSDataSetEntryContextSelector, reward: RewardDownStayUp = None, notify_react=False, max_steps_per_episode=64, on_terminate=-1, verbose=0):
        super().__init__(state_selector, reward, notify_react, max_steps_per_episode, on_terminate, verbose)
        # Use symetrical continous actions -1., 1.
        self.action_space = spaces.Box(low=-1., high=1., shape=(1,), dtype=float, seed=self.ds.seed) # Continous and symetrycal action space, as recommended in STB3 tips (https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html) 
        self._action_minmax = reward.actions.minMax()

    # Returns obs, rew, terminate, info
    def step(self, action: ActType) -> GymStepReturn:
        assert self.action_space.contains(action), f"Error, invalid action {action} type:{type(action)} in action_space: {self.action_space}"

        # Unscale a normalized action in [-1.,1.] interval to an action in minmax interval 
        u_action = round(((action[0] + 1) * (self._action_minmax[1] - self._action_minmax[0]) // 2) + self._action_minmax[0])

        return self._step_unscaled_action(u_action)



import abc

class VSMonitorCallback(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def onStart(self, env: VSEnv):
        # Happens only when training, test or inference starts
        pass

    @abc.abstractmethod
    def onReset(self, env: VSEnv):
        pass

    @abc.abstractmethod
    def onStep(self, env: VSEnv, info: dict={}):
        pass

# LATER: should be a sub-class of Gymnasium Monitor wrapper
class VSMonitor(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
    def __init__(self, env: VSEnv, perf_meter: PerfMeter=None):    
        super().__init__(env)
        self._rew_min = np.inf
        self._rew_max = -np.inf
        self.setEpisodesMax()
        self.perf_meter = perf_meter
        self._callbacks = []

    def addCallback(self, cb: VSMonitorCallback):
        self._callbacks.append(cb)

    def setEpisodesMax(self, episodes_max=1, status_at_episodes=0):
        self._results = {}
        self._partial = False
        self._reward_per_episode = None
        self._regret_per_episode = None
        self._iregret_per_episode = None
        self._reg_performance = 0
        self._rew_performance = 0
        self._episodes_max = episodes_max
        self._status_at_episodes = status_at_episodes

    def _on_training_start(self):
        log.info(f'Starting {self._episodes_max} episodes of {self.unwrapped.max_steps_per_episode} steps Print every {self._status_at_episodes} Ep. (to change: config.verbosity on train, config.xtraverbosity on predict)')

        self._rew_min = np.inf      # Global for ALL episodes
        self._rew_max = -np.inf     # Global for ALL episodes
        self._results = {}

        self._partial = False
        self._reward_per_episode = np.zeros(self._episodes_max)
        self._regret_per_episode = np.zeros(self._episodes_max)
        self._iregret_per_episode = np.zeros(self._episodes_max)
        self._reg_performance = 0.
        self._rew_performance = 0.
        if self.perf_meter:
            self.perf_meter.startSession()
        for cb in self._callbacks:
            cb.onStart(self.unwrapped)

    def _on_step(self, obs, rew, terminated, truncated, info):
        self._rew_min = min(self._rew_min, rew)
        self._rew_max = max(self._rew_max, rew)

        if self.perf_meter:
            self.perf_meter.onNextStep(regret=info["regret"], idelta=info["delta_perf"], buf_size=info["buf_size_mb"])        

        for cb in self._callbacks:
            cb.onStep(self.unwrapped, info)

        if terminated or truncated:
            env = self.unwrapped

            if env.cur_episode >= self._episodes_max and not self._partial:
                self._episodes_max = int(self._episodes_max*1.5)
                #print(f'Reallocate results for {self._episodes_max} episodes...')
                self._reward_per_episode.resize(self._episodes_max)
                self._regret_per_episode.resize(self._episodes_max)
                self._iregret_per_episode.resize(self._episodes_max)                
                #self._partial = True
                #print(f'WARNING: Results Truncated Ep/Max:{env.cur_episode}/{self._episodes_max} and Step:{env.cur_step}')

            partial = '/!\\' if self._partial else '' 

            ep_ctx = env.getEpisodeWorkloadContext()
            episode = env.cur_episode

            if self._results.get(ep_ctx) is None:
                self._results[ep_ctx] = { "MeanMemGain": 0, "MeanRegret": 0, "MeanIRegret": 0, "Episodes": [] }

            ep_per_wl = len(self._results[ep_ctx]["Episodes"])            
            self._results[ep_ctx]["MeanMemGain"] = ((self._results[ep_ctx]["MeanMemGain"]*ep_per_wl)+info["mem_gain"])/(ep_per_wl+1)  # Incremental averaging                
            self._results[ep_ctx]["MeanRegret"] = ((self._results[ep_ctx]["MeanRegret"]*ep_per_wl)+info["regret_cumul"])/(ep_per_wl+1)  # Incremental averaging
            self._results[ep_ctx]["MeanIRegret"] = ((self._results[ep_ctx]["MeanIRegret"]*ep_per_wl)+info["iregret_cumul"])/(ep_per_wl+1)  # Incremental averaging

            self._results[ep_ctx]["Episodes"].append([f'E{episode} S{env.cur_step} {env.bufsize_t0}MB', round(info["rew_cumul"], 4), round(info["regret_cumul"],4), round(info["iregret_cumul"],4)])
            if not self._partial:
                self._reward_per_episode[episode] = info["rew_cumul"]
                self._regret_per_episode[episode] = info["regret_cumul"]
                self._iregret_per_episode[episode] = info["iregret_cumul"]

            if self._status_at_episodes > 0 and episode%self._status_at_episodes == 0:
                msg = f'E{env.cur_episode} REWmin:{round(self._rew_min,2)} REWmax:{round(self._rew_max,2)}'
                epidx = env.cur_episode+1
                msg += f' REWmean/Episode{partial}:{round(self._reward_per_episode[:epidx].mean(),4)} on Last{self._status_at_episodes}{partial}:{round(self._reward_per_episode[:epidx][-self._status_at_episodes:].mean(),4)}'
                msg += f' REWcumul{partial}:{round(self._reward_per_episode[:epidx].sum(),4)} REGcumul{partial}:{round(self._regret_per_episode[:epidx].sum(),4)} IREGcumul{partial}:{round(self._iregret_per_episode[:epidx].sum(),4)}'
                msg += f' MeanRegret:{round(self._results[ep_ctx]["MeanRegret"],3)} MeanIRegret:{round(self._results[ep_ctx]["MeanIRegret"],3)} MemGain:{info["mem_gain"]}MB, {len(self._results)} workload types'
                log.info(msg)
        
        return obs, rew, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> GymResetReturn:
        episode = self.unwrapped.cur_episode
        if episode == -1:
            self._on_training_start()

        #assert episode <= self._episodes_max and episode >= -1, f"Invalid episodes count/Max {episode}/{self._episodes_max}"

        context, info = self.unwrapped.reset(seed=seed, options=options)

        if self.perf_meter:
            self.perf_meter.onNextEpisode(self.unwrapped.workload_name_id+'_'+str(self.unwrapped.cur_episode))

        for cb in self._callbacks:
            cb.onReset(self.unwrapped)

        return context, info

    def step(self, action: ActType) -> GymStepReturn:
        obs, rew, terminated, truncated, info = self.unwrapped.step(action=action)

        return self._on_step(obs, rew, terminated, truncated, info)
    
    def showResults(self):
        total_wl_types = len(self._results)
        if total_wl_types > 0:
            total_ep = 0
            total_mean_regret = 0
            total_mean_iregret = 0
            total_gain = 0
            for wl, res in self._results.items():
                ep_count = len(res["Episodes"])
                total_ep += ep_count
                mean_regret = res["MeanRegret"]
                total_mean_regret += mean_regret
                mean_iregret = res["MeanIRegret"]
                total_mean_iregret += mean_iregret
                gain = res["MeanMemGain"]
                total_gain += gain
                log.info(f'{wl.ljust(36)} #Ep:{ep_count} MeanMemGain/ep:{int(gain)}MB MeanRegret/ep:{round(mean_regret,4)} Episodes:{res["Episodes"]}')

            partial = '/!\\' if self._partial else '' 

            log.info('**Results')
            log.info(f'Total WL types learned/availables:{total_wl_types}/{self.unwrapped.getWorkloadsCount()} Total episodes:{total_ep}')
            log.info(f'REWmin:{round(self._rew_min,2)} REWmax:{round(self._rew_max,2)}')
            log.info(f'REWmean/Episode{partial}:{round(self._reward_per_episode[:total_ep].mean(),2)}')
            log.info(f'REWcumul{partial}:{round(self._reward_per_episode[:total_ep].sum(),4)}')
            log.info(f"Total memory gain:{int(total_gain)}MB (Out of unreacheable SLA)")
            log.info(f"Mean memory gain/WLtypes:{total_gain//total_wl_types}MB (Out of unreacheable SLA)")
            log.info(f"Mean of (MeanRegrets/WL):{round(total_mean_regret/total_wl_types, 5)}")
            log.info(f"Sum of (MeanRegrets/WL):{round(total_mean_regret, 5)}")
            log.info(f"Mean of MeanIRegrets/WL:{round(total_mean_iregret/total_wl_types, 5)}")
            log.info("")

        log.info('WL=WorkLoad type, i.e. "15 1000000 12 uniform 0.99" stands for 15 Tables, 100k rows, 12 clients, uniform access to data and SLA at 10ms (0.99)')

    # def test(self):
    #     print(self.unwrapped._idelta_list)

    def getGraph(self, title):
        #results = dict(sorted(results.items())) #, key=lambda k: k[0].split(maxsplit=1),reverse=False)) # Sort on first field of each line
        #assert len(reward_per_episode) == episodes_max  
        assert len(self._regret_per_episode) == len(self._iregret_per_episode)

        total_ep = self.unwrapped.cur_episode 
        self._reg_performance = round(self._regret_per_episode[:total_ep].sum(),3)
        self._ireg_performance = round(self._iregret_per_episode[:total_ep].sum(),3)
        self._rew_performance = round(self._reward_per_episode[:total_ep].sum(),3)
    
        partial = '/!\\' if self._partial else '' 

        # self.unwrapped._idelta_list
        # Initialize the two lists with zeros

        # seperate and pad the lists into ok and violation
        #less_than_zero_list = [0] * len(self.unwrapped._idelta_list)
        #greater_or_equal_zero_list = [0] * len(self.unwrapped._idelta_list)
        
        # Populate the lists according to the condition
        #for i, value in enumerate(self.unwrapped._idelta_list):
        #    if value < 0:
        #        less_than_zero_list[i] = abs(value)
        #    else:
        #        greater_or_equal_zero_list[i] = value

        if self.unwrapped._on_terminate >= 0:
            title_x=f"{total_ep+1} Episodes - {(total_ep+1)//2} Workloads - {self.unwrapped.max_steps_per_episode} steps per Workload -  #EpSuccess: {self.unwrapped.terminated_count}"
        else:
            title_x=f"{total_ep+1} Episodes - {(total_ep+1)//2} Workloads - {self.unwrapped.max_steps_per_episode} steps per Workload -  No termination cond"

        graph = GraphPx(x=[idx for idx in range(total_ep)], title=title, title_x=title_x)

        graph.addCurve(f"Regret cumul{partial}:{self._reg_performance} (CRew:{self._rew_performance})", self._regret_per_episode[:total_ep].cumsum())
        graph.addCurve(f"Total IRegret cumul{partial}:{self._ireg_performance} (CRew:{self._rew_performance})", self._iregret_per_episode[:total_ep].cumsum())
        # graph.addCurve(f"Negative IRegret cumul{partial}:{sum(less_than_zero_list)} (CRew:{self._rew_performance})", np.array(less_than_zero_list)[:total_ep].cumsum())
        # graph.addCurve(f"Positive IRegret cumul{partial}:{sum(greater_or_equal_zero_list)} (CRew:{self._rew_performance})", np.array(greater_or_equal_zero_list)[:total_ep].cumsum())
        graph.setPieData(self.unwrapped._idelta_list, 0.01)
        return graph

    def getPerfMeterGraph(self, name="", baseline_perf_meter=None):
        if self.perf_meter is None:
            return None

        idelta_pos, idelta_neg, bufsz_vol_pos, bufsz_vol_neg, violations = self.perf_meter.getIndicators()
        total_ep = self.perf_meter.getSessionEpisodesCount()
        assert total_ep == (self.unwrapped.cur_episode +1), f"ERROR, discrepancy in episodes count perf_meter={total_ep} env={self.unwrapped.cur_episode +1}"
        ram_outflow_pos = bufsz_vol_pos.sum()/total_ep
        ram_outflow_neg = bufsz_vol_neg.sum()/total_ep

        episodes_names = self.perf_meter.getSessionEpisodesIds()

        if baseline_perf_meter:
            bl_idelta_pos, bl_idelta_neg, bl_bufsz_vol_pos, bl_bufsz_vol_neg, bl_violations = baseline_perf_meter.getIndicators()
            bl_total_ep = baseline_perf_meter.getSessionEpisodesCount()
            assert total_ep == bl_total_ep, f"ERROR, discrepancy in episodes count perf_meter={total_ep} baseline_perf_meter={bl_total_ep}"
            bl_sla_violations_msg = f' vs {baseline_perf_meter.name} #: {bl_violations.sum()}'
            bl_ram_outflow_pos = bl_bufsz_vol_pos.sum()/bl_total_ep
            bl_ram_outflow_neg = bl_bufsz_vol_neg.sum()/bl_total_ep
            ram_outflow_pos = ram_outflow_pos - bl_ram_outflow_pos
            ram_outflow_neg = ram_outflow_neg - bl_ram_outflow_neg
            ram_outflow_msg = f"less {baseline_perf_meter.name}"
            bufsz_vol_pos = bufsz_vol_pos - bl_bufsz_vol_pos
            bufsz_vol_neg = bufsz_vol_neg - bl_bufsz_vol_neg
        else:
            ram_outflow_msg = ""
            bl_sla_violations_msg = ""

        title_x=f"{total_ep} Ep - {total_ep//2} WL - {self.unwrapped.max_steps_per_episode} steps per WL"
        if baseline_perf_meter:
            perf = baseline_perf_meter.getSessionPerformanceMultiObj()
            perf = tuple(map(lambda x: round(x, 2), perf))
            title_x += f' {baseline_perf_meter.name}: {perf}'
        graph = GraphPx(x=episodes_names, title=f"PerfMeter-{name}", title_x=title_x)
        graph.addCurve(f"Reg+iDelta+ Cumul:{round(idelta_pos.sum(),3)}", idelta_pos)
        graph.addCurve(f"Reg+iDelta- Cumul:{round(idelta_neg.sum(),3)}", idelta_neg)
        graph.addCurve(f"#SLA Violations:{violations.sum()}{bl_sla_violations_msg}", violations)
        graph.addCurve(f"RAMOutFlow+/Ep {ram_outflow_msg} (MeanMB/Ep):{round(ram_outflow_pos,1)}", bufsz_vol_pos)
        graph.addCurve(f"RAMOutFlow-/Ep {ram_outflow_msg} (MeanMB/Ep):{round(ram_outflow_neg,1)}", bufsz_vol_neg)

        if baseline_perf_meter:
            graph.addCurve(f"{baseline_perf_meter.name} Reg+iDelta+ Cumul:{round(bl_idelta_pos.sum(),3)}", bl_idelta_pos)
            graph.addCurve(f"{baseline_perf_meter.name} Reg+iDelta- Cumul:{round(bl_idelta_neg.sum(),3)}", bl_idelta_neg)

        return graph

    def save_regrets(self, algo_name = '', seed = 0, train_or_test = "train"):
        filename = f"/home/cloud/poc_fanfan/poc/auto-tuning/trainer/_app/bandits/diverse_graphics/lists_data/regret_per_episode_{algo_name}_{seed}_{train_or_test}.pkl"
        # Save the list to a file
        with open(filename, 'wb') as file:
            pickle.dump(self._regret_per_episode, file)
        filename = f"/home/cloud/poc_fanfan/poc/auto-tuning/trainer/_app/bandits/diverse_graphics/lists_data/iregret_per_episode_{algo_name}_{seed}_{train_or_test}.pkl"
        # Save the list to a file
        with open(filename, 'wb') as file:
            pickle.dump(self._iregret_per_episode, file)
        filename = f"/home/cloud/poc_fanfan/poc/auto-tuning/trainer/_app/bandits/diverse_graphics/lists_data/idelta_list_{algo_name}_{seed}_{train_or_test}.pkl"
        # Save the list to a file
        with open(filename, 'wb') as file:
            pickle.dump(self.unwrapped._idelta_list, file)

    def graphPerWorkload(self, title="Regret/workload", title_x=None):
        results = self._results
        
        graph_x = np.array([wl for wl in results.keys()])
        n_wl = len(graph_x)
        pref_title_x=f"{self.unwrapped.cur_episode+1} Episodes - {n_wl} Workloads - {self.unwrapped.max_steps_per_episode} steps per Workload"

        if title_x is None:
            title_x = pref_title_x
        else:
            title_x = pref_title_x+" "+title_x


        return GraphPx(graph_x, title, title_x)

    def resultPerWorkload(self, result_field="MeanRegret"):
        results = self._results
        return np.array([res[result_field] for wl, res in results.items()])
    
    def getRegretPerformance(self):
        return self._reg_performance
        
    def getIRegretPerformance(self):
        return self._ireg_performance

    def getRewardPerformance(self):
        return self._rew_performance


