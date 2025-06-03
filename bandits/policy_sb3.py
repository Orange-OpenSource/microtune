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
import os
import time

from bandits.policy import CtxPolicy
from bandits.actions import Actions

from stable_baselines3.common.base_class import BaseAlgorithm

import gymnasium as gym
import torch


# See how to change the NN layrs: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html


# A virtual class that can NOT be instanciated. See implementation below for PPO, DDPG, A2C, DQN, SAC ...
class SB3Policy(CtxPolicy):
    def __init__(self, actions: Actions | tuple, discrete_arms_mode=True, ctx=[], model_class=None, qvf=None, model_args=None):
        super().__init__(actions, discrete_arms_mode, ctx=ctx, use_tips=False)
        self.model = None
        self._model_class = model_class
        #self._model_args = model_args
        self._qvf = qvf
        self._model_args = self.handle_hydra_arch_nn(model_args)

    def handle_hydra_arch_nn(self, model_args: dict = {}):
        policy_kwargs = model_args.get("policy_kwargs")
        if policy_kwargs == "null":
            # Will Use SB3 default 
            model_args.pop("policy_kwargs")
        elif policy_kwargs:
            # Will Use configurated NN arch -> rebuild, as expected, policy_kwargs fom config
            net_arch = model_args["policy_kwargs"]["net_arch"]
            if hasattr(net_arch, '__iter__') and self._qvf:
                net_arch = model_args["policy_kwargs"]["net_arch"]
            else:
                pi = model_args["policy_kwargs"]["net_arch"]["pi"]
                qvf = model_args["policy_kwargs"]["net_arch"]["qvf"]
                net_arch = dict(net_arch={"pi": pi, self._qvf: qvf})
            activation_fn= policy_kwargs.get("activation_fn")
            # Replace policy_kwargs as well...
            model_args["policy_kwargs"] = dict(net_arch=net_arch) 
            if activation_fn:
                model_args["policy_kwargs"]["activation_fn"] = activation_fn.__class__ 
            print(f'SB3Policy set policy_kwargs: { model_args["policy_kwargs"]}')
        
        return model_args

    def initWithEnv(self, env: gym.Wrapper):
        super().initWithEnv(env)
        if env:
            if self.model is None:
                self.model = self._model_class(env=env, **self._model_args)
                self.renameFromType(self.model, origin='SB3')
            self.model.set_env(env, force_reset=True)

    def select_arm(self, context, deterministic=False, debug=False):
        if debug:
            print(f"ContextDict:{self.getCtxDict(context)}")

        action, _states = self.model.predict(observation=context, deterministic=deterministic)

        return action 

    def dataToSave(self):
        model = self.model
        self.model = None # Do NOT save model (because of local reference error with pickle.dump())
        data = super().dataToSave()
        self.model = model

        sb3modelzip = f"{time.time_ns()}_model-{self.name}.zip"
        self.model.save(sb3modelzip)
        data.append(sb3modelzip) 

        return data
    
    def restoreData(self, data=[]):
        data = super().restoreData(data)
        sb3modelzip = data.pop(0)
        self._loadModel(sb3modelzip)
        os.remove(sb3modelzip)
        return data
    
    def _loadModel(self, sb3modelzip):
        self.model = self._model_class.load(sb3modelzip, force_reset=True)
        self.renameFromType(self.model, origin='SB3')



from stable_baselines3 import PPO, SAC, DDPG, DQN, A2C

# Manage either Continous or  Discrete actions, default is continous
class SB3PolicyPPO(SB3Policy):
    def __init__(self, actions: Actions | tuple = (-1, 1), ctx=[], discrete_arms_mode=False, **kwargs):
        super().__init__(actions, discrete_arms_mode, ctx, PPO, "vf", kwargs)
    
# Manage Continous actions only
class SB3PolicySAC(SB3Policy):
    def __init__(self, actions: Actions | tuple = (-1, 1), ctx=[], **kwargs):
        super().__init__(actions, False, ctx, SAC, "qf", kwargs)

# Manage Continous actions only
class SB3PolicyDDPG(SB3Policy):
    def __init__(self, actions: Actions | tuple = (-1, 1), ctx=[], **kwargs):
        super().__init__(actions, False, ctx, DDPG, "qf", kwargs)

# Manage Discrete actions only
class SB3PolicyDQN(SB3Policy):
    def __init__(self, actions: Actions | tuple = (-1, 1), ctx=[], **kwargs):
        super().__init__(actions, True, ctx, DQN, "vf", kwargs)

# Manage Discrete actions only
class SB3PolicyA2C(SB3Policy):
    def __init__(self, actions: Actions | tuple = (-1, 1), ctx=[], **kwargs):
        super().__init__(actions, True, ctx, A2C, "vf", kwargs)

