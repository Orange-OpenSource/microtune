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

from bandits.policy import CtxPolicy
from bandits.actions import Actions

# Create class object for a single linear ucb disjoint arm
class linucb_disjoint_arm():
    
    def __init__(self, arm_index, d, alpha):
        
        # Track arm index
        self.arm_index = arm_index
        
        # Keep track of alpha
        self.alpha = alpha
        
        # A: (d x d) matrix = D_a.T * D_a + I_d. 
        # The inverse of A is used in ridge regression 
        self.A = np.identity(d)
        
        # b: (d x 1) corresponding response vector. 
        # Equals to D_a.T * c_a in ridge regression formulation
        self.b = np.zeros([d,1])
        self._d = d
        
    def calc_UCB(self, x_array):
        # Find A inverse for ridge regression
        #A_inv = np.linalg.pinv(self.A) # pinv to get rid off matrix with Determinant=0 
        A_inv = np.linalg.inv(self.A)
        #print(self.A)
        #assert self._d == len(x_array), f"Invalid context len: {len(x_array)}, expected:{self._d}"
        assert self._d == x_array.size, f"Invalid context len: {x_array.size}, expected:{self._d}"
        
        # Perform ridge regression to obtain estimate of covariate coefficients theta
        # theta is (d x 1) dimension vector
        self.theta = np.dot(A_inv, self.b)

        # Reshape covariates input into (d x 1) shape vector
        x = x_array.reshape([-1,1])
        
        # Find ucb based on p formulation (mean + std_dev) 
        # p is (1 x 1) dimension vector
        confid = np.dot(x.T, np.dot(A_inv,x))
        #assert np.min(confid) >= 0, f'ERROR, Arm:{self.arm_index} np.dot(x.T, np.dot(A_inv,x))={confid} is <0 with x.T:{x.T} np.dot(A_inv,x)={np.dot(A_inv,x)} A_inv:{A_inv} x:{x} CTX:{x_array}'
        assert np.min(confid) >= 0, f'ERROR, Arm:{self.arm_index} confid={confid} is <0 LENCTX:{x_array.size}'
        p = np.dot(self.theta.T,x) + self.alpha * np.sqrt(confid)
        
        return p
    
    def reward_update(self, reward, x_array):
        # Reshape covariates input into (d x 1) shape vector
        x = x_array.reshape([-1,1])
        
        # Update A which is (d * d) matrix.
        self.A += np.dot(x, x.T)
        assert len(self.A)==self._d and len(self.A[0])==self._d, f"Error, self.A structure has changed {len(self.A)}x{len(self.A[0])} vs {self._d}x{self._d}"
        
        # Update b which is (d x 1) vector
        # reward is scalar
        self.b += reward * x 
        assert len(self.b)==self._d and len(self.b[0])==1, f"Error, self.b structure has changed {len(self.A)}x{len(self.A[0])} vs {self._d}x1"
        
# KFOOFW algo from  https://github.com/kfoofw/bandit_simulations/blob/master/python/contextual_bandits/notebooks/LinUCB_disjoint.ipynb
class LinUCBPolicy_kfoofw(CtxPolicy):
    
    def __init__(self, actions: Actions | tuple = (-1, 1), ctx=[], learning_rate=0.8, seed=None, use_tips=True):
        super().__init__(actions, True, ctx=ctx, use_tips=use_tips, seed=seed)    # Use tips for LinUCB
        self.linucb_arms = [linucb_disjoint_arm(arm_index = i, d = self._nfeatures, alpha = learning_rate) for i in range(self._k_arms)]
        
    def _getPolicyFeatures(self, x_array):
        return  np.asmatrix(super()._getPolicyFeatures(x_array))
    
    # Overrides virtual method of ABC CtxPolicy class
    def select_arm(self, x_array, deterministic=False, debug=False):
        # Initiate ucb to be 0
        highest_ucb = -np.inf
        
        # Track index of arms to be selected on if they have the max UCB.
        candidate_arms = []
        
        for arm_index in range(self.getArmsCount()):
            # Calculate ucb based on each arm using current covariates at time t
            arm_ucb = self.linucb_arms[arm_index].calc_UCB(self._getPolicyFeatures(x_array))
            #print(f'{arm_index} Theta:{self.linucb_arms[arm_index].theta} b:{self.linucb_arms[arm_index].b}')
            
            # If current arm is highest than current highest_ucb
            if arm_ucb > highest_ucb:         
                # Set new max ucb
                highest_ucb = arm_ucb
                
                # Reset candidate_arms list with new entry based on current arm
                candidate_arms = [arm_index]

            # If there is a tie, append to candidate_arms
            if arm_ucb == highest_ucb:
                candidate_arms.append(arm_index)
        
        # Choose based on candidate_arms randomly (tie breaker)
        assert len(candidate_arms)> 0, f'ERROR: #Arms:{self.getArmsCount()} ARMIDX:{arm_index} ARMUCB:{arm_ucb} HUCB:{highest_ucb} LENCTX:{len(x_array)} CTX{x_array}'
        #chosen_arm = np.random.choice(candidate_arms if len(candidate_arms) > 0 else [1]) # if len(candidate_arms) > 0 else np.random.randint(0, self.getArmsCount())
        chosen_arm = np.random.choice(candidate_arms)
        
        return chosen_arm
    
    def update(self, arm_index, reward, x_array, next_x_array=None):
        self.linucb_arms[arm_index].reward_update(reward, self._getPolicyFeatures(x_array))
