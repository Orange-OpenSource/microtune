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
import torch
import math
import copy
#from itertools import combinations
#from itertools import product
device = torch.device('cpu')


# Typically for us:
# hidden_dim = [ 100, 30]
class NeuralLinUCB():
    def __init__(self,memory = -1,nArms = 3, nFeature = 10, lambd = 25,hidden_dim = 1, beta = 1, H_q = 100, interT = 200, et = 0.0001):
        self._nFeature = nFeature
        self.memory = memory
        self._nArms = nArms
        self.lambd = lambd
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.max_H_q = H_q
        self.H_q = self.max_H_q
        self.interT = interT
        self.init_et = et
        self.reset()
        
    def export_model(self):
        return (self.W0,self.W,self._A,self.bb,self.theta)
    
    def import_model(self,model,envLoad = None,agentLoad = None):
        self.W0 = model[0]
        self.W = model[1]
        self._A = model[2]
        self.bb = model[3]
        self.theta = model[4]
        
    def INI(self,dim):
        w = []
        total_dim = 0
        for i in range(0, len(dim) - 1):
            if i < len(dim) - 2:
                temp = np.random.randn(dim[i + 1], dim[i]) / np.sqrt(dim[i + 1])
                temp = np.kron(np.eye(2, dtype=int), temp)
                temp = torch.from_numpy(temp).to(device)
                w.append(temp)
                total_dim += dim[i + 1] * dim[i] *4
            else:
                temp = np.random.randn(dim[i + 1], dim[i]) / np.sqrt(dim[i])
                temp = np.kron([[1, -1]], temp)
                temp = torch.from_numpy(temp).to(device)
                w.append(temp)
                total_dim += dim[i + 1] * dim[i]*2
        return w, total_dim
    
    def FUNC_FE(self,x, W):
        depth = len(W)
        output = x
        for i in range(0, depth - 1):
            output = torch.mm(W[i], output)
            output = output.clamp(min=0)
        output = output * math.sqrt(W[depth - 1].size()[1])
        return output
    
    def GRAD_LOSS(self,X, Y, W, THETA):
        depth = len(W)
        num_sample = Y.shape[0]
        loss = []
        grad = []
        relu = []
        output = X
        loss.append(output)
        for i in range(0, depth - 1):
            output = torch.mm(W[i], output)
            relu.append(output)
            output = output.clamp(min=0)
            loss.append(output)

        THETA_t = torch.transpose(THETA,0,1).view(num_sample, 1, -1)
        output_t = torch.transpose(output,0,1).view(num_sample, -1, 1)
        output = torch.bmm(THETA_t, output_t).squeeze().view(1,-1)

        loss.append(output)
        feat = self.FUNC_FE(X, W)
        feat_t = torch.transpose(feat, 0, 1).view(num_sample, -1, 1)
        output_t = torch.bmm(THETA_t, feat_t).squeeze().view(1, -1)

        #### backward gradient propagation
        back = output_t - Y
        back = back.double()
        grad_t = torch.mm(back, loss[depth - 1].t())
        grad.append(grad_t)

        for i in range(1, depth):
            back = torch.mm(W[depth - i].t(), back)
            back[relu[depth - i - 1] < 0] = 0
            grad_t = torch.mm(back, loss[depth - i - 1].t())
            grad.append(grad_t)
        grad1 = []
        for i in range(0, depth):
            grad1.append(grad[depth - 1 - i] * math.sqrt(W[depth - 1].size()[1]) / len(X[0, :]))

        if (grad1[0] != grad1[0]).any():
            log.error('nan found in grad loss')
            import sys; sys.exit('nan found')
        return grad1
    
    def loss(self,X, Y, W, THETA):
        #### total loss
        num_sample = len(X[0, :])
        output = self.FUNC_FE(X, W)
        THETA_t = torch.transpose(THETA, 0, 1).view(num_sample, 1, -1)
        output_t = torch.transpose(output, 0, 1).view(num_sample, -1, 1)
        output_y = torch.bmm(THETA_t, output_t).squeeze().view(1, -1)
        summ = (Y - output_y).pow(2).sum() / num_sample
        return summ

    def TRAIN_SE(self,X, Y, W_start, T, et, THETA, H):
        W = copy.deepcopy(W_start)
        num_sample = H
        X = X[:, -H:]
        Y = Y[-H:]
        THETA = THETA[:, -H:]

        prev_loss = 1000000
        prev_loss_1k = 1000000
        for i in range(0, T):
            grad = self.GRAD_LOSS(X, Y, W, THETA)
            if (grad[0] != grad[0]).any():
                log.error('nan found in train se')
            for j in range(0, len(W)-1):
                W[j] = W[j] - et * grad[j]
            curr_loss = self.loss(X, Y, W, THETA)
            if i % 100 == 0:
                log.debug('------',curr_loss)
                if curr_loss > prev_loss_1k:
                    et = et * 0.1
                    log.debug('lr/10 to', et)

                prev_loss_1k = curr_loss

            # early stopping
            if abs(curr_loss - prev_loss) < 1e-6:
                break
            prev_loss = curr_loss
        return W
    
    def init_H_q(self):
        self.H_q = 1
        self.t = 0
    
    def UCB(self,A, phi):
        _A_inv = np.linalg.inv(A)
        return torch.from_numpy(np.sqrt(np.dot(np.transpose(phi),np.dot(_A_inv,phi)))).to(device)

    def TRANS(self,c, a, arm_size):
        dim = len(c)
        action = np.zeros(arm_size)
        action[a] = 1
        c_final = np.append(c, action)
        c_final = torch.from_numpy(c_final).to(device)
        c_final = c_final.view((len(c_final), 1))
        c_final = c_final.repeat(2, 1)
        return c_final
    

        
    def select(self, context, tie_break_mode = "random"):
        ucb = []
        value = []
        confidence = []
        self.bphi = []
        for a in range(0, self._nArms):
            temp = self.TRANS(context, a, self._nArms)
            self.bphi.append(temp)
            feat = self.FUNC_FE(temp, self.W)
            value.append(torch.mm(self.theta.view(1,-1), feat).detach().cpu().numpy()[0][0])
            confidence.append(self.beta * self.UCB(self._A, feat).detach().cpu().numpy()[0][0])
            ucb.append(torch.mm(self.theta.view(1,-1), feat) + self.beta * self.UCB(self._A, feat))

        # use round-robin, #initial_pull = 3
        if self.t< 3*self._nArms:
            a_choose = self.t % self._nArms
        else:
            a_choose = ucb.index(max(ucb))
        best_action = a_choose
        return int(best_action), np.array(value), np.array(confidence)

    #Not necessary
    def external_select(self, context, tie_break_mode = "random"):
        ucb = []
        value = []
        confidence = []
        self.bphi = []
        for a in range(0, self._nArms):
            temp = self.TRANS(context, a, self._nArms)
            self.bphi.append(temp)
            feat = self.FUNC_FE(temp, self.W)
            value.append(np.dot(self.theta.view(1,-1),feat)[0][0])
            confidence.append(self.beta * self.UCB(self._A, feat).detach().cpu().numpy()[0][0])
            ucb.append(np.add(np.dot(self.theta.view(1,-1),feat), self.beta * self.UCB(self._A, feat)))

        # use round-robin, #initial_pull = 3
        if self.t< 3*self._nArms:
            a_choose = self.t % self._nArms
        else:
            a_choose = ucb.index(max(ucb))
        best_action = a_choose
        return np.array(ucb), np.array(value), np.array(confidence)    
    
    def observe(self, played_arm, context, next_context, reward, update = False):
        if np.mod(self.t, 10) == 0:
            log.debug("Observe step:", self.t)
        if self.memory == -1:
            if np.mod(self.t, self.H_q) == 0:
                log.debug("clean")
                self.CONTEXT_action = []
                self.REWARD_action = []
                self.CONTEXT_action = self.bphi[played_arm]
                self.REWARD_action = torch.tensor([reward], device=device, dtype=torch.double)
            else:
                self.CONTEXT_action = torch.cat((self.CONTEXT_action, self.bphi[played_arm]), 1)
                self.REWARD_action = torch.cat((self.REWARD_action, torch.tensor([reward], device=device, dtype=torch.double)), 0)
        else:
            if len(self.CONTEXT_action) == 0:
                
                self.CONTEXT_action = self.bphi[played_arm]
                self.REWARD_action = torch.tensor([reward], device=device, dtype=torch.double)
            else:
                self.CONTEXT_action = torch.cat((self.CONTEXT_action, self.bphi[played_arm]), 1)
                self.REWARD_action = torch.cat((self.REWARD_action, torch.tensor([reward], device=device, dtype=torch.double)), 0)
            if self.CONTEXT_action.size()[1] > self.memory:
                
                self.CONTEXT_action = self.CONTEXT_action[:,1:]
                self.REWARD_action = self.REWARD_action[1:] 

        context_func_fe = self.FUNC_FE(self.bphi[played_arm], self.W)
        self._A = self._A + np.dot(context_func_fe,np.transpose(context_func_fe))
        self.bb = self.bb + reward * context_func_fe
        #self.theta, LU = torch.solve(self.bb,self._A)
        self.theta = torch.linalg.solve(self._A,self.bb)
        if self.memory == -1:
            if np.mod(self.t, self.H_q) == 0:
                
                self.THETA_action = []
                self.THETA_action = self.theta.view(-1,1)
            else:
                self.THETA_action = torch.cat((self.THETA_action, self.theta.view(-1,1)), 1)
            
        else:
            if len(self.THETA_action) == 0:
                self.THETA_action = self.theta.view(-1,1)
            else:
                self.THETA_action = torch.cat((self.THETA_action, self.theta.view(-1,1)), 1)
            if self.THETA_action.size()[1] > self.memory:
                self.THETA_action= self.THETA_action[:,1:]

        if np.mod(self.t, self.H_q) == self.H_q-1:
            log.debug("Call TRAIN SE from observe")
            if self.memory == -1:
                self.W = self.TRAIN_SE(self.CONTEXT_action, self.REWARD_action, self.W0, self.interT, self.et, self.THETA_action, self.H_q)
            else:
                self.W = self.TRAIN_SE(self.CONTEXT_action, self.REWARD_action, self.W0, self.interT, self.et, self.THETA_action, self.memory)
        self.t+=1
        
    def reset(self):
        self.et = self.init_et
        self.t = 0
        hid_dim_lst = self.hidden_dim
        log.debug("hid_dim_lst = self.hidden_dim", self.hidden_dim)
        log.debug("self.hidden_dim[-1]:", self.hidden_dim[-1])
        log.debug("dim_second_last = self.hidden_dim[-1] *2:", self.hidden_dim[-1] *2)
        dim_second_last = self.hidden_dim[-1] *2
        log.debug("self._nFeature:", self._nFeature)
        log.debug("self._nFeature + self._nArms:",self._nFeature + self._nArms)
        log.debug("dim_for_init = [self._nFeature + self._nArms] + hid_dim_lst + [1]:",[self._nFeature + self._nArms] + hid_dim_lst + [1])
        dim_for_init = [self._nFeature + self._nArms] + hid_dim_lst + [1]
        self.W0, total_dim = self.INI(dim_for_init)
        self._A = self.lambd * torch.eye(dim_second_last, device=device, dtype=torch.double)
        self.bb = torch.zeros(self._A.size()[0], device=device, dtype=torch.double).view(-1, 1)
        self.theta = np.random.randn(dim_second_last, 1) / np.sqrt(dim_second_last)
        self.theta = torch.from_numpy(self.theta).to(device)
        log.debug("self.theta.size():", self.theta.size())
        self.THETA_action = []
        self.CONTEXT_action = []
        self.REWARD_action = []
        self.W = copy.deepcopy(self.W0)
        log.debug("END INIT")
        



from bandits.policy import CtxPolicy
from bandits.actions import Actions

import logging
# A logger for this file
log = logging.getLogger(__name__)


class NeuralLinUCBPolicy(CtxPolicy):   
    def __init__(self, actions: Actions | tuple = (-1, 1), ctx=[], beta=1, hidden_dim1=100, hidden_dim2=30, H_q=100, seed=None, use_tips=True):
        super().__init__(actions, ctx=ctx, use_tips=use_tips)   # Use tips, features +1 ?
        self.neurallinucb = NeuralLinUCB(memory = -1, nArms = self._k_arms, nFeature = self._nfeatures, 
                                         lambd=1, hidden_dim = [hidden_dim1, hidden_dim2], beta=beta, H_q=H_q, interT = 200, et = 0.0001)
        #self.dims = -1

#    def _getPolicyFeatures(self, x_array):
#        return  np.asmatrix(super()._getPolicyFeatures(x_array))

    # Overrides virtual method of ABC CtxPolicy class
    def select_arm(self, x_array, deterministic=False, debug=False):
        x_array = self._getPolicyFeatures(x_array)
        # if self.dims == -1:
        #     self.dims = x_array.shape
        #     new_shape = self.dims
        # else:
        #     new_shape = x_array.shape

        # assert new_shape == self.dims, f'SELECT Shape error new:{new_shape} old:{self.dims}'
        #log.info(f'SELECT CONTEXT: {x_array} _nFeature: {self.neurallinucb._nFeature}')

        best_arm, value, confidence = self.neurallinucb.select(x_array)
        return best_arm
    
    def update(self, arm_index, reward, x_array, next_x_array):
        x_array = self._getPolicyFeatures(x_array)
        
        #new_shape = x_array.shape
        #assert new_shape == self.dims, f'UPDATE Shape error new:{new_shape} old:{self.dims}'

        next_x_array = self._getPolicyFeatures(next_x_array)
        #log.info(f'UPDATE CONTEXT: {x_array} _nFeature: {self.neurallinucb._nFeature}')
        #log.info(f'REWARD: {reward}')
        self.neurallinucb.observe(played_arm=arm_index, context=x_array, next_context=next_x_array, reward=reward)

        