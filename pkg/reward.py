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

from pkg.datasource.dataset import ADBMSDataSetEntryContextSelector
from pkg.actions import Actions

# Note: It is possible to Disable ALPHA and BETA effect by setting them to -1. BETA apply to reward on ARM STAY. ALPHA apply to both UP with ALPHA factor and DOWN with (1-ALPHA) factor.
# strengthened_action: A reward option to use ARMs UP & DOWN index as a (booster) coeficient in the rewarding computation. Do not apply to ARM STAY, as its index is always 0
class RewardNA():
    def __init__(self, action_minmax=(-1,1), alpha=-1, beta=-1, strengthened_action=False):
        self.actions = Actions(action_minmax)
        self._action_rewards = np.zeros(self.actions.count(), dtype=float)     # All Current action's values => reward
        self.renameFromType()
        if alpha < 0:
            alpha = 1
            alpha_inv = 1
        else:
            alpha_inv = 1. - alpha

        if beta < 0:
            beta = 1

        #assert alpha >= 0 and alpha <= 1, f'Invalid reward alpha value={alpha} must be in [0,1]'
        #assert beta >= 0 and beta <= 1, f'Invalid reward beta value={beta} must be in [0,1]'
        self._alpha = alpha
        self._alpha_inv = alpha_inv
        self._beta = beta

        # When strengthened_action is True, the effect of UP/DOWN actions is strenghtened (i.e. multiplied by 2)
        self._strengthened_action = strengthened_action
        if strengthened_action:
            self._down_factor = abs(self.actions.min())
            self._up_factor = self.actions.max()
        else:
            self._down_factor = 1
            self._up_factor = 1


    def renameFromType(self, obj=None, origin=''):
        if obj is None:
            obj = self
        name = type(obj).__name__.replace('ADBMSBufferCache', '')
        self.name = f"{origin}{name}{self.actions.minMax()}"

    def _getRewReg(self, action):
        return 0, 0
    
    # Find out max action value and depending on its state (Violation OK or Over), apply a lambda function
    def actionMax2Lambda(self, under=lambda:"VIOLATION", ok=lambda:"OK", over=lambda:"OVER"):        
        return "NA"
    
    def action2Name(self, action):
        return self.actions.name(int(action))

    def reset(self, ds: ADBMSDataSetEntryContextSelector):
        pass

    def compute(self, ds: ADBMSDataSetEntryContextSelector):
        pass
    
    # Get reward's value for each action (from less action value to the greatest, i.e. from DOWN to UP)
    def getRewards4Actions(self):
        return self._action_rewards

    def get(self, action: int):
        return 0, 0, 1


# Implementation for 3 actions. To be upgraded later for more actions.....
# Possible actions min-max values are N,M where... (TBC)
class RewardDownStayUp(RewardNA):
    def __init__(self, action_minmax=(-1,1), alpha=-1, beta=-1, strengthened_action=False):
        super().__init__(action_minmax, alpha, beta, strengthened_action=strengthened_action)

    # Return reward, regret
    def _getRewReg(self, action):
       # ai Action's Index
       ai = self.actions.arm(int(action))
       rew = self._action_rewards[ai]
       reg = self._action_rewards.max() - rew
       return rew, reg
    
    # Find out max action value and depending on its state (Violation OK or Over), apply a lambda function
    def actionMax2Lambda(self, under=lambda:"VIOLATION", ok=lambda:"OK", over=lambda:"OVER"):        
        arm_max = self._action_rewards.argmax() # Which arm give the best reward ?
        if arm_max < self.actions.armStay(): # 
            return over()
        elif arm_max == self.actions.armStay(): # Stay
            return ok()

        return under() # Up

    def compute(self, ds: ADBMSDataSetEntryContextSelector):
        assert False, "Implementation Error: Reward.compute() Reward method not implemented."

    # Default method to get reward and regret. Must be overriden if a probability has to be computed
    # Action-1 => Decreases buffer
    # Action0  => Stay at current value
    # Action1  => increases buffer 
    # Retrun Reward , Regret and Action's Probability (not computed right now, always 1) 
    def get(self, action: int):
        #action = int(action)
        min, max = self.actions.minMax()
        assert action>=min and action <= max, f"Action {action} is out of action space [{min},{max}]"
        rew, reg = self._getRewReg(action)
#                    parm = 0.0000000001 if parm == 0 else parm
#                    rew /= parm

        if action < 0: # DOWN
            rew *= self._alpha_inv * self._down_factor
        elif action == 0: # STAY
            rew *= self._beta
        else: # UP
            rew *= self._alpha * self._up_factor

        return rew, reg, 1 



class ADBMSBufferCacheRewardContinousV2(RewardDownStayUp):
    def __init__(self, action_minmax=(-1,1), alpha=-1, beta=-1):
        super().__init__(action_minmax=action_minmax, alpha=alpha, beta=beta)            
        
    def compute(self, ds: ADBMSDataSetEntryContextSelector):
        def rew(idx, bufincr):
            iperf, idelta, threshold = ds.getIPerfIndicators()
            down = stay = up = 0
 
            if idelta >= 0:
                dd = idelta/(1-threshold)
                down = dd #**2
                stay = (1-dd)**8
                up   = stay-1
            else:
                # VIOLATION
    #            dd = -max(1, math.log(0.1+abs(idelta/threshold))+1)
                dd = idelta/threshold
                down = dd
                stay = dd 
                up   = -dd*1.2

            if bufincr == 0:
                self._action_rewards[idx]=stay
            elif bufincr < 0:
                self._action_rewards[idx]=down
            else:
                self._action_rewards[idx]=up
        
        ds.applyLambda2actions(self.actions.vals(), rew)

class ADBMSBufferCacheRewardContinousV3(RewardDownStayUp):
    def __init__(self, action_minmax=(-1,1), alpha=-1, beta=-1):
        super().__init__(action_minmax=action_minmax, alpha=alpha, beta=beta)            
        
    def compute(self, ds: ADBMSDataSetEntryContextSelector):
        def rew(idx, bufincr):
            iperf, idelta, threshold = ds.getIPerfIndicators()
            down = stay = up = 0
 
            # accelerator: dd = + or - min(1, math.log(0.1+dd)+1) => doesn't work well...
            if idelta >= 0:
                dd = idelta/(1-threshold)
                down = dd
                stay = (1-dd)**8
                up   = stay-1
                #stay *= 2
            else:
                # VIOLATION
                dd = idelta/threshold
                down = max(-1, dd*1.5)
                stay = dd 
                up   = -down

            if bufincr == 0:
                self._action_rewards[idx]=stay
            elif bufincr < 0:
                self._action_rewards[idx]=down
            else:
                self._action_rewards[idx]=up
        
        ds.applyLambda2actions(self.actions.vals(), rew)
        #print(f'Actions vals:{self.actions.vals()} Rewards:{self._action_rewards}')

# V3 + 2 Tricks (see below)
class ADBMSBufferCacheRewardContinousV3E(RewardDownStayUp):
    def __init__(self, action_minmax=(-1,1), alpha=-1, beta=-1):
        super().__init__(action_minmax=action_minmax, alpha=alpha, beta=beta)            
        
    def compute(self, ds: ADBMSDataSetEntryContextSelector):
        iperf, cur_idelta, threshold = ds.getIPerfIndicators()

        def rew(idx, bufincr):
            iperf, idelta, threshold = ds.getIPerfIndicators()
            down = stay = up = 0
 
            # accelerator: dd = + or - min(1, math.log(0.1+dd)+1) => doesn't work well...
            if idelta >= 0:
                dd = idelta/(1-threshold)
                down = dd
                stay = (1-dd)**8
                #up   = stay-1
                up = stay-1 if cur_idelta >= 0. else 1-stay  # Trick1 HERE! to manage the case where UP is downgraded when the current state is just before the limit
            else:
                # VIOLATION
                dd = idelta/threshold
                down = max(-1, dd*1.5)
                stay = down               # Trick2 HERE! Enforce bad choice 
                up   = -down

            if bufincr == 0:
                self._action_rewards[idx]=stay
            elif bufincr < 0:
                self._action_rewards[idx]=down
            else:
                self._action_rewards[idx]=up
        
        ds.applyLambda2actions(self.actions.vals(), rew)

class ADBMSBufferCacheRewardContinousV4(RewardDownStayUp):
    def __init__(self, action_minmax=(-1,1), alpha=-1, beta=-1):
        super().__init__(action_minmax=action_minmax, alpha=alpha, beta=beta)            
        
    def compute(self, ds: ADBMSDataSetEntryContextSelector):
        def adjusted_sigmoid_lower_part(x, threshold, k = 10):
            # Calculate parameters
            x_0 = threshold
            # Choose a value for k to control the steepness
            
            # Adjusted sigmoid formula
            y = 2 * (1 / (1 + np.exp(-k * (x - x_0))) - 0.5)
            return y

        iperf, cur_idelta, threshold = ds.getIPerfIndicators()

        def rew(idx, bufincr):
            iperf, idelta, threshold = ds.getIPerfIndicators()
        
            down = stay = up = 0

            if idelta >= 0.:
                dd = idelta/(1-threshold)
                res = (1-dd)**10
                stay = res
                down = dd
                up = -1 if cur_idelta > 0. else stay
            else:
                # VIOLATION
                # When iperf is below the threshold
                res =  adjusted_sigmoid_lower_part(idelta + threshold, threshold, 50 ) # Strong penalty for going further down
                up = -res
                down = stay = -1

            if bufincr == 0:
                self._action_rewards[idx]=stay
            elif bufincr < 0:
                self._action_rewards[idx]=down
            else:
                self._action_rewards[idx]=up
        
        ds.applyLambda2actions(self.actions.vals(), rew)
        #iperf, idelta, threshold = ds.getIPerfIndicators()
        #print(f'IPERF:{iperf} IDELTA:{idelta} Threshold:{threshold} Actions vals:{self.actions.vals()} Rewards:{self._action_rewards}')

class ADBMSBufferCacheRewardContinousV5(RewardDownStayUp):
    def __init__(self, action_minmax=(-1,1), alpha=-1, beta=-1):
        super().__init__(action_minmax=action_minmax, alpha=alpha, beta=beta)            
        
    def compute(self, ds: ADBMSDataSetEntryContextSelector):
        def rew(idx, bufincr):
            iperf, idelta, threshold = ds.getIPerfIndicators()
            return max(-1, idelta/(1-threshold))
        
        ds.applyLambda2actions(self.actions.vals(), rew)


class ADBMSBufferCacheRewardSigmoid(RewardDownStayUp):
    def __init__(self, action_minmax=(-1,1), alpha=-1, beta=-1):
        super().__init__(action_minmax=action_minmax, alpha=alpha, beta=beta)            
        
    def compute(self, ds: ADBMSDataSetEntryContextSelector):
        def adjusted_sigmoid_lower_part(x, threshold, k = 10):
            # Calculate parameters
            x_0 = threshold
            # Choose a value for k to control the steepness
            
            # Adjusted sigmoid formula
            y = 2 * (1 / (1 + np.exp(-k * (x - x_0))) - 0.5)
            return y

        def adjusted_sigmoid_higher_part(x, threshold=0.5, k=10):
            # Rescale x to [0, 1] where x = threshold maps to 0 and x = 1 maps to 1
            scaled_x = (x - threshold) / (1 - threshold)
            # Apply the sigmoid function
            return 1 / (1 + np.exp(-k * (scaled_x - 0.5)))


        def rew(idx, bufincr):
            iperf, idelta, threshold = ds.getIPerfIndicators()
            down = stay = up = 0
             
            if idelta > 0:
                # When iperf is above or at the threshold
                down =  adjusted_sigmoid_higher_part(idelta + threshold ,  threshold , 10 )  # Encourage to go down if above threshold
                stay = 1- down #(1-dd)**4
                up   = stay-1
            else:
                # When iperf is below the threshold
                down =  adjusted_sigmoid_lower_part(idelta + threshold, threshold, 20 ) # Strong penalty for going further down
                stay = down
                up   = -stay

            if bufincr == 0:
                self._action_rewards[idx]=stay
            elif bufincr < 0:
                self._action_rewards[idx]=down
            else:
                self._action_rewards[idx]=up
        
        ds.applyLambda2actions(self.actions.vals(), rew)


class ADBMSBufferCacheRewardDiscrete(RewardDownStayUp):
    def __init__(self, action_minmax=(-1,1), alpha=-1, beta=-1):
        super().__init__(action_minmax=action_minmax, alpha=alpha, beta=beta)            
        
    def compute(self, ds: ADBMSDataSetEntryContextSelector):
        def rew(idx, bufincr):
            iperf, idelta, threshold = ds.getIPerfIndicators()
            down = stay = up = 0
            
            tolerance = 0.01
            if idelta >=0 and idelta <=tolerance:
                stay = 1 
            else :
                stay = -1
            if (idelta >=0 and idelta <=tolerance) or idelta < 0:
                down = -1
            else:
                down = 1
            if idelta < 0:
                up = 1
            else:
                up = -1

            if bufincr == 0:
                self._action_rewards[idx]=stay
            elif bufincr < 0:
                self._action_rewards[idx]=down
            else:
                self._action_rewards[idx]=up
        
        ds.applyLambda2actions(self.actions.vals(), rew)


class ADBMSBufferCacheRewardCrossed(RewardDownStayUp):
    def __init__(self, action_minmax=(-1,1), alpha=-1, beta=-1):
        super().__init__(action_minmax=action_minmax, alpha=alpha, beta=beta)            
        
    def compute(self, ds: ADBMSDataSetEntryContextSelector):
        def rew(idx, bufincr):
            iperf, idelta, threshold = ds.getIPerfIndicators()
            down = stay = up = 0
 
            if idelta >= 0:
                # When iperf is above or at the threshold
                dd = idelta / (1 - threshold)
                down = dd  # Encourage to go down if above threshold
                stay = max(0, (1 - dd) ** 2)  # Smaller power for smoother transition
                up = -abs(dd)  # Strong penalty for going further up
            else:
                # When iperf is below the threshold
                dd = idelta / threshold
                down = -abs(dd)  # Strong penalty for going further down
                stay = max(0, (1 + dd) ** 2)  # Reward for getting closer
                up = abs(dd)  # Encourage to go up if below threshold

            if bufincr == 0:
                self._action_rewards[idx]=stay
            elif bufincr < 0:
                self._action_rewards[idx]=down
            else:
                self._action_rewards[idx]=up
        
        ds.applyLambda2actions(self.actions.vals(), rew)

### New Reward Functions
class ADBMSBufferCacheRewardSigmoidHybridDiscreteDownCoeff(RewardDownStayUp):
    def __init__(self, action_minmax=(-1,1), alpha=-1, beta=-1):
        super().__init__(action_minmax=action_minmax, alpha=alpha, beta=beta)            
        
    def compute(self, ds: ADBMSDataSetEntryContextSelector):
        def adjusted_sigmoid_lower_part(x, threshold, k = 10):
            # Calculate parameters
            x_0 = threshold
            # Choose a value for k to control the steepness
            
            # Adjusted sigmoid formula
            y = 2 * (1 / (1 + np.exp(-k * (x - x_0))) - 0.5)
            return y

        def adjusted_sigmoid_higher_part(x, threshold=0.5, k=10):
            # Rescale x to [0, 1] where x = threshold maps to 0 and x = 1 maps to 1
            scaled_x = (x - threshold) / (1 - threshold)
            # Apply the sigmoid function
            return 1 / (1 + np.exp(-k * (scaled_x - 0.5)))


        def rew(idx, bufincr):
            iperf, idelta, threshold = ds.getIPerfIndicators()
            down = stay = up = 0
            down_coeff = 0.9
            if idelta > 0:
                # When iperf is above or at the threshold
                down =  (adjusted_sigmoid_higher_part(idelta + threshold ,  threshold , 10 )  - 1 + down_coeff) / down_coeff  # Encourage to go down if above threshold
                stay = 1- adjusted_sigmoid_higher_part(idelta + threshold ,  threshold , 10 ) #(1-dd)**4
                up   = stay-1
            else:
                # When iperf is below the threshold
                down = -1
                stay = -1
                up   = 1
            if bufincr == 0:
                self._action_rewards[idx]=stay
            elif bufincr < 0:
                self._action_rewards[idx]=down
            else:
                self._action_rewards[idx]=up
        
        ds.applyLambda2actions(self.actions.vals(), rew)

class ADBMSBufferCacheRewardContinousSymetrie(RewardDownStayUp):
    def __init__(self, action_minmax=(-1,1), alpha=-1, beta=-1):
        super().__init__(action_minmax=action_minmax, alpha=alpha, beta=beta)            
        
    def compute(self, ds: ADBMSDataSetEntryContextSelector):
        def rew(idx, bufincr):
            iperf, idelta, threshold = ds.getIPerfIndicators()
            down = stay = up = 0
            down_coeff = 1
            if idelta >= 0:
                dd = idelta/(1-threshold)
                stay = (1-dd)**8
                down = ( - stay  + down_coeff) /  down_coeff
                up   = stay-1
            else:
                # VIOLATION
        #            dd = -max(1, math.log(0.1+abs(idelta/threshold))+1)
                dd = idelta/threshold
                down = dd* down_coeff - 1 + down_coeff
                stay = dd 
                up   = -dd

            if bufincr == 0:
                self._action_rewards[idx]=stay
            elif bufincr < 0:
                self._action_rewards[idx]=down
            else:
                self._action_rewards[idx]=up
        
        ds.applyLambda2actions(self.actions.vals(), rew)

class ADBMSBufferCacheRewardContinousSymetrieDownCoeff(RewardDownStayUp):
    def __init__(self, action_minmax=(-1,1), alpha=-1, beta=-1):
        super().__init__(action_minmax=action_minmax, alpha=alpha, beta=beta)            
        
    def compute(self, ds: ADBMSDataSetEntryContextSelector):
        def rew(idx, bufincr):
            iperf, idelta, threshold = ds.getIPerfIndicators()
            down = stay = up = 0
            down_coeff = 0.8
            if idelta >= 0:
                dd = idelta/(1-threshold)
                stay = (1-dd)**8
                down = ( - stay  + down_coeff) /  down_coeff
                up   = stay-1
            else:
                # VIOLATION
        #            dd = -max(1, math.log(0.1+abs(idelta/threshold))+1)
                dd = idelta/threshold
                down = dd* down_coeff - 1 + down_coeff
                stay = dd 
                up   = -dd

            if bufincr == 0:
                self._action_rewards[idx]=stay
            elif bufincr < 0:
                self._action_rewards[idx]=down
            else:
                self._action_rewards[idx]=up
        
        ds.applyLambda2actions(self.actions.vals(), rew)

class ADBMSBufferCacheRewardContinousSymetrieDownCoeffHybridDiscrete(RewardDownStayUp):
    def __init__(self, action_minmax=(-1,1), alpha=-1, beta=-1):
        super().__init__(action_minmax=action_minmax, alpha=alpha, beta=beta)            
        
    def compute(self, ds: ADBMSDataSetEntryContextSelector):
        def rew(idx, bufincr):
            iperf, idelta, threshold = ds.getIPerfIndicators()
            down = stay = up = 0
            down_coeff = 0.8
            if idelta >= 0:
                dd = idelta/(1-threshold)
                stay = (1-dd)**8
                down = (1 - stay - 1 + down_coeff) /  down_coeff
                up   = stay-1
            else:
                # VIOLATION
        #            dd = -max(1, math.log(0.1+abs(idelta/threshold))+1)
                dd = idelta/threshold
                down = -1
                stay = -1
                up   = 1

            if bufincr == 0:
                self._action_rewards[idx]=stay
            elif bufincr < 0:
                self._action_rewards[idx]=down
            else:
                self._action_rewards[idx]=up
        
        ds.applyLambda2actions(self.actions.vals(), rew)
        
### adjustable idelta decale
class ADBMSBufferCacheRewardSigmoidHybridDiscreteDownCoeffMoveIdelta(RewardDownStayUp):
    def __init__(self, action_minmax=(-1,1), alpha=-1, beta=-1, move_idelta=0, strengthened_action=False):
        super().__init__(action_minmax=action_minmax, alpha=alpha, beta=beta, strengthened_action=strengthened_action)   
        self.move_idelta = move_idelta
        # to avoid divsision by 0 error, and when move_idelta > 1, the effect is the same for all
        if self.move_idelta == 1:
            self.move_idelta = 1.1
            
    def compute(self, ds: ADBMSDataSetEntryContextSelector):


        def adjusted_sigmoid_higher_part(x, threshold=0.5, k=10):
            # Rescale x to [0, 1] where x = threshold maps to 0 and x = 1 maps to 1
            scaled_x = (x - threshold) / (1 - threshold)
            # Apply the sigmoid function
            return 1 / (1 + np.exp(-k * (scaled_x - 0.5)))


        def rew(idx, bufincr):
            iperf, idelta, threshold = ds.getIPerfIndicators()
            threshold += self.move_idelta
            down = stay = up = 0
            down_coeff = 0.9
            if idelta > 0:
                # When iperf is above or at the threshold
                sigmo = adjusted_sigmoid_higher_part(idelta + threshold ,  threshold , 10 )
                down =  (sigmo  - 1 + down_coeff) / down_coeff  # Encourage to go down if above threshold
                stay = 1- sigmo 
                up   = stay-1
            else:
                # When iperf is below the threshold
                down = -1
                stay = -1
                up   = 1
            if bufincr == 0:
                self._action_rewards[idx]=stay
            elif bufincr < 0:
                self._action_rewards[idx]=down * (-bufincr)
            else:
                self._action_rewards[idx]=up * bufincr
        
        ds.applyLambda2actions(self.actions.vals(), rew)

class ADBMSBufferCacheRewardDistanceToOptimalPolicy(RewardDownStayUp):
    def __init__(self, action_minmax=(-1,1), alpha=-1, beta=-1, strengthened_action=False):
        super().__init__(action_minmax=action_minmax, alpha=alpha, beta=beta, strengthened_action=strengthened_action)

    # Compute reward for each possible actions at the current state, BEFORE applying any actionany action applied
    def compute(self, ds: ADBMSDataSetEntryContextSelector):
        dt_1 = ds.getDistanceToPolicyOptimal()
        #print(f'Compute distance to PO: DT1:{dt_1}')

        # Do not update current usla and cram here, as no action has been applied yet. Just compute the reward for each possible action
        def rew(idx, bufincr):
            dt = ds.getDistanceToPolicyOptimal()
            rew = dt_1 - dt
            # Enforce penalty on DOWN action
            #if bufincr < 0 and rew < 0:
            #    rew += bufincr
            self._action_rewards[idx] = rew
            #print(f'Compute distance to PO: DT:{dt} Rew:{self._action_rewards[idx]}')

        # Apply the reward computation for each possible action. self.actions_vals provides all possible actions. 
        ds.applyLambda2actions(self.actions.vals(), rew)



# OLD OLD OLD
class ADBMSBufferCacheRewardDistanceToOptimalPolicyBAK(RewardDownStayUp):
    def __init__(self, action_minmax=(-1,1), alpha=-1, beta=-1, strengthened_action=False):
        super().__init__(action_minmax=action_minmax, alpha=alpha, beta=beta, strengthened_action=strengthened_action)
        self._op_usla = 0
        self._op_cram = 0
        self._op_ob_usla = 0 # Over budget USLA
        self._op_ob_cram = 0 # Over budget CRAM
        self._used_budget = 0
        self._usla = 0
        self._cram = 0

    def _get_cur_perf(self, cur_usla, cur_cram):
        return ((cur_usla - self._op_usla) ** 2 + (cur_cram - self._op_cram) ** 2) ** 0.5

    def reset(self, ds: ADBMSDataSetEntryContextSelector):
        self._op_budget, self._op_usla, self._op_cram, self._op_ob_usla, self._op_ob_cram = ds.getPolicyOptimalPerf01Inficators()
        #print(f'Reset RewardDistanceToOptimalPolicy: OpBudget:{self._op_budget} OpUSLA:{self._op_usla} OpCRAM:{self._op_cram} OpOBUSLA:{self._op_ob_usla} OpOBCRAM:{self._op_ob_cram}')
        self._used_budget = 0
        self._usla = 0
        self._cram = 0

    # Compute reward for each possible actions at the current state, BEFORE applying any actionany action applied
    def compute(self, ds: ADBMSDataSetEntryContextSelector):
        #self._used_budget += 1
        #if self._used_budget >= self._op_budget:
        #    self._op_usla += self._op_ob_usla
        #    self._op_cram += self._op_ob_cram

        self._cram += ds.getBufferSizeMB()
        _, idelta, _ = ds.getIPerfIndicators()
        if idelta < 0:
            self._usla += 1

        dt_1 = self._get_cur_perf(self._usla, self._cram)
        #print(f'Compute current perf: USLA:{self._usla} CRAM:{self._cram} DT1:{dt_1}')

        # Do not update current usla and cram here, as no action has been applied yet. Just compute the reward for each possible action
        def rew(idx, bufincr):
            _, idelta, _ = ds.getIPerfIndicators()
            if idelta < 0:
                dt = self._get_cur_perf(self._usla+1, self._cram + ds.getBufferSizeMB())
            else:
                dt = self._get_cur_perf(self._usla, self._cram + ds.getBufferSizeMB())
            self._action_rewards[idx] = dt_1 - dt
            #print(f'Compute next rewards: BUF: {ds.getBufferSizeMB()}MB Action:{bufincr} Reward:{self._action_rewards[idx]} DT1:{dt_1} DT:{dt}')

        # Apply the reward computation for each possible action. self.actions_vals provides all possible actions. 
        ds.applyLambda2actions(self.actions.vals(), rew)

