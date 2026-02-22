import copy
from typing import Dict, List
import numpy as np
import pandas as pd
import torch

import gymnasium as gym
from gymnasium import spaces

import warnings
# warnings.filterwarnings("ignore", category=UserWarning)  # Get variables from wrapper warnings


class MinMaxScaler(gym.ObservationWrapper):
    """
    A gym observation wrapper that scales the observation to the range [0, 1].

    :param env: A gym environment.
    """

    def __int__(self, env):
        """
        Initializes the MinMaxScaler class.

        :param env: A gym environment.
        """
        super().__init__(env)

    def observation(self, obs):
        """
        Scales the observation to the range [0, 1].

        :param obs: A numpy array that represents the observation.
        :return: A numpy array that represents the scaled observation.
        """
        scaled_obs = {}
        if obs is not None:
            for key in obs:
                low = self.observation_space[key].low
                high = self.observation_space[key].high
                scaled_obs[key] = (obs[key] - low) / (high - low)
        return scaled_obs


class DiscreteActions(gym.ActionWrapper):
    """
    A gym action wrapper that converts discrete actions to continuous actions.

    :param env: A gym environment.
    :param disc_to_cont: A list that represents the discrete actions.
    """

    def __init__(self, env, disc_to_cont):
        """
        Initializes the DiscreteActions class.

        :param env: A gym environment.
        :param disc_to_cont: A list that represents the discrete actions.
        """
        super().__init__(env)
        self.disc_to_cont = disc_to_cont

        # Check on correct action dimension
        assert env.action_space.shape[0] == len(disc_to_cont[0]), \
            "Number action dimension after discretization must match environment's action dimensions."

        self.action_space = gym.spaces.Discrete(len(disc_to_cont))

    def action(self, act):
        """
        Converts the discrete action to a continuous action.

        :param act: An integer that represents the discrete action.
        :return: A numpy array that represents the continuous action.
        """
        return np.array(self.disc_to_cont[act]).astype(self.env.action_space.dtype)

    def reverse_action(self, action):
        """
        Raises a NotImplementedError.

        :param action: A numpy array that represents the action.
        :raises: NotImplementedError.
        """
        raise NotImplementedError


class RescaleActionSpace(gym.ActionWrapper):
    """
    A gym action wrapper that rescales the action space, taking into account non-zero lower bounds.
    """

    def __init__(self, env):
        """
        Initializes the RescaleActionSpace class.

        :param env: A gym environment.
        """
        super(RescaleActionSpace, self).__init__(env)
        self.orig_action_space = self.env.action_space
        # Calculate the scale and offset for each action dimension based on original bounds
        self.scale = (self.orig_action_space.high - self.orig_action_space.low) / 2.0
        self.offset = (self.orig_action_space.high + self.orig_action_space.low) / 2.0
        # Define the new action space as [-1, 1] for all dimensions
        self.action_space = spaces.Box(low=-1, high=1, shape=self.orig_action_space.shape,
                                       dtype=self.orig_action_space.dtype)

    def action(self, action):
        """
        Rescales the action from [-1,1] to the original action space.

        :param action: A numpy array that represents the action in the [-1,1] space.
        :return: A numpy array that represents the rescaled action in the original action space.
        """
        # Rescale actions to the original space
        rescaled_action = action * self.scale + self.offset
        return rescaled_action

    def reverse_action(self, action):
        """
        Reverses the rescaling of the action from the original action space to [-1,1].

        :param action: A numpy array that represents the action in the original action space.
        :return: A numpy array that represents the reversed action in the [-1,1] space.
        """
        # Reverse scaling from original space to [-1,1]
        reversed_action = (action - self.offset) / self.scale
        return reversed_action


class PreDefinedDiscreteActions(gym.ActionWrapper):
    """
    Wrapper that defines a discrete action space with adaptive actions
    (based on the then-statement of if-then-else rules).

    Basic version with 6 actions for standard GT-BES-RE-Demand env.
    """

    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.action_space.shape[0] == 2, 'Action space not fitting to this pre-defined action wrapper!'
        assert len(self.env.get_wrapper_attr('gts')) == 1, 'Env for this wrapper should have one GT!'

        # Define a new discrete action space with 6 actions
        self.action_space = gym.spaces.Discrete(6)
        self.num_gts = len(self.env.get_wrapper_attr('gts'))

        self.bes = self.env.unwrapped.storage

        self.avg_gt_max = 35  # in MW
        self.gt_tolerance = 0.00  # Increase GT action on [0,1] scale by this amount to compensate for amb. conditions

    def action(self, action):
        """
        Takes a discrete action and maps it to a continuous action.
        """
        # Map the discrete action to a continuous action
        continuous_action = self.map_action_to_continuous(action)
        # Return the continuous action to be taken in the environment
        return continuous_action

    def map_action_to_continuous(self, action):
        """
        Maps the discrete action to the continuous action space of the environment.
        Assumes that the original continuous action space is a Box with the first
        dimensions corresponding to GTs and the last dimension to the Battery.
        """
        continuous_action = np.zeros(self.num_gts + 1)  # +1 for the battery

        demand = self.env.unwrapped.obs['demand'].item()
        re_power = self.env.unwrapped.obs['re_power'].item()

        diff = demand - re_power  # positive diff => additional energy needed

        # Keep GT and BES off/idle
        if action == 0:
            pass

        # Charge BES with surplus REs (no GT usage)
        elif action == 1:
            if diff >= 0:  # If no surplus REs
                pass  # Leave BES idle
            else:  # Surplus REs
                bes_action = max(diff / self.bes.max_charge_rate, -1.0)
                continuous_action[-1] = bes_action  # Charge BES

        # Meet deficient power supply with BES (no GT usage)
        elif action == 2:
            if diff <= 0:  # If no deficiency
                pass  # Leave BES idle
            else:
                bes_action = min(diff / self.bes.max_discharge_rate / self.bes.discharge_eff, 1.0)
                continuous_action[-1] = bes_action

        # Meet deficient power supply with GT (no BES usage)
        elif action == 3:
            if diff <= 0:  # If no deficiency
                pass  # Leave/turn GT off
            else:
                gt_action = min((diff / self.avg_gt_max) + self.gt_tolerance, 1.0)
                continuous_action[0] = gt_action

        # Meet deficient power supply with BES + GT (Prioritizing BES)
        elif action == 4:
            if diff <= 0:  # If no deficiency
                pass  # Leave/turn GT off
            else:
                # Note: This doesn't account for insufficient SOC
                # First, use as much BES power as possible/necessary
                bes_action = min(diff / self.bes.max_discharge_rate / self.bes.discharge_eff, 1.0)
                continuous_action[-1] = bes_action
                # Meet difference from GT
                bes_flow = bes_action * self.bes.max_discharge_rate * self.bes.discharge_eff
                gt_action = min(((diff - bes_flow) / self.avg_gt_max) + self.gt_tolerance, 1.0)
                continuous_action[0] = max(0, gt_action)

        # Use GT for both deficient power supply + BES charging
        elif action == 5:
            if diff <= 0:  # If no deficiency
                pass  # Leave/turn GT off
            else:
                # Note: This doesn't account for full SOC
                gt_action_needed = min((diff / self.avg_gt_max) + self.gt_tolerance, 1.0)
                gt_action = min(gt_action_needed + 0.32, 1.0)  # 0.32 ~= 10 MW
                continuous_action[0] = gt_action

                surplus_gt_power = (gt_action - gt_action_needed) * self.avg_gt_max
                bes_action = max(-surplus_gt_power / self.bes.max_charge_rate, -1.0)
                continuous_action[-1] = bes_action

        # Correct for GT startup (less power produced due to ramping)
        if continuous_action[0] != 0 and self.env.unwrapped.gts[0].GT_state == 0:
            # Note: Start time is saved in hour-fraction, e.g. 0.25 = 15min.
            start_time = self.env.unwrapped.gts[0].start_reg_h
            if ('t2m' in self.env.unwrapped.obs and
                    self.env.unwrapped.gts[0].start_long_h is not None and
                    self.env.unwrapped.obs['t2m'] < 273.15):
                start_time = self.env.unwrapped.gts[0].start_long_h

            new_gt_actions = min(continuous_action[0] * (1 / (1 - start_time)), 1)
            continuous_action[0] = new_gt_actions

        return continuous_action.astype(self.env.get_wrapper_attr('precision')['float'])

    def reset(self, **kwargs):
        """
        Resets the environment and updates 'self.bes'.
        """
        obs = self.env.reset(**kwargs)  # Reset the underlying environment
        self.bes = self.env.unwrapped.storage  # Update BES  after reset
        return obs


class ABPreDefinedDiscreteActions(gym.ActionWrapper):
    """
    Wrapper that defines a discrete action space with adaptive actions
    (based on the then-statement of if-then-else rules).

    P2G version with 8 actions for GT-BES-P2G-RE-DEMAND env.
    """

    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.action_space.shape[0] == 3, 'Action space not fitting to this pre-defined action wrapper!'
        assert len(self.env.get_wrapper_attr('gts')) == 1, 'Env for this wrapper should have one GT!'

        # Define a new discrete action space with 8 actions
        self.action_space = gym.spaces.Discrete(8)
        self.num_gts = len(self.env.get_wrapper_attr('gts'))

        self.bes = self.env.unwrapped.storage
        self.p2g = self.env.unwrapped.p2g

        self.avg_gt_max = 35  # in MW
        self.gt_tolerance = 0.00  # Increase GT action on [0,1] scale by this amount to compensate for amb. conditions

    def action(self, action):
        """
        Takes a discrete action and maps it to a continuous action.
        """
        # Map the discrete action to a continuous action
        continuous_action = self.map_action_to_continuous(action)
        # Return the continuous action to be taken in the environment
        return continuous_action

    def map_action_to_continuous(self, action):
        """
        Maps the discrete action to the continuous action space of the environment.
        Assumes that the original continuous action space is a Box with the first
        dimensions corresponding to GTs and the last dimension to the Battery.
        """
        continuous_action = np.zeros(self.num_gts + 2)  # + 1 for the P2G system +1 for the battery

        demand = self.env.unwrapped.obs['demand'].item()
        re_power = self.env.unwrapped.obs['re_power'].item()

        diff = demand - re_power  # positive diff => additional energy needed

        # Keep GT, P2G and BES off/idle
        if action == 0:
            pass

        # Charge BES with REs (as much as possible, no GT or P2G usage)
        elif action == 1:
            bes_action = -1.0
            continuous_action[2] = bes_action  # Charge BES

        # Charge BES with surplus REs (no GT or P2G usage)
        elif action == 2:
            if diff >= 0:  # If no surplus REs
                pass  # Leave BES idle
            else:  # Surplus REs
                bes_action = max(diff / self.bes.max_charge_rate, -1.0)
                continuous_action[2] = bes_action  # Charge BES

        # Operate P2G with REs (as much as possible, no GT or BES usage)
        elif action == 3:
            p2g_action = 1.0
            continuous_action[1] = p2g_action  # Operate P2G

        # Operate P2G with surplus REs (no GT or BES usage)
        elif action == 4:
            if diff >= 0:  # If no surplus REs
                pass  # Leave P2G idle
            else:  # Surplus REs
                p2g_action = min(abs(diff) / self.p2g.max_charge_rate, 1.0)
                continuous_action[1] = p2g_action  # Operate P2G

        # Discharge BES
        elif action == 5:
            if diff <= 0:  # If no deficiency
                pass  # Leave BES idle
            else:
                bes_action = min(diff / self.bes.max_discharge_rate / self.bes.discharge_eff, 1.0)
                continuous_action[2] = bes_action

        # Meet deficient power supply with GT (no BES usage)
        elif action == 6:
            if diff <= 0:  # If no deficiency
                pass  # Leave/turn GT off
            else:
                gt_action = min((diff / self.avg_gt_max) + self.gt_tolerance, 1.0)
                continuous_action[0] = gt_action

        # Meet deficient power supply with BES + GT (Prioritizing BES)
        elif action == 7:
            if diff <= 0:  # If no deficiency
                pass  # Leave/turn GT off
            else:
                # Note: This doesn't account for insufficient SOC
                # First, use as much BES power as possible/necessary
                bes_action = min(diff / self.bes.max_discharge_rate / self.bes.discharge_eff, 1.0)
                continuous_action[2] = bes_action
                # Meet difference from GT
                bes_flow = bes_action * self.bes.max_discharge_rate * self.bes.discharge_eff
                gt_action = min(((diff - bes_flow) / self.avg_gt_max) + self.gt_tolerance, 1.0)
                continuous_action[0] = max(0, gt_action)

        # Correct for GT startup (less power produced due to ramping)
        if continuous_action[0] != 0 and self.env.unwrapped.gts[0].GT_state == 0:
            # Note: Start time is saved in hour-fraction, e.g. 0.25 = 15min.
            start_time = self.env.unwrapped.gts[0].start_reg_h
            if ('t2m' in self.env.unwrapped.obs and
                    self.env.unwrapped.gts[0].start_long_h is not None and
                    self.env.unwrapped.obs['t2m'] < 273.15):
                start_time = self.env.unwrapped.gts[0].start_long_h

            new_gt_actions = min(continuous_action[0] * (1 / (1 - start_time)), 1)
            continuous_action[0] = new_gt_actions

        return continuous_action.astype(self.env.get_wrapper_attr('precision')['float'])

    def reset(self, **kwargs):
        """
        Resets the environment and updates 'self.bes' and 'self.p2g'.
        """
        obs = self.env.reset(**kwargs)  # Reset the underlying environment
        self.bes = self.env.unwrapped.storage  # Update BES after reset
        self.p2g = self.env.unwrapped.p2g  # Update P2G after reset
        return obs


class ABSmallPreDefinedDiscreteActions(gym.ActionWrapper):
    """
    Wrapper that defines a discrete action space with adaptive actions
    (based on the then-statement of if-then-else rules).

    P2G version with 5 actions for GT-BES-P2G-RE-DEMAND env.
    """

    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.action_space.shape[0] == 3, 'Action space not fitting to this pre-defined action wrapper!'
        assert len(self.env.get_wrapper_attr('gts')) == 1, 'Env for this wrapper should have one GT!'

        # Define a new discrete action space with 5 actions
        self.action_space = gym.spaces.Discrete(5)
        self.num_gts = len(self.env.get_wrapper_attr('gts'))

        self.bes = self.env.unwrapped.storage
        self.p2g = self.env.unwrapped.p2g

        self.avg_gt_max = 35  # in MW
        self.gt_tolerance = 0.00  # Increase GT action on [0,1] scale by this amount to compensate for amb. conditions

    def action(self, action):
        """
        Takes a discrete action and maps it to a continuous action.
        """
        # Map the discrete action to a continuous action
        continuous_action = self.map_action_to_continuous(action)
        # Return the continuous action to be taken in the environment
        return continuous_action

    def map_action_to_continuous(self, action):
        """
        Maps the discrete action to the continuous action space of the environment.
        Assumes that the original continuous action space is a Box with the first
        dimensions corresponding to GTs and the last dimension to the Battery.
        """
        gt_action = 0
        bes_action = 0
        p2g_action = 0

        demand = self.env.unwrapped.obs['demand'].item()
        re_power = self.env.unwrapped.obs['re_power'].item()

        diff = demand - re_power  # positive diff => additional energy needed

        # Keep GT, P2G and BES off/idle
        if action == 0:
            pass

        # Charge SURPLUS into BES (priority) and P2G
        elif action == 1:
            if diff <= 0:  # enough re power available
                if self.bes.max_soc > self.env.unwrapped.storage.soc:  # charge BES if not full already
                    bes_action = max(diff / self.bes.max_charge_rate, -1.0)
                    new_diff = diff - bes_action * self.bes.max_charge_rate
                    if abs(new_diff) > self.p2g.min_charge_rate:  # REs left after BES charge
                        p2g_action = min(abs(new_diff) / self.p2g.max_charge_rate, 1.0)
                else:  # charge P2G (note: could fall under min_charge_rate)
                    p2g_action = min(abs(diff) / self.p2g.max_charge_rate, 1.0)

        # Charge ANYWAY into BES (priority) and P2G (note: BES priority incorporated into env already)
        elif action == 2:
            bes_action = -1.0
            p2g_action = 1.0

        # Discharge BES
        elif action == 3:
            if diff <= 0:  # If no deficiency
                pass  # Leave BES idle
            else:
                bes_action = min(diff / self.bes.max_discharge_rate / self.bes.discharge_eff, 1.0)

        # Meet deficient power supply with BES + GT (Prioritizing BES)
        elif action == 4:
            if diff <= 0:  # If no deficiency
                pass  # Leave/turn GT off
            else:
                # Note: This doesn't account for insufficient SOC
                # First, use as much BES power as possible/necessary
                bes_action = min(diff / self.bes.max_discharge_rate / self.bes.discharge_eff, 1.0)
                # Meet difference from GT
                bes_flow = bes_action * self.bes.max_discharge_rate * self.bes.discharge_eff
                gt_action = min(((diff - bes_flow) / self.avg_gt_max) + self.gt_tolerance, 1.0)
                gt_action = max(0, gt_action)

        # Correct for GT startup (less power produced due to ramping)
        if gt_action != 0 and self.env.unwrapped.gts[0].GT_state == 0:
            # Note: Start time is saved in hour-fraction, e.g. 0.25 = 15min.
            start_time = self.env.unwrapped.gts[0].start_reg_h
            if ('t2m' in self.env.unwrapped.obs and
                    self.env.unwrapped.gts[0].start_long_h is not None and
                    self.env.unwrapped.obs['t2m'] < 273.15):
                start_time = self.env.unwrapped.gts[0].start_long_h

            gt_action = min(gt_action * (1 / (1 - start_time)), 1)

        continuous_action = np.array([gt_action, p2g_action, bes_action])

        return continuous_action.astype(self.env.get_wrapper_attr('precision')['float'])

    def reset(self, **kwargs):
        """
        Resets the environment and updates 'self.bes' and 'self.p2g'.
        """
        obs = self.env.reset(**kwargs)  # Reset the underlying environment
        self.bes = self.env.unwrapped.storage  # Update BES after reset
        self.p2g = self.env.unwrapped.p2g  # Update P2G after reset
        return obs


class P2GSOCPenalty(gym.RewardWrapper):
    """
    A gym reward wrapper that applies a penalty to the reward based on the SOC of the P2G system.

    This wrapper reduces the reward given to the agent by a penalty that increases as the SOC of the P2G system
    decreases. This encourages the agent to maintain a higher SOC in the P2G system.

    :param env: A gymnasium environment.
    :param penalty: A float representing the penalty applied for each unit below the max SOC.
    :param max_soc: A float representing the maximum SOC value, default is 1.
    """
    def __init__(self, env, penalty, max_soc=1):
        """
        Initializes the P2GSOCPenalty class.

        :param env: A gym environment.
        :param penalty: A float representing the penalty applied for each unit below the max SOC.
        :param max_soc: A float representing the maximum SOC value, default is 1.
        """
        super().__init__(env)
        self.penalty = penalty
        self.max_soc = max_soc

    def reward(self, reward):
        """
        Adjusts the reward based on the current SOC of the P2G system.

        Applies a penalty to the reward if the SOC of the P2G system is below the max_soc. The penalty is proportional
        to the difference between the current SOC and max_soc.

        :param reward: A float representing the original reward for the current step.
        :return: A float representing the adjusted reward after applying the penalty.
        """
        p = max((self.max_soc - self.env.get_wrapper_attr('obs')['p2g_soc'].item()), 0)/self.max_soc * self.penalty
        reward -= p
        return reward


class P2GInactivityPenalty(gym.RewardWrapper):
    """
    A gym reward wrapper that applies a penalty to the reward if the electricity price is below
    a certain threshold of the running mean and the P2G system is not used.

    This encourages the agent to utilize the P2G system more efficiently, especially when electricity
    prices are particularly low.

    :param env: A gymnasium environment.
    :param penalty: A float representing the penalty applied when conditions are met.
    :param alpha: A float representing the coefficient for updating the running mean of the price.
    :param threshold: A float representing the percentage of the running mean below which the penalty is applied.
                      The value should be between 0 and 1, where 1 means 100% of the running mean.
    """

    def __init__(self, env, penalty, alpha=0.1, threshold=0.9):
        """
        Initializes the P2GInactivityPenalty class.

        :param env: A gym environment.
        :param penalty: A float representing the penalty applied when conditions are met.
        :param alpha: A float representing the coefficient for updating the running mean of the price.
        :param threshold: A float representing the threshold percentage of the running mean.
        """
        super().__init__(env)
        self.penalty = penalty
        self.alpha = alpha
        self.threshold = threshold
        self.running_mean_price = None  # Will be initialized on the first call to reward()

    def reward(self, reward):
        """
        Adjusts the reward based on the current electricity price and P2G system activity.

        Applies a penalty to the reward if the current electricity price is below the running mean and
        the P2G system is not being used (negative power flow).

        :param reward: A float representing the original reward for the current step.
        :return: A float representing the adjusted reward after applying the penalty.
        """
        # Extract current electricity price from the environment's observation
        # Using class variable instead of obs-space ensures current price is used
        current_price = self.env.get_wrapper_attr('pool_price')

        # Initialize the running mean with the first observed price, if not already done
        if self.running_mean_price is None:
            self.running_mean_price = current_price

        # Update the running mean of the electricity price
        self.running_mean_price += self.alpha * (current_price - self.running_mean_price)

        # Calculate threshold price
        threshold_price = self.running_mean_price * self.threshold

        #  Check if current price is below running mean and if P2G was used (if used to generate gas,
        #  power flow must be negative)
        if (current_price < threshold_price and
                self.env.get_wrapper_attr('p2g_power') >= 0 and
                self.env.get_wrapper_attr('obs')['re_power'] >= self.env.get_wrapper_attr('p2g').min_charge_rate):
            reward -= self.penalty  # Apply penalty

        return reward


class RandomEpisodesWrapper(gym.Wrapper):
    """
    A gym observation wrapper that samples random subsets from the underlying time-series data based on the specified
     mode ('day', 'week', or 'month') and iterates over these subsets at every new episode.

    :param env: A gymnasium environment.
    :param mode: A string specifying the episode length to pick from ['day', 'week', 'month'].
    :param num: An integer representing the number of episodes to sample.
    """

    def __init__(
            self,
            env,
            mode: str = 'day',  # Episode length to pick
            num: int = 1,  # Number of episodes to sample.
    ):
        """
        Initializes the RandomEpisodesWrapper class.

        :param env: A gymnasium environment.
        :param mode: A string specifying the subset selection mode ('day', 'week', or 'month').
        :param num: An integer representing the number of subsets (episodes) to be randomly sampled.
        """
        super().__init__(env)

        def get_random_subset(mode: str = 'day'):
            """Returns a sampled subset of the entire dataset."""
            while True:  # Keep trying until a suitable subset is found
                if mode == 'day':
                    date_only_np = np.array([pd.to_datetime(date).date() for date in dates_np])
                    unique_days = np.unique(date_only_np)
                    random_day = np.random.choice(unique_days)
                    random_day_indices = np.where(date_only_np == random_day)[0]
                    if len(random_day_indices) >= 24:  # Check if at least 24 hours of data
                        random_day_data = {var: self.org_data[var][random_day_indices] for var in self.org_data}
                        print('Day picked: ', random_day)
                        return random_day_data
                elif mode == 'week':
                    week_only_np = np.array([pd.to_datetime(date).strftime('%Y-%U') for date in dates_np])
                    unique_weeks = np.unique(week_only_np)
                    random_week = np.random.choice(unique_weeks)
                    random_week_indices = np.where(week_only_np == random_week)[0]
                    if len(random_week_indices) >= 168:  # Check if at least 168 hours of data
                        random_week_data = {var: self.org_data[var][random_week_indices] for var in self.org_data}
                        print('Week picked: ', random_week)
                        return random_week_data
                elif mode == 'month':
                    month_year_np = np.array([pd.to_datetime(date).strftime('%Y-%m') for date in dates_np])
                    unique_months = np.unique(month_year_np)
                    random_month = np.random.choice(unique_months)
                    random_month_indices = np.where(month_year_np == random_month)[0]
                    if len(random_month_indices) >= 28 * 24:  # At least 28 days
                        random_month_data = {var: self.org_data[var][random_month_indices] for var in self.org_data}
                        print('Month picked: ', random_month)
                        return random_month_data
                elif mode == 'year':
                    unique_years = np.unique(dates_np.astype('datetime64[Y]'))
                    random_year = np.random.choice(unique_years)
                    random_year_indices = np.where(dates_np.astype('datetime64[Y]') == random_year)[0]
                    if len(random_year_indices) >= 365 * 24:  # At least 365 days
                        random_year_data = {var: self.org_data[var][random_year_indices] for var in self.org_data}
                        print('Year picked: ', random_year)
                        return random_year_data
                else:
                    raise NotImplementedError('Mode not supported!')

        # Save original dataset
        self.org_data = self.env.unwrapped.data

        # Remove timezone and convert to datetime
        mod_data = [dt.split('+')[0] for dt in self.org_data['Date']]
        dates_np = np.array([np.datetime64(date) for date in mod_data])

        # Placeholder for dataset subsets
        self.datasets = []

        # Create desired number of subsets
        for i in range(num):
            subset = get_random_subset(mode=mode)
            self.datasets.append(subset)

        # Iterator
        self.i = 0

    def reset(self, **kwargs):
        """
        Resets the environment with a new randomly selected subset of data based on the specified mode
        and updates the dataset for the next episode.

        This method ensures that each episode the agent experiences is based on a different subset,
        cycling through the pre-generated list of subsets.

        :param kwargs: Additional keyword arguments passed to the environment's reset method.
        :return: The initial observation and info from the environment's reset method.
        """
        # Reset env variables with new episode length and dataset
        self.env.unwrapped.len_episode = self.datasets[self.i]['Date'].shape[0]
        self.env.unwrapped.data = self.datasets[self.i]

        # Update iterator
        self.i += 1
        self.i %= len(self.datasets)

        obs, info = self.env.reset(**kwargs)
        return obs, info


class UpdateEnvConfig(gym.ObservationWrapper):
    """
    A gymnasium observation wrapper that modifies the environment's configuration and observation space dynamically
    based on predefined configurations. This wrapper allows for customization of environment parameters such as
    renewable energy power, gas prices, penalties, and battery storage capacities, while also expanding the
    observation space.

    :param env: A gymnasium environment.
    :param configs: A list of dictionaries, where each dictionary contains configuration values for the environment.
                    These configurations are applied sequentially at the start of each new episode.
    """

    def __init__(
            self,
            env,
            configs: list[dict]
    ):
        """
        Initializes the UpdateEnvConfig class.

        :param env: A gymnasium environment.
        :param configs: A list of dictionaries containing environment configuration values. Each dictionary should
                        specify parameters such as 'num_wt', 'rate_gas_price', 'penalty', 'bes_cap', and 'bes_rate'.
        """
        super().__init__(env)

        if self.env.unwrapped.pv_cap_mw != 0:
            warnings.warn('Wrapper must be adapted to work with PV power.')

        # Save original dataset
        self.org_data = copy.deepcopy(self.env.unwrapped.data)

        # Step 1: Pre-saved information
        self.defaults = {'num_wt': self.env.unwrapped.num_wt}
        # self.configs = [{'num_wt': 10, 'rate_gas_price': 0.5, 'penalty': 250, 'bes_cap': 45, 'bes_rate': 5}]
        self.configs = configs
        self.new_state_vars = dict(
            penalty=(0, 5000),
            bes_cap=(0, 200),
            bes_rate=(0, 50),
        )

        # Step 2: Expand state space
        for var in self.new_state_vars:
            self.observation_space[var] = spaces.Box(low=self.new_state_vars[var][0],
                                                     high=self.new_state_vars[var][1],
                                                     shape=(1,))

        # Iterator
        self.i = -1  # Start with config 0 when calling reset() for the first time

        # print('\nPlant configurations: ', self.configs)

    def observation(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Updates the observation space with the current configuration values.

        :param obs: A dictionary containing the original observation from the environment.
        :return: A dictionary containing the updated observation, with additional variables representing the current
                 configuration values.
        """
        for var in self.new_state_vars:
            obs[var] = np.array([self.configs[self.i][var]])
        return obs

    def reset(self, **kwargs):
        """
        Resets the environment with updated configuration values based on the current episode index and modifies the
        environment's underlying dataset. It also ensures that the observation space is updated accordingly.

        :param kwargs: Additional keyword arguments passed to the environment's reset method.
        :return: The modified initial observation and info from the environment's reset method.
        """
        # Update iterator
        self.i += 1  # Note, must be done before env update to avoid mismatch with observation()-method.
        self.i %= len(self.configs)

        # Update the env's dataset
        # Note: Computationally cheap solution, avoid re-reading of csv files
        self.env.unwrapped.data['re_power'] = self.org_data['re_power'] * (
                self.configs[self.i]['num_wt'] / self.defaults['num_wt'])
        self.env.unwrapped.data['gas_price'] = self.org_data['gas_price'] * self.configs[self.i]['rate_gas_price']

        # Update the env's components
        self.env.unwrapped.grid.penalty = self.configs[self.i]['penalty']

        self.env.unwrapped.storage.total_cap = self.configs[self.i]['bes_cap']
        self.env.unwrapped.storage.investment_cost = (self.configs[self.i]['bes_cap'] *
                                                      self.env.unwrapped.storage_dict['degradation']['battery_capex'])

        self.env.unwrapped.storage.max_charge_rate = self.configs[self.i]['bes_rate']
        self.env.unwrapped.storage.max_discharge_rate = self.configs[self.i]['bes_rate']

        # Call reset and update obs space
        obs, info = self.env.reset(**kwargs)
        obs = self.observation(obs)

        # print('\tNext config: ', self.configs[self.i])

        return obs, info


class UpdateEnvConfigP2G(gym.ObservationWrapper):
    """
    A gymnasium observation wrapper that modifies the environment's configuration and observation space dynamically
    based on predefined configurations. This wrapper allows for customization of environment parameters such as
    renewable energy power, gas prices, penalties, and battery storage capacities, while also expanding the
    observation space.

    Version for P2G environment.

    :param env: A gymnasium environment.
    :param configs: A list of dictionaries, where each dictionary contains configuration values for the environment.
                    These configurations are applied sequentially at the start of each new episode.
    """

    def __init__(
            self,
            env,
            configs: list[dict]
    ):
        """
        Initializes the UpdateEnvConfig class.

        :param env: A gymnasium environment.
        :param configs: A list of dictionaries containing environment configuration values. Each dictionary should
                        specify parameters such as 'num_wt', 'rate_gas_price', 'penalty', 'bes_cap', and 'bes_rate'.
        """
        super().__init__(env)

        if self.env.unwrapped.pv_cap_mw != 0:
            warnings.warn('Wrapper must be adapted to work with PV power.')

        # Save original dataset
        self.org_data = copy.deepcopy(self.env.unwrapped.data)

        # Step 1: Pre-saved information
        self.defaults = {'num_wt': self.env.unwrapped.num_wt,
                         'p2g_degr_cost': self.env.unwrapped.p2g.degradation_cost_per_timestep}
        self.configs = configs
        self.new_state_vars = dict(
            rate_p2g_degr_cost=(0, 3),
            bes_cap=(0, 200),
            bes_rate=(0, 50),
        )

        # Step 2: Expand state space
        for var in self.new_state_vars:
            self.observation_space[var] = spaces.Box(low=self.new_state_vars[var][0],
                                                     high=self.new_state_vars[var][1],
                                                     shape=(1,))

        # Iterator
        self.i = -1  # Start with config 0 when calling reset() for the first time

        # print('\nPlant configurations: ', self.configs)

    def observation(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Updates the observation space with the current configuration values.

        :param obs: A dictionary containing the original observation from the environment.
        :return: A dictionary containing the updated observation, with additional variables representing the current
                 configuration values.
        """
        for var in self.new_state_vars:
            obs[var] = np.array([self.configs[self.i][var]])
        return obs

    def reset(self, **kwargs):
        """
        Resets the environment with updated configuration values based on the current episode index and modifies the
        environment's underlying dataset. It also ensures that the observation space is updated accordingly.

        :param kwargs: Additional keyword arguments passed to the environment's reset method.
        :return: The modified initial observation and info from the environment's reset method.
        """
        # Update iterator
        self.i += 1  # Note, must be done before env update to avoid mismatch with observation()-method.
        self.i %= len(self.configs)

        # Update the env's dataset
        # Note: Computationally cheap solution, avoid re-reading of csv files
        self.env.unwrapped.data['re_power'] = self.org_data['re_power'] * (
                self.configs[self.i]['num_wt'] / self.defaults['num_wt'])
        self.env.unwrapped.data['pool_price'] = self.org_data['pool_price'] * self.configs[self.i]['rate_e_price']

        # Update the env's components
        self.env.unwrapped.p2g.degradation_cost_per_timestep = self.defaults['p2g_degr_cost'] * self.configs[self.i]['rate_p2g_degr_cost']

        self.env.unwrapped.storage.total_cap = self.configs[self.i]['bes_cap']
        self.env.unwrapped.storage.investment_cost = (self.configs[self.i]['bes_cap'] *
                                                      self.env.unwrapped.storage_dict['degradation']['battery_capex'])

        self.env.unwrapped.storage.max_charge_rate = self.configs[self.i]['bes_rate']
        self.env.unwrapped.storage.max_discharge_rate = self.configs[self.i]['bes_rate']

        # Call reset and update obs space
        obs, info = self.env.reset(**kwargs)
        obs = self.observation(obs)

        # print('\tNext config: ', self.configs[self.i])

        return obs, info
