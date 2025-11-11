import copy
import os
import random
import itertools
from typing import Optional, Union, Tuple, Any, Deque, List, Dict

import numpy as np
import pandas as pd
import torch
import gymnasium as gym


def set_seeds(seed):
    """
    Fixes the random seed for all relevant packages.

    :param seed: An integer that represents the seed to be set.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def start_count(lst_of_actions_taken):
    """
    Calculates the number of GT starts from a list of GT dispatches.

    :param lst_of_actions_taken: A list that represents the GT dispatch history.
    :type lst_of_actions_taken: list
    :return: An integer that represents the number of GT starts.
    """
    count = 0
    count += 1 if lst_of_actions_taken[0] else 0
    for i in range(len(lst_of_actions_taken) - 1):
        if lst_of_actions_taken[i] == 0 and lst_of_actions_taken[i + 1] != 0:
            count += 1
    return count


def generate_discrete_actions(
        gt_specs: Optional[List[Dict[str, float]]] = None,
        p2g_specs: Optional[List[Dict[str, float]]] = None,
        bes_specs: Optional[List[Dict[str, float]]] = None
) -> List[np.ndarray]:
    """
    Generate discrete action space for GTs and/or BESs based on provided specifications.
    Can also generate actions for exclusively GTs or BESs if only one of them is provided.

    Parameters:
    gt_specs (Optional[List[Dict[str, float]]]): List of dictionaries, each containing
                                                 'start', 'stop', and 'num' for np.linspace for each GT.
    p2g_specs (Optional[List[Dict[str, float]]]): List of dictionaries, each containing
                                                 'start', 'stop', and 'num' for np.linspace for each P2G.
    bes_specs (Optional[List[Dict[str, float]]]): List of dictionaries, each containing
                                                 'start', 'stop', and 'num' for np.linspace for each BES.

    Returns:
    List[np.ndarray]: A list containing numpy arrays of all possible action combinations.

    Example:
        gt_specs = [{'start': -1, 'stop': 1, 'num': 9}]
        bes_specs = [{'start': -1, 'stop': 1, 'num': 9}]
        discrete_actions = generate_discrete_actions(gt_specs, bes_specs)
    """
    # Initialize action lists
    gt_actions, p2g_actions, bes_actions = [], [], []

    if gt_specs is not None:
        gt_actions = [np.linspace(**spec) for spec in gt_specs]

    if p2g_specs is not None:
        p2g_actions = [np.linspace(**spec) for spec in p2g_specs]

    if bes_specs is not None:
        bes_actions = [np.linspace(**spec) for spec in bes_specs]

    # Combine the action sets based on provided specs
    all_actions = []
    if gt_actions:
        all_actions += gt_actions
    if p2g_actions:
        all_actions += p2g_actions
    if bes_actions:
        all_actions += bes_actions

    # Ensure there's at least one action set provided
    if not all_actions:
        raise ValueError("At least one of gt_specs, p2g_specs, or bes_specs must be provided.")

    # Get all combinations
    combinations = list(itertools.product(*all_actions))

    # Convert each combination tuple to a numpy array
    discrete_actions = [np.array(combination) for combination in combinations]

    return discrete_actions


def plant_config_train_test_split():
    """
    Returns 3 pre-defined lists of dictionaries with different plant configurations;
    - training
    - testing (interpolation)
    - testing (extrapolation)
    """
    set_seeds(22)  # Fix random seed to ensure receiving the same train/test combo

    config = {
        'num_wt': [10, 11, 12, 13, 14, 15],  # 8, 17
        'rate_gas_price': [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],  # 0.2, 3.0
        'penalty': [250, 500, 750, 1000, 1500, 2000],  # 100, 3000
        'bes_cap': [45, 60, 75, 90, 105, 120],  # 30, 150
        'bes_rate': [5, 10, 15, 20, 25, 30],  # 3, 40
    }

    # Use itertools.product to generate all combinations
    all_combinations = list(itertools.product(*config.values()))

    dict_combinations = [
        dict(zip(config.keys(), combination)) for combination in all_combinations
    ]

    # Sample 10 unique dicts without replacement
    test_dicts = random.sample(dict_combinations, 10)

    # Form a list with the rest (excluding the 10 sampled)
    train_dicts = [d for d in dict_combinations if d not in test_dicts]

    # print(f"Number of sampled dicts: {len(test_dicts)}")
    # print(f"Number of dicts in the rest: {len(train_dicts)}")

    interpolation_test = test_dicts[:5]
    extrapolation_test = copy.deepcopy(test_dicts[5:])

    # Make extrapolation test-set with outliers
    extrapolation_test[0]['num_wt'] = 8
    extrapolation_test[1]['rate_gas_price'] = 3.0
    extrapolation_test[2]['penalty'] = 3000
    extrapolation_test[3]['bes_cap'] = 30
    extrapolation_test[4]['bes_rate'] = 40

    return train_dicts, interpolation_test, extrapolation_test


def plant_config_train_test_split_p2g():
    """
    Returns 3 pre-defined lists of dictionaries with different plant configurations;
    - training
    - testing (interpolation)
    - testing (extrapolation)
    """
    set_seeds(22)  # Fix random seed to ensure receiving the same train/test combo

    config = {
        'num_wt': [16, 18, 20, 22, 24],  # 12, 28
        'rate_e_price': [0.5, 0.75, 1.0, 1.5, 2.0, 2.5],  # 0.25, 3.0
        'rate_p2g_degr_cost': [0.5, 0.75, 1, 1.25, 1.5],  # 0.25, 2
        'bes_cap': [30, 40, 50, 60, 70],  # 20, 80
        'bes_rate': [10, 15, 20, 25, 30],  # 5, 40
    }

    # Use itertools.product to generate all combinations
    all_combinations = list(itertools.product(*config.values()))

    dict_combinations = [
        dict(zip(config.keys(), combination)) for combination in all_combinations
    ]

    # Sample 10 unique dicts without replacement
    test_dicts = random.sample(dict_combinations, 10)

    # Form a list with the rest (excluding the 10 sampled)
    train_dicts = [d for d in dict_combinations if d not in test_dicts]

    # print(f"Number of sampled dicts: {len(test_dicts)}")
    # print(f"Number of dicts in the rest: {len(train_dicts)}")

    interpolation_test = test_dicts[:5]
    extrapolation_test = copy.deepcopy(test_dicts[5:])

    # Make extrapolation test-set with outliers
    extrapolation_test[0]['num_wt'] = 12
    extrapolation_test[1]['rate_e_price'] = 3.0
    extrapolation_test[2]['rate_p2g_degr_cost'] = 0.25
    extrapolation_test[3]['bes_cap'] = 80
    extrapolation_test[4]['bes_rate'] = 5

    return train_dicts, interpolation_test, extrapolation_test


def calculate_mean_relative_scores(inter_rewards, extra_rewards):
    """
    Calculates the relative scores for inter and extra rewards based on the best scores and returns their mean values.

    :param inter_rewards: A list of numeric values representing the interpolation rewards.
    :param extra_rewards: A list of numeric values representing the extrapolation rewards.
    :return: A tuple containing: The mean of the relative interpolation scores, the mean of the relative
        extrapolation scores.
    """
    # Define the best scores
    best_inter_scores_dqn = [-5466518, -7783063, -5134396, -3179877, -6884899]
    best_inter_scores_td3 = [-5033188.79, -7293411.263, -5088136.76, -2892332.163, -6925745.579]
    best_inter_scores_aappo = [-5351900.19, -8031003.53, -4586798.95, -2576827.92, -7085096.94]
    best_inter_scores_aadqn = [-5296853.426, -7931724.502, -4534242.954, -2532732.371, -6912642.657]
    best_inter_scores = [
        max(i, j, k, l) for i, j, k, l in zip(
            best_inter_scores_td3, best_inter_scores_dqn,
            best_inter_scores_aappo, best_inter_scores_aadqn
        )
    ]

    best_extra_scores_dqn = [-7405513.336, -10923548.74, -4591372.595, -6629414.653, -5348889.839]
    best_extra_scores_td3 = [-6631201.376, -9295226.849, -4000128.007, -6049442.354, -5364683.191]
    best_extra_scores_aappo = [-6860054.95, -11165581.34, -3915714.38, -6261990.17, -5448765.58]
    best_extra_scores_aadqn = [-6848474.404, -11066694.97, -3736321.479, -6282409.291, -5240709.099]
    best_extra_scores = [
        max(i, j, k, l) for i, j, k, l in zip(
            best_extra_scores_td3, best_extra_scores_dqn,
            best_extra_scores_aappo, best_extra_scores_aadqn
        )
    ]

    # Calculate relative inter scores
    rel_inter_scores = [
        j / k if k != 0 else 0 for j, k in zip(inter_rewards, best_inter_scores)
    ]

    # Calculate relative extra scores
    rel_extra_scores = [
        j / k if k != 0 else 0 for j, k in zip(extra_rewards, best_extra_scores)
    ]

    # Return means of both lists
    return np.mean(rel_inter_scores), np.mean(rel_extra_scores)


class SavedPolicy:
    """
    Converts a list of actions (e.g. from MIP) to a policy-callable.
    """
    def __init__(self, actions, env, use_predefined_actions=False):
        """
        Initialize the policy.
        :param actions: List of continuous actions from MIQP.
        :param env: The environment instance wrapped with P2GPreDefinedDiscreteActions.
        :param use_predefined_actions: Boolean indicating whether to convert to discrete actions.
        """
        self.actions = actions
        self.env = env
        self.step = 0
        self.use_predefined_actions = use_predefined_actions

        if self.use_predefined_actions:
            assert isinstance(env.action_space, gym.spaces.Discrete), \
                "Environment must have a discrete action space when using predefined actions."
            self.discrete_actions = []
            # self.discrete_actions = [env.unwrapped.envs[0].map_action_to_continuous(i) for i in range(env.action_space.n)]

    def __call__(self, obs, state=None, dones=None):
        """Callable interface for the policy."""
        if self.step < len(self.actions):
            continuous_action = self.actions[self.step]
            self.step += 1

            if self.use_predefined_actions:
                # Find the nearest discrete action
                discrete_action = self.get_nearest_discrete_action(continuous_action)
                return np.array([discrete_action]), state
            else:
                # Use the MIQP action directly (continuous action)
                return np.array([continuous_action]), state
        else:
            # If the action sequence is exhausted, take a zero-action (or any default fallback)
            fallback_action = np.zeros_like(self.actions[0])
            if self.use_predefined_actions:
                discrete_action = self.get_nearest_discrete_action(fallback_action)
                return np.array([discrete_action]), state
            else:
                return np.array([fallback_action]), state

    def get_nearest_discrete_action(self, continuous_action):
        """
        Finds the nearest discrete action by comparing the MIQP action to all predefined discrete actions.
        :param continuous_action: A continuous action from MIQP.
        :return: Index of the nearest discrete action.
        """
        # Compute the Euclidean distance between the continuous action and each discrete action
        self.discrete_actions = [self.env.unwrapped.envs[0].map_action_to_continuous(i) for i in range(self.env.action_space.n)]
        distances = [np.linalg.norm(continuous_action - discrete_action) for discrete_action in self.discrete_actions]
        nearest_index = int(np.argmin(distances))
        return nearest_index
