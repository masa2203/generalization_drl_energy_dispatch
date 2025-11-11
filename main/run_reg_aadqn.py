# REGULARIZED VERSION

import os
import json
import socket
from typing import Any

from config import src_dir
from envs.environments import *
from envs.env_params import *
from utils.net_design import activation_fn_dict, net_arch_dict, CustomANN
from utils.logger import sb3_create_stats_file
from train.train import train_rl_agent

# PLANT PARAMS
ENV = GasTurbineP2GBatteryRenewablesDemandEnv
ENV_KWARGS = ab_3y_train

# LOG
CREATE_LOG = False
SAVE_MODEL = False
VERBOSE = 0
LOGGER_TYPE = ["csv"]
SAVE_PATH = os.path.join(src_dir, 'log', ENV_KWARGS['env_name'], 'aadqn', 'run', input('Save in folder: ')) \
    if CREATE_LOG else None

# ACTIONS - USES PREDEFINED DISCRETE ACTIONS
discretization_params = 'predefined'
DISCRETE_ACTIONS = None

# EXP PARAMS
EXP_PARAMS = {
    'n_runs': 5,
    'n_episodes': 10,
    'len_episode': int(ENV_KWARGS['modeling_period_h'] / ENV_KWARGS['resolution_h']),
    'seed': 22,
    # Env
    'use_predefined_action_wrapper': 'ab_default',  # USES PREDEFINED DISCRETE ACTIONS
    'flatten_obs': True,
    'frame_stack': 6,
    # Normalization
    'minmax_scaling': True,
    'norm_obs': False,
    'norm_reward': True,
    # Evaluation
    'eval_while_training': True,
    'eval_freq': int(ENV_KWARGS['modeling_period_h'] / ENV_KWARGS['resolution_h']) * 1,
    # Penalties/Reward Modifiers
    # 'p2g_soc_penalty': (0, 0.05),  # Tuple (Penalty weight, upper SOC limit)
    # 'p2g_inactivity_penalty': {'penalty': 0, 'alpha': 0.1, 'threshold': 0.9},
    'p2g_cost_assign_discharge': False,
    'p2g_lost_profits_assign_discharge': True,
}

ENV_KWARGS['state_vars'] = dict(
        re_power=(0.0, 50.0),
        sin_h=(-1.0, 1.0),
        cos_h=(-1.0, 1.0),
        pool_price=(0.0, 1000.0),
    )

# DQN PARAMS
RL_PARAMS: dict[str, Any] = {
    'policy': "MlpPolicy" if EXP_PARAMS['flatten_obs'] else 'MultiInputPolicy',
    # 'learning_rate': 0.000373343249282699,  # Default: 1e-4
    'learning_rate': 2.03e-4,  # Default: 1e-4
    'use_lr_sched': True,
    'buffer_size': 1_000_000,  # Default: 1e6
    'learning_starts': 1_000,  # Default: 50_000
    'batch_size': 128,  # Default: 32
    'tau': 0.335,  # Default: 1.0
    'gamma': 1-0.099,  # Default: 0.99
    'train_freq': 75,  # Default: 4
    'gradient_steps': -1,  # Default: 1
    'target_update_interval': 10_000,  # Default: 1e4
    'exploration_fraction': 0.5,  # Default: 0.1
    'exploration_initial_eps': 0.8,  # Default: 1.0
    'exploration_final_eps': 0.3,  # Default: 0.05
    'max_grad_norm': 7.1,  # Default: 10

    'policy_kwargs': {
        # Defaults reported for MultiInputPolicy
        'net_arch': [128],  # Default: None
        'activation_fn': 'leaky_relu',  # Default: tanh
        'features_extractor_class': CustomANN,
        'features_extractor_kwargs': dict(
            ann_net_shape=[128, 256],
            activation='leaky_relu',
            dropout=0.05,
            batch_norm=False,
        ),
        'optimizer_kwargs': dict(weight_decay=1e-4)
    }
}

if SAVE_PATH is not None:
    os.makedirs(SAVE_PATH, exist_ok=True)
    with open(os.path.join(SAVE_PATH, 'inputs.json'), 'w') as f:
        json.dump({
            'DISCRETE_ACTIONS': discretization_params,
            'EXP_PARAMS': EXP_PARAMS,
            'RL_PARAMS': str(RL_PARAMS),
            'PLANT_PARAMS': ENV_KWARGS,
            'PC_NAME': socket.gethostname(),
        }, f)

# RL_PARAMS['policy_kwargs']['net_arch'] = net_arch_dict[RL_PARAMS['policy_kwargs']['net_arch']]['qf']
RL_PARAMS['policy_kwargs']['activation_fn'] = activation_fn_dict[RL_PARAMS['policy_kwargs']['activation_fn']]

for run in range(EXP_PARAMS['n_runs']):
    train_rl_agent(
        agent='dqn',
        run=run,
        path=SAVE_PATH,
        exp_params=EXP_PARAMS,
        env_id=ENV,
        env_kwargs=ENV_KWARGS,
        discrete_actions=DISCRETE_ACTIONS,
        rl_params=RL_PARAMS,
        verbose=VERBOSE,
        logger_type=LOGGER_TYPE,
        save_model_and_env=SAVE_MODEL,
    )

# GET STATISTICS FROM MULTIPLE RUNS
if CREATE_LOG and EXP_PARAMS['n_runs'] > 1:
    if EXP_PARAMS['eval_while_training']:
        sb3_create_stats_file(path=SAVE_PATH, exp_params=EXP_PARAMS, style='eval')
    sb3_create_stats_file(path=SAVE_PATH, exp_params=EXP_PARAMS, style='train')
