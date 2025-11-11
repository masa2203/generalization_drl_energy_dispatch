import itertools
import os
import json
import socket
from typing import Any

from config import src_dir
from envs.environments import *
from envs.env_params import *
from utils.net_design import activation_fn_dict, net_arch_dict
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
SAVE_PATH = os.path.join(src_dir, 'log', ENV_KWARGS['env_name'], 'aappo', 'run', input('Save in folder: ')) \
    if CREATE_LOG else None

# ACTIONS - USES PREDEFINED DISCRETE ACTIONS
discretization_params = 'predefined'
DISCRETE_ACTIONS = None


# EXP PARAMS
EXP_PARAMS = {
    'n_runs': 5,
    'n_episodes': 70,
    'len_episode': int(ENV_KWARGS['modeling_period_h'] / ENV_KWARGS['resolution_h']),
    'seed': 22,
    # Env
    'use_predefined_action_wrapper': 'ab_default',  # USES PREDEFINED DISCRETE ACTIONS
    'flatten_obs': True,
    'frame_stack': 5,
    # Normalization
    'minmax_scaling': False,
    'norm_obs': True,
    'norm_reward': True,
    # Evaluation
    'eval_while_training': True,
    'eval_freq': int(ENV_KWARGS['modeling_period_h'] / ENV_KWARGS['resolution_h']) * 2,
    # Penalties/Reward Modifiers
    'p2g_soc_penalty': (1000, 0.05),  # Tuple (Penalty weight, upper SOC limit)
    'p2g_inactivity_penalty': {'penalty': 2000, 'alpha': 0.02, 'threshold': 0.9},
    'p2g_cost_assign_discharge': False,
    'p2g_lost_profits_assign_discharge': True,
    # Forecasts
    # 'perfect_forecasts': {'pool_price': [1, 2, 3, 6, 12, 18, 24]},
}

# PPO PARAMS
RL_PARAMS: dict[str, Any] = {
    'policy': "MlpPolicy" if EXP_PARAMS['flatten_obs'] else 'MultiInputPolicy',
    'device': 'cpu',
    'learning_rate': 1.87e-4,  # Default: 3e-4
    'use_lr_sched': True,
    'n_steps': 2048,  # Default: 2048
    'batch_size': 64,  # Default: 64
    'n_epochs': 10,  # Default: 10
    # 'gamma': 1-0.003512209278341583,  # Default: 0.99
    'gamma': 0.98097,  # Default: 0.99
    'gae_lambda': 0.90616,  # Default: 0.95
    'clip_range': 0.1,  # Default: 0.2
    'use_cr_sched': False,
    'clip_range_vf': 0.2,  # Default: None
    'use_cr_vf_sched': False,
    'normalize_advantage': True,  # Default: True
    'ent_coef': 0.0,  # Default: 0.0
    'vf_coef': 0.75,  # Default: 0.5
    'max_grad_norm': 9.7879,  # Default: 0.5
    'use_sde': False,  # Default: False
    'target_kl': None,

    'policy_kwargs': {
        # Defaults reported for MultiInputPolicy
        'net_arch': 'large',  # Default: None
        'activation_fn': 'tanh',  # Default: tanh
        'ortho_init': True,  # Default: True
        'use_expln': False,  # Default: False
        'squash_output': False,  # Default: False
        'share_features_extractor': False,  # Default: True
    }
}

if SAVE_PATH is not None:
    os.makedirs(SAVE_PATH, exist_ok=True)
    with open(os.path.join(SAVE_PATH, 'inputs.json'), 'w') as f:
        json.dump({
            'DISCRETE_ACTIONS': discretization_params if DISCRETE_ACTIONS is not None else None,
            'EXP_PARAMS': EXP_PARAMS,
            'RL_PARAMS': str(RL_PARAMS),
            'PLANT_PARAMS': ENV_KWARGS,
            'PC_NAME': socket.gethostname(),
        }, f)


RL_PARAMS['policy_kwargs']['net_arch'] = net_arch_dict[RL_PARAMS['policy_kwargs']['net_arch']]
RL_PARAMS['policy_kwargs']['activation_fn'] = activation_fn_dict[RL_PARAMS['policy_kwargs']['activation_fn']]

for run in range(EXP_PARAMS['n_runs']):
    train_rl_agent(
        agent='ppo',
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
