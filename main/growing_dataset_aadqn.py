import os
import json
import copy
import socket
import time
from typing import Any

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure

from config import src_dir
from envs.environments import *
from envs.env_params import *
from utils.callbacks import ProgressBarManager
from utils.make_env import make_env
from utils.net_design import *
from utils.scheduler import linear_scheduler_sb3
from utils.utilities import set_seeds
from utils.logger import sb3_create_stats_file, get_env_log_data
from train.train import train_rl_agent

# PLANT PARAMS
ENV = GasTurbineBatteryRenewablesDemandEnv
ENV_KWARGS = on_3y_train
TEST_ENV_KWARGS = on_3y_test

# LOG
CREATE_LOG = False
SAVE_MODEL = False
VERBOSE = 0
LOGGER_TYPE = ["csv"]

# ACTIONS - USES PREDEFINED DISCRETE ACTIONS
discretization_params = 'predefined'
DISCRETE_ACTIONS = None

# EXP PARAMS
EXP_PARAMS = {
    'n_runs': 10,
    'n_episodes': 20_000,
    'mode': 'day',
    'nums': [1, 2, 4, 8],
    'seed': 22,
    # Env
    'use_predefined_action_wrapper': 'on_default',  # USES PREDEFINED DISCRETE ACTIONS
    'flatten_obs': True,
    'frame_stack': 6,
    # Normalization
    'minmax_scaling': True,
    'norm_obs': False,
    'norm_reward': True,
    # Evaluation
    'eval_while_training': True,
    # 'eval_freq': int(ENV_KWARGS['modeling_period_h'] / ENV_KWARGS['resolution_h']) * 10,
    # Penalties/Reward Modifiers
    # 'p2g_soc_penalty': (0, 0.05),  # Tuple (Penalty weight, upper SOC limit)
    # 'p2g_inactivity_penalty': {'penalty': 0, 'alpha': 0.1, 'threshold': 0.9},
    # 'p2g_cost_assign_discharge': False,
    # 'p2g_lost_profits_assign_discharge': True,
}

SAVE_PATH = os.path.join(src_dir, 'log', ENV_KWARGS['env_name'], 'aadqn', 'run', f"growing_{EXP_PARAMS['mode']}_dataset") \
    if CREATE_LOG else None

# DQN PARAMS
RL_PARAMS: dict[str, Any] = {
    'policy': "MlpPolicy" if EXP_PARAMS['flatten_obs'] else 'MultiInputPolicy',
    # 'learning_rate': 0.000373343249282699,  # Default: 1e-4
    'learning_rate': 0.0002,  # Default: 1e-4
    'use_lr_sched': True,
    'buffer_size': 1_000_000,  # Default: 1e6
    'learning_starts': 1_000,  # Default: 50_000
    'batch_size': 128,  # Default: 32
    'tau': 0.35,  # Default: 1.0
    'gamma': 0.91,  # Default: 0.99
    'train_freq': 20,  # Default: 4
    'gradient_steps': -1,  # Default: 1
    'target_update_interval': 1_000,  # Default: 1e4
    'exploration_fraction': 0.5,  # Default: 0.1
    'exploration_initial_eps': 1.0,  # Default: 1.0
    'exploration_final_eps': 0.01,  # Default: 0.05
    'max_grad_norm': 0.2,  # Default: 10

    'policy_kwargs': {
        # Defaults reported for MultiInputPolicy
        'net_arch': 'extra_large',  # Default: None
        'activation_fn': 'tanh',  # Default: tanh
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

RL_PARAMS['policy_kwargs']['net_arch'] = net_arch_dict[RL_PARAMS['policy_kwargs']['net_arch']]['qf']
RL_PARAMS['policy_kwargs']['activation_fn'] = activation_fn_dict[RL_PARAMS['policy_kwargs']['activation_fn']]

# Iterate over data-subset numbers
for num in EXP_PARAMS['nums']:
    print('|| Mode: {} | Number: {} ||'.format(EXP_PARAMS['mode'], num))
    # Create path for logging
    num_path = os.path.join(SAVE_PATH, f"{EXP_PARAMS['mode']}_{num}") if SAVE_PATH is not None else None
    if num_path is not None:
        os.makedirs(num_path, exist_ok=True)

    for run in range(EXP_PARAMS['n_runs']):
        # Training logic for single run

        # Get total timesteps
        if EXP_PARAMS['mode'] == 'day':
            tt = int(EXP_PARAMS['n_episodes'] * 24)  # total timesteps
            EXP_PARAMS['len_episode'] = 24
            EXP_PARAMS['eval_freq'] = int(24 * 100)
        elif EXP_PARAMS['mode'] == 'week':
            tt = int(EXP_PARAMS['n_episodes'] * 168)  # total timesteps
            EXP_PARAMS['len_episode'] = 168
            EXP_PARAMS['eval_freq'] = int(168 * 10)
        elif EXP_PARAMS['mode'] == 'month':
            tt = int(EXP_PARAMS['n_episodes'] * 24 * 30)  # total timesteps
            EXP_PARAMS['len_episode'] = 24 * 30
            EXP_PARAMS['eval_freq'] = int(24 * 30 * 6)
        elif EXP_PARAMS['mode'] == 'year':
            tt = int(EXP_PARAMS['n_episodes'] * 24 * 365)  # total timesteps
            EXP_PARAMS['len_episode'] = 24 * 365
            EXP_PARAMS['eval_freq'] = int(24 * 365 * 2)
        else:
            raise NotImplementedError('Mode not supported!')

        # Create path for logging
        run_path = os.path.join(num_path, f'run_{run}') if num_path is not None else None
        if num_path is not None:
            os.makedirs(run_path, exist_ok=True)

        # Update seed
        seed = EXP_PARAMS['seed'] + run
        set_seeds(seed)
        print('\n|| Run #{} | Seed #{} ||'.format(run, seed))

        start = time.time()

        train_env_kwargs = copy.deepcopy(ENV_KWARGS)
        if 'p2g' in train_env_kwargs:
            train_env_kwargs['p2g']['assign_cost_at_discharge'] = EXP_PARAMS.get('p2g_cost_assign_discharge', False)
            train_env_kwargs['p2g']['assign_lost_profits_at_discharge'] = EXP_PARAMS.get(
                'p2g_lost_profits_assign_discharge', False)

        # CREATE ENVIRONMENT
        train_env = make_env(env=ENV,
                             env_kwargs=train_env_kwargs,
                             path=os.path.join(run_path, "train_monitor.csv") if run_path is not None else None,
                             use_random_episodes=dict(mode=EXP_PARAMS['mode'], num=num),
                             minmax_scaling=EXP_PARAMS['minmax_scaling'],
                             use_predefined_action_wrapper=EXP_PARAMS['use_predefined_action_wrapper'],
                             p2g_soc_penalty=EXP_PARAMS.get('p2g_soc_penalty', None),
                             p2g_inactivity_penalty=EXP_PARAMS.get('p2g_inactivity_penalty', None),
                             flatten_obs=EXP_PARAMS['flatten_obs'],
                             frame_stack=EXP_PARAMS['frame_stack'],
                             norm_obs=EXP_PARAMS['norm_obs'],
                             norm_reward=EXP_PARAMS['norm_reward'],
                             gamma=RL_PARAMS['gamma'],
                             verbose=bool(VERBOSE),
                             )

        # Handling of schedulers (remove from dict and apply function)
        if 'use_lr_sched' in RL_PARAMS:
            use_lr_sched = RL_PARAMS.pop('use_lr_sched')
            if use_lr_sched:
                RL_PARAMS['learning_rate'] = linear_scheduler_sb3(RL_PARAMS['learning_rate'])

        # DEFINE MODEL
        model = DQN(env=train_env, verbose=VERBOSE, seed=seed, **RL_PARAMS)
        if run_path is not None:
            logger = configure(run_path, LOGGER_TYPE)
            model.set_logger(logger)

        with ProgressBarManager(total_timesteps=tt) as callback:
            # Evaluation during training
            if EXP_PARAMS['eval_while_training']:
                eval_env = make_env(env=ENV,
                                    env_kwargs=ENV_KWARGS,
                                    path=os.path.join(run_path, 'eval_monitor.csv') if run_path is not None else None,
                                    use_random_episodes=dict(mode=EXP_PARAMS['mode'], num=num),
                                    minmax_scaling=EXP_PARAMS['minmax_scaling'],
                                    use_predefined_action_wrapper=EXP_PARAMS['use_predefined_action_wrapper'],
                                    flatten_obs=EXP_PARAMS['flatten_obs'],
                                    frame_stack=EXP_PARAMS['frame_stack'],
                                    norm_obs=EXP_PARAMS['norm_obs'],
                                    norm_reward=EXP_PARAMS['norm_reward'],
                                    gamma=RL_PARAMS['gamma'],
                                    )

                evaluate_policy(model, eval_env, n_eval_episodes=num)  # test this to add untrained agent's stats

                eval_callback = EvalCallback(eval_env=eval_env,
                                             n_eval_episodes=num,
                                             eval_freq=EXP_PARAMS['eval_freq'],
                                             deterministic=True,
                                             # best_model_save_path=run_path,
                                             best_model_save_path=None,
                                             verbose=VERBOSE)

                model.learn(total_timesteps=tt, callback=[eval_callback, callback])
            # Evaluation only after training
            else:
                model.learn(total_timesteps=tt, callback=callback)

        # train_env.training = False
        # train_env.norm_reward = False

        test_env = make_env(env=ENV,
                            env_kwargs=TEST_ENV_KWARGS,
                            path=os.path.join(run_path, 'test_monitor.csv') if run_path is not None else None,
                            use_random_episodes=False,
                            minmax_scaling=EXP_PARAMS['minmax_scaling'],
                            use_predefined_action_wrapper=EXP_PARAMS['use_predefined_action_wrapper'],
                            flatten_obs=EXP_PARAMS['flatten_obs'],
                            frame_stack=EXP_PARAMS['frame_stack'],
                            norm_obs=EXP_PARAMS['norm_obs'],
                            norm_reward=False,
                            gamma=RL_PARAMS['gamma'],
                            )

        test_env.training = False

        test_env.unwrapped.envs[0].unwrapped.start_tracking()  # Start tracking env variables for evaluation
        mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=1)

        print(f"mean_reward: {mean_reward:,.2f} +/- {std_reward:,.2f}")

        if run_path is not None:
            if SAVE_MODEL:
                model.save(os.path.join(run_path, 'model'))  # Save best model instead through callback
                train_env.save(os.path.join(run_path, 'env.pkl'))
            log_data = get_env_log_data(env=test_env, mean_reward=mean_reward, start_time=start)
            with open(os.path.join(run_path, 'output.json'), 'w') as f:
                json.dump(log_data, f)

        train_env.close()
        test_env.close()

        print()
        print(f'Execution time = {time.time() - start}s')

    # GET STATISTICS FROM MULTIPLE RUNS
    # if CREATE_LOG and EXP_PARAMS['n_runs'] > 1:
        # Does not make sense for current logger when multiple episodes are evaluated
        # if EXP_PARAMS['eval_while_training']:
        #     sb3_create_stats_file(path=num_path, exp_params=EXP_PARAMS, style='eval')
        # sb3_create_stats_file(path=num_path, exp_params=EXP_PARAMS, style='train')
