import json
import re
import socket
import time
from typing import Any

from stable_baselines3 import DQN

from stable_baselines3.common.evaluation import evaluate_policy

import imitation
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.policies.serialize import load_policy
from imitation.util import logger as imit_logger

from config import src_dir
from envs.environments import *
from envs.env_params import *
from utils.logger import get_env_log_data, get_reward_stats_from_output
from utils.net_design import net_arch_dict, activation_fn_dict, create_reward_net
from utils.wrappers import *
from utils.utilities import set_seeds
from utils.make_env import make_env
from utils.scheduler import linear_scheduler_sb3


# SPECIFY ENVS
ENV = GasTurbineBatteryRenewablesDemandEnv
TRAIN_ENV_KWARGS = on_3y_train
TEST_ENV_KWARGS = on_3y_test

EXP_PARAMS = {
    'n_runs': 5,  # also uses different expert runs
    'seed': 22,
    'expert': 'aadqn',
    'rollout_episodes': 10,  # expert data collection
    'total_timesteps': 7200,  # number of generator updates, rounds are computed as r =  tt // gen_train_t
    'demo_batch_size': 128,
    'n_disc_updates_per_round': 8,  # Default: 2 (discriminator updates per round (see above))
    'disc_opt_kwargs': dict(lr=1.9e-5),
    'gen_train_timesteps': None,  # Default: None (= batch size for on-policy and num of envs for off-policy)
    'gen_replay_buffer_capacity': 50_000,  # Default: None = equal 'gen_train_timesteps' (most recent batch used only)

    # Discriminator network params
    'reward_net_is_shaped': True,  # Whether to use potential or not | False
    'disc_norm_input': False,  # Normalize input layer | True
    'reward_hid_sizes': (128, ),  # If shaped: Tuple defining structure of reward net | (32, )
    'potential_hid_sizes': (32, 32),  # If shaped: Tuple defining structure of potential net | (32, 32)
}


# EXPERT (takes 'run_0' if n_runs = 1)
path_to_expert = os.path.join(src_dir, 'log', TRAIN_ENV_KWARGS['env_name'], EXP_PARAMS['expert'], 'run', 'test')

with open(os.path.join(path_to_expert, 'inputs.json')) as f:
    inputs = json.load(f)

# LOG
CREATE_LOG = False
SAVE_PATH = os.path.join(src_dir, 'log', TRAIN_ENV_KWARGS['env_name'], 'gail', 'run', input('Save in folder: ')) \
    if CREATE_LOG else None

# DQN PARAMS
RL_PARAMS: dict[str, Any] = {
    'policy': "MlpPolicy",
    # 'learning_rate': 0.000373343249282699,  # Default: 1e-4
    'learning_rate': 2.66e-5,  # Default: 1e-4
    'use_lr_sched': False,
    'buffer_size': 1_000_000,  # Default: 1e6
    'learning_starts': 1_000,  # Default: 50_000
    'batch_size': 256,  # Default: 32
    'tau': 0.287,  # Default: 1.0
    'gamma': 0.976,  # Default: 0.99
    'train_freq': 90,  # Default: 4
    'gradient_steps': -1,  # Default: 1
    'target_update_interval': 1_000,  # Default: 1e4
    'exploration_fraction': 0.5,  # Default: 0.1
    'exploration_initial_eps': 1.0,  # Default: 1.0
    'exploration_final_eps': 0.01,  # Default: 0.05
    'max_grad_norm': 0.274,  # Default: 10

    'policy_kwargs': {
        # Defaults reported for MultiInputPolicy
        'net_arch': 'medium',  # Default: None
        'activation_fn': 'tanh',  # Default: tanh
    }
}

if SAVE_PATH is not None:
    os.makedirs(SAVE_PATH, exist_ok=True)
    with open(os.path.join(SAVE_PATH, 'inputs.json'), 'w') as f:
        json.dump({
            'EXP_PARAMS': EXP_PARAMS,
            'EXPERT': path_to_expert,
            'RL_PARAMS': RL_PARAMS,
            'TRAIN_ENV_KWARGS': TRAIN_ENV_KWARGS,
            'TEST_ENV_KWARGS': TEST_ENV_KWARGS,
            'PC_NAME': socket.gethostname(),
        }, f)

# Handling of schedulers (remove from dict and apply function)
if 'use_lr_sched' in RL_PARAMS:
    use_lr_sched = RL_PARAMS.pop('use_lr_sched')
    if use_lr_sched:
        RL_PARAMS['learning_rate'] = linear_scheduler_sb3(RL_PARAMS['learning_rate'])

RL_PARAMS['policy_kwargs']['net_arch'] = net_arch_dict[RL_PARAMS['policy_kwargs']['net_arch']]['qf']
RL_PARAMS['policy_kwargs']['activation_fn'] = activation_fn_dict[RL_PARAMS['policy_kwargs']['activation_fn']]


if EXP_PARAMS['expert'] == 'aadqn':
    imitation.policies.serialize._add_stable_baselines_policies_from_file(dict(aadqn="stable_baselines3:DQN"))

# Get gamma from input file
match = re.search(r"'gamma': ([\d.]+)", inputs['RL_PARAMS'])
gamma = float(match.group(1))


for run in range(EXP_PARAMS['n_runs']):
    # UPDATE SEED
    seed = EXP_PARAMS['seed'] + run
    set_seeds(seed)
    rng = np.random.default_rng(seed)
    print('|| Run #{} | Seed #{} ||'.format(run, seed))

    # Create path for logging
    run_path = os.path.join(SAVE_PATH, f'run_{run}') if SAVE_PATH is not None else None
    custom_logger = imit_logger.configure(run_path, ["csv", "tensorboard"]) if CREATE_LOG else None
    if SAVE_PATH is not None:
        os.makedirs(run_path, exist_ok=True)

    demo_env = make_env(
        env=ENV,
        env_kwargs=TRAIN_ENV_KWARGS,
        path=None,
        flatten_obs=inputs['EXP_PARAMS']['flatten_obs'],
        use_rollout_info_wrapper=True,
        frame_stack=inputs['EXP_PARAMS']['frame_stack'],
        minmax_scaling=inputs['EXP_PARAMS'].get('minmax_scaling', False),
        use_predefined_action_wrapper=inputs['EXP_PARAMS'].get('use_predefined_action_wrapper', None),
        norm_obs=inputs['EXP_PARAMS']['norm_obs'],
        norm_reward=inputs['EXP_PARAMS']['norm_reward'],
    )

    eval_env = make_env(
        env=ENV,
        env_kwargs=TEST_ENV_KWARGS,
        path=None,
        flatten_obs=inputs['EXP_PARAMS']['flatten_obs'],
        use_rollout_info_wrapper=True,
        frame_stack=inputs['EXP_PARAMS']['frame_stack'],
        minmax_scaling=inputs['EXP_PARAMS'].get('minmax_scaling', False),
        use_predefined_action_wrapper=inputs['EXP_PARAMS'].get('use_predefined_action_wrapper', None),
        norm_obs=inputs['EXP_PARAMS']['norm_obs'],
        norm_reward=inputs['EXP_PARAMS']['norm_reward'],
    )

    start = time.time()

    expert = load_policy(
        policy_type=EXP_PARAMS['expert'],
        path=os.path.join(path_to_expert, f'run_{run}'),
        venv=demo_env,
    )

    rollouts = rollout.rollout(
        policy=expert,  # Set to none for a random policy as "expert"
        venv=demo_env,
        sample_until=rollout.make_sample_until(min_timesteps=None, min_episodes=EXP_PARAMS['rollout_episodes']),
        rng=rng,
        unwrap=False,
        # deterministic_policy=True,
    )

    learner = DQN(
        env=demo_env,
        **RL_PARAMS,
    )

    # Discriminator is trained on reward_net
    # -> predicts if transition stems from expert or generator, does not predict reward
    reward_net = create_reward_net(
        observation_space=demo_env.observation_space,
        action_space=demo_env.action_space,
        is_shaped=EXP_PARAMS.get('reward_net_is_shaped', False),
        norm_input=EXP_PARAMS.get('disc_norm_input', False),
        reward_hid_sizes=EXP_PARAMS.get('reward_hid_sizes', (32, )),
        potential_hid_sizes=EXP_PARAMS.get('potential_hid_sizes', (32, 32)),
        gamma=gamma,
    )

    gail_trainer = GAIL(
        demonstrations=rollouts,
        demo_batch_size=EXP_PARAMS['demo_batch_size'],
        venv=demo_env,
        gen_algo=learner,
        reward_net=reward_net,
        n_disc_updates_per_round=EXP_PARAMS['n_disc_updates_per_round'],  # discriminator updates before generator update(s)
        disc_opt_kwargs=EXP_PARAMS['disc_opt_kwargs'],
        gen_train_timesteps=EXP_PARAMS['gen_train_timesteps'],
        gen_replay_buffer_capacity=EXP_PARAMS['gen_replay_buffer_capacity'],  # None = default -> sample only from most recent batch of gen samples
        custom_logger=custom_logger,
    )

    gail_trainer.train(total_timesteps=EXP_PARAMS['total_timesteps'])

    demo_env.training = False
    demo_env.norm_reward = False
    demo_mean_reward, demo_std_reward = evaluate_policy(gail_trainer.policy, demo_env, n_eval_episodes=1)

    print(f"DEMO ENV: mean_reward: {demo_mean_reward:,.2f} +/- {demo_std_reward:,.2f}")

    eval_env.training = False
    eval_env.norm_reward = False
    eval_env.unwrapped.envs[0].unwrapped.start_tracking()  # Start tracking env variables for evaluation
    mean_reward, std_reward = evaluate_policy(gail_trainer.policy, eval_env, n_eval_episodes=1)

    print(f"EVAL ENV: mean_reward: {mean_reward:,.2f} +/- {std_reward:,.2f}")

    demo_env.close()
    eval_env.close()

    if run_path is not None:
        log_data = get_env_log_data(env=eval_env, mean_reward=mean_reward, start_time=start)
        log_data['demo_env_reward_sum'] = demo_mean_reward
        with open(os.path.join(run_path, 'output.json'), 'w') as f:
            json.dump(log_data, f)


# GET STATISTICS FROM MULTIPLE RUNS
if CREATE_LOG and EXP_PARAMS['n_runs'] > 1:
    get_reward_stats_from_output(path=SAVE_PATH)