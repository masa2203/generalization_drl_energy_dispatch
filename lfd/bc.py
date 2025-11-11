import json
import socket
import time

import torch

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import policies

import imitation
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.policies.serialize import load_policy
from imitation.util import logger as imit_logger

from config import src_dir
from envs.environments import *
from envs.env_params import *
from utils.logger import get_env_log_data, get_reward_stats_from_output
from utils.net_design import net_arch_dict, activation_fn_dict
from utils.wrappers import *
from utils.utilities import set_seeds, generate_discrete_actions
from utils.make_env import make_env
from utils.scheduler import linear_scheduler_sb3


# SPECIFY ENVS
ENV = GasTurbineBatteryRenewablesDemandEnv
TRAIN_ENV_KWARGS = on_3y_train
TEST_ENV_KWARGS = on_3y_test

EXP_PARAMS = {
    'n_runs': 5,  # iterates over experts with different random seeds
    'seed': 22,
    'expert': 'aadqn',
    'rollout_episodes': 10,  # expert data collection
    'n_epochs': 30,  # BC training epochs
    'batch_size': 64,
    'lr': 5.3e-4,
    'ent_weight': 7.7e-3,
    'l2_weight': 2.1e-3,
    'net_arch': 'large',
    'activation_fn': 'leaky_relu',
}

# EXPERT (takes 'run_0' if n_runs = 1)
path_to_expert = os.path.join(src_dir, 'log', TRAIN_ENV_KWARGS['env_name'], EXP_PARAMS['expert'], 'run', 'test')

with open(os.path.join(path_to_expert, 'inputs.json')) as f:
    inputs = json.load(f)

# ACTIONS
disc_actions = None
discretization_params = inputs.get('DISCRETE_ACTIONS', None)
if discretization_params != 'predefined' and discretization_params is not None:
    disc_actions = generate_discrete_actions(**discretization_params)


# LOG
CREATE_LOG = False
SAVE_PATH = os.path.join(src_dir, 'log', TRAIN_ENV_KWARGS['env_name'], 'bc', 'run', input('Save in folder: ')) \
    if CREATE_LOG else None


if SAVE_PATH is not None:
    os.makedirs(SAVE_PATH, exist_ok=False)
    with open(os.path.join(SAVE_PATH, 'inputs.json'), 'w') as f:
        json.dump({
            'EXP_PARAMS': EXP_PARAMS,
            'EXPERT': path_to_expert,
            'TRAIN_ENV_KWARGS': TRAIN_ENV_KWARGS,
            'TEST_ENV_KWARGS': TEST_ENV_KWARGS,
            'PC_NAME': socket.gethostname(),
        }, f)


if EXP_PARAMS['expert'] == 'td3':
    imitation.policies.serialize._add_stable_baselines_policies_from_file(dict(td3="stable_baselines3:TD3"))
if EXP_PARAMS['expert'] == 'dqn':
    imitation.policies.serialize._add_stable_baselines_policies_from_file(dict(dqn="stable_baselines3:DQN"))
if EXP_PARAMS['expert'] == 'aadqn':
    imitation.policies.serialize._add_stable_baselines_policies_from_file(dict(aadqn="stable_baselines3:DQN"))


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
        discrete_actions=disc_actions,
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
        discrete_actions=disc_actions,
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
    transitions = rollout.flatten_trajectories(rollouts)

    policy = policies.ActorCriticPolicy(
        observation_space=demo_env.observation_space,
        action_space=demo_env.action_space,
        lr_schedule=lambda _: torch.finfo(torch.float32).max,  # no effect as optimizer is overwritten by BC class
        net_arch=net_arch_dict[EXP_PARAMS['net_arch']],
        activation_fn=activation_fn_dict[EXP_PARAMS['activation_fn']],
    )

    bc_trainer = bc.BC(
        observation_space=demo_env.observation_space,
        action_space=demo_env.action_space,
        demonstrations=transitions,
        batch_size=EXP_PARAMS['batch_size'],
        rng=rng,
        policy=policy,  # None = two hidden layers with 32 units
        optimizer_kwargs={'lr': EXP_PARAMS['lr']},
        ent_weight=EXP_PARAMS['ent_weight'],
        l2_weight=EXP_PARAMS['l2_weight'],
        custom_logger=custom_logger,
    )

    bc_trainer.train(n_epochs=EXP_PARAMS['n_epochs'])

    demo_env.training = False
    demo_env.norm_reward = False
    demo_mean_reward, demo_std_reward = evaluate_policy(bc_trainer.policy, demo_env, n_eval_episodes=1)

    print(f"DEMO ENV: mean_reward: {demo_mean_reward:,.2f} +/- {demo_std_reward:,.2f}")

    eval_env.training = False
    eval_env.norm_reward = False
    eval_env.unwrapped.envs[0].unwrapped.start_tracking()  # Start tracking env variables for evaluation
    mean_reward, std_reward = evaluate_policy(bc_trainer.policy, eval_env, n_eval_episodes=1)

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
