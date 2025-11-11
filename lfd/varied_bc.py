import json
import pickle
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
from utils.logger import get_inter_extra_stats_from_output
from utils.net_design import net_arch_dict, activation_fn_dict
from utils.wrappers import *
from utils.utilities import set_seeds, generate_discrete_actions, plant_config_train_test_split, calculate_mean_relative_scores
from utils.make_env import make_env
from utils.scheduler import linear_scheduler_sb3


# SPECIFY ENVS
ENV = GasTurbineBatteryRenewablesDemandEnv
TRAIN_ENV_KWARGS = on_2017
TEST_ENV_KWARGS = on_2017

EXP_PARAMS = {
    'n_runs': 3,  # deterministic (keep at 1) unless multiple experts are used
    'seed': 22,
    'expert': 'aadqn',
    'rollout_folder': '1600_episodes',  # name of folder in which rollouts are saved
    'n_epochs': 5,  # BC training epochs
    'batch_size': 128,
    'lr': 7.33e-4,
    'ent_weight': 0.0445,
    'l2_weight': 2.56e-5,
    'net_arch': 'td3',
    'activation_fn': 'relu',
}

# EXPERT (takes 'run_0' if n_runs = 1)
path_to_expert = os.path.join(src_dir, 'log', TRAIN_ENV_KWARGS['env_name'], EXP_PARAMS['expert'], 'run',
                              'varied_configs', 'test')

with open(os.path.join(path_to_expert, 'inputs.json')) as f:
    inputs = json.load(f)

# ACTIONS
disc_actions = None
discretization_params = inputs.get('DISCRETE_ACTIONS', None)
if discretization_params != 'predefined' and discretization_params is not None:
    disc_actions = generate_discrete_actions(**discretization_params)


# LOG
CREATE_LOG = True
SAVE_PATH = os.path.join(src_dir, 'log', TRAIN_ENV_KWARGS['env_name'], 'bc', 'run', 'varied_configs', input('Save in folder: ')) \
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


train, test_inter, test_extra = plant_config_train_test_split()

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

    eval_env_inter = make_env(
        env=ENV,
        env_kwargs=TEST_ENV_KWARGS,
        path=None,
        update_plant_config=test_inter,
        flatten_obs=inputs['EXP_PARAMS']['flatten_obs'],
        discrete_actions=disc_actions,
        use_rollout_info_wrapper=True,
        frame_stack=inputs['EXP_PARAMS']['frame_stack'],
        minmax_scaling=inputs['EXP_PARAMS'].get('minmax_scaling', False),
        use_predefined_action_wrapper=inputs['EXP_PARAMS'].get('use_predefined_action_wrapper', None),
        norm_obs=inputs['EXP_PARAMS']['norm_obs'],
        norm_reward=inputs['EXP_PARAMS']['norm_reward'],
    )

    eval_env_extra = make_env(
        env=ENV,
        env_kwargs=TEST_ENV_KWARGS,
        path=None,
        update_plant_config=test_extra,
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

    # Load rollouts from a file
    with open(os.path.join(path_to_expert, 'rollouts', EXP_PARAMS['rollout_folder'], f'run_{run}', 'rollouts.pkl'), "rb") as f:
        rollouts = pickle.load(f)

    transitions = rollout.flatten_trajectories(rollouts)

    policy = policies.ActorCriticPolicy(
        observation_space=eval_env_inter.observation_space,
        action_space=eval_env_inter.action_space,
        lr_schedule=lambda _: torch.finfo(torch.float32).max,  # no effect as optimizer is overwritten by BC class
        net_arch=net_arch_dict[EXP_PARAMS['net_arch']],
        activation_fn=activation_fn_dict[EXP_PARAMS['activation_fn']],
    )

    bc_trainer = bc.BC(
        observation_space=eval_env_inter.observation_space,
        action_space=eval_env_inter.action_space,
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

    # Score env for interpolation
    eval_env_inter.training = False
    eval_env_inter.norm_reward = False
    inter_rewards, _ = evaluate_policy(
        model=bc_trainer.policy,
        env=eval_env_inter,
        n_eval_episodes=len(test_inter),
        return_episode_rewards=True
    )
    eval_env_inter.close()

    # Score env for extrapolation
    eval_env_extra.training = False
    eval_env_extra.norm_reward = False
    extra_rewards, _ = evaluate_policy(
        model=bc_trainer.policy,
        env=eval_env_extra,
        n_eval_episodes=len(test_extra),
        return_episode_rewards=True
    )
    eval_env_extra.close()

    # Get relative scores
    mean_rel_inter_score, mean_rel_extra_score = calculate_mean_relative_scores(inter_rewards, extra_rewards)

    print(f"EVAL ENV REWARDS (INTER): {mean_rel_inter_score}")
    print(f"EVAL ENV REWARDS (EXTRA): {mean_rel_extra_score}")

    if run_path is not None:
        log_data = {
            'time': time.time() - start,
            'rewards_inter': inter_rewards,
            'mean_rel_inter_score': mean_rel_inter_score,
            'rewards_extra': extra_rewards,
            'mean_rel_extra_score': mean_rel_extra_score,
        }

        with open(os.path.join(run_path, 'output.json'), 'w') as f:
            json.dump(log_data, f)


# GET STATISTICS FROM MULTIPLE RUNS
if CREATE_LOG and EXP_PARAMS['n_runs'] > 1:
    get_inter_extra_stats_from_output(path=SAVE_PATH)
