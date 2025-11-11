import os
import json
import pickle
import random

import imitation
from imitation.data import rollout
from imitation.policies.serialize import load_policy

from config import src_dir
from envs.environments import *
from envs.env_params import *
from utils.make_env import make_env
from utils.utilities import set_seeds, plant_config_train_test_split

"""
GENERATES DEMONSTRATION DATA AND SAVES THE ROLLOUTS TO BE USED BY BC, GAIL, & AIRL.
- Uses trained and saved expert (e.g. aaDQN)
- Can handle multiple runs (i.e. multiple experts trained with different random seeds) in which case it produces
    separate rollout folders for each seed.
"""

# SPECIFY ENVS FOR DEMONSTRATION
ENV = GasTurbineBatteryRenewablesDemandEnv
TRAIN_ENV_KWARGS = on_2017


EXP_PARAMS = {
    'n_runs': 1,  # also uses different expert runs
    'seed': 22,
    'expert': 'aadqn',
    'rollout_episodes': 10,  # expert data collection
}


# EXPERT (takes 'run_0' if n_runs = 1)
path_to_expert = os.path.join(src_dir, 'log', TRAIN_ENV_KWARGS['env_name'], EXP_PARAMS['expert'], 'run',
                              'varied_configs', 'test')
with open(os.path.join(path_to_expert, 'inputs.json')) as f:
    inputs = json.load(f)

# LOG
CREATE_LOG = False
SAVE_PATH = os.path.join(path_to_expert, 'rollouts', f'{EXP_PARAMS["rollout_episodes"]}_episodes') \
    if CREATE_LOG else None


train, test_inter, test_extra = plant_config_train_test_split()

if EXP_PARAMS['expert'] == 'aadqn':
    imitation.policies.serialize._add_stable_baselines_policies_from_file(dict(aadqn="stable_baselines3:DQN"))
if EXP_PARAMS['expert'] == 'aappo':
    imitation.policies.serialize._add_stable_baselines_policies_from_file(dict(aappo="stable_baselines3:PPO"))

for run in range(EXP_PARAMS['n_runs']):
    # UPDATE SEED
    seed = EXP_PARAMS['seed'] + run
    set_seeds(seed)
    rng = np.random.default_rng(seed)
    print('|| Run #{} | Seed #{} ||'.format(run, seed))

    # Create path for logging
    run_path = os.path.join(SAVE_PATH, f'run_{run}') if SAVE_PATH is not None else None
    if SAVE_PATH is not None:
        os.makedirs(run_path, exist_ok=True)

    demo_env = make_env(
        env=ENV,
        env_kwargs=TRAIN_ENV_KWARGS,
        path=None,
        update_plant_config=random.choices(train, k=EXP_PARAMS['rollout_episodes']),
        flatten_obs=inputs['EXP_PARAMS']['flatten_obs'],
        use_rollout_info_wrapper=True,
        frame_stack=inputs['EXP_PARAMS']['frame_stack'],
        minmax_scaling=inputs['EXP_PARAMS'].get('minmax_scaling', False),
        use_predefined_action_wrapper=inputs['EXP_PARAMS'].get('use_predefined_action_wrapper', None),
        norm_obs=inputs['EXP_PARAMS']['norm_obs'],
        norm_reward=inputs['EXP_PARAMS']['norm_reward'],
    )

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

    if SAVE_PATH is not None:
        # Save rollouts to a file
        with open(os.path.join(run_path, 'rollouts.pkl'), "wb") as f:
            pickle.dump(rollouts, f)
