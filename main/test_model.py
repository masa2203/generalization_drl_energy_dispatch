import os
import json
import time
import statistics
from stable_baselines3 import PPO, SAC, DDPG, DQN, A2C, TD3
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

from config import src_dir
from envs.environments import *
from envs.env_params import *
from utils.make_env import make_env
from utils.utilities import generate_discrete_actions
from utils.logger import get_env_log_data

"""LOAD A SAVED MODEL AND ENV -> TEST ON ANOTHER ENV"""
AGENTS = {
    'ppo': PPO,
    'sac': SAC,
    'a2c': A2C,
    'dqn': DQN,
    'ddpg': DDPG,
    'td3': TD3,
    'aadqn': DQN,
    'aappo': PPO,
}


def test_saved_model(
        agent,
        env,
        inputs: dict,
        test_env_kwargs: dict,
        saved_env_path: str,
        saved_model_path: str,
        save_path: Optional[str] = None,
        run_id: str = 'run_0',
):
    """
    Loads a saved model & env and evaluates it on a test env.

    Returns mean reward on a single evaluation episode.

    """
    agent = AGENTS[agent]

    # ACTIONS
    disc_actions = None
    discretization_params = inputs.get('DISCRETE_ACTIONS', None)
    if discretization_params != 'predefined' and discretization_params is not None:
        disc_actions = generate_discrete_actions(**discretization_params)

    env = make_env(
        env=env,
        env_kwargs=test_env_kwargs,
        flatten_obs=inputs['EXP_PARAMS']['flatten_obs'],
        discrete_actions=disc_actions,
        frame_stack=inputs['EXP_PARAMS']['frame_stack'],
        minmax_scaling=inputs['EXP_PARAMS'].get('minmax_scaling', False),
        use_predefined_action_wrapper=inputs['EXP_PARAMS'].get(
            'use_predefined_action_wrapper',  # New argument
            inputs['EXP_PARAMS'].get('use_predefined_discrete_actions', None)  # Fallback to old argument
        ),
        use_vec_env=False,  # Don't call VecNormalize (called with .load() below)
    )

    env = VecNormalize.load(saved_env_path, env)
    env.training = False
    env.norm_reward = False
    env.unwrapped.envs[0].unwrapped.start_tracking()  # Start tracking env variables for evaluation
    model = agent.load(saved_model_path, env=env)

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1)
    log_data = get_env_log_data(env=env, mean_reward=mean_reward, start_time=time.time())

    # print(f"mean_reward: {mean_reward:,.2f} +/- {std_reward:,.2f}")

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, f'{run_id}_test.json'), 'w') as f:
            json.dump(log_data, f)

    return mean_reward


# ENV (Must be same for train & test)
ENV = GasTurbineBatteryRenewablesDemandEnv
# ENV = GasTurbineP2GBatteryRenewablesDemandEnv

# ENV-PARAMS FOR TEST ENV
ENV_KWARGS = on_3y_test

ENV_KWARGS['state_vars'] = dict(
    re_power=(0.0, 50.0),
    sin_h=(-1.0, 1.0),
    cos_h=(-1.0, 1.0),
    sin_w=(-1.0, 1.0),
    cos_w=(-1.0, 1.0),
    sin_m=(-1.0, 1.0),
    cos_m=(-1.0, 1.0),
    workday=(0.0, 1.0),
    gas_price=(0.0, 10.0)
    # pool_price=(0.0, 1000.0),
)

# PATH TO SAVED EXPERIMENT (must include inputs.json)
agent = 'aadqn'

path = os.path.join(src_dir, 'log', 'on_3y_train', agent, 'run', 'test')

# Save path for loaded model
# SAVE_PATH = None
SAVE_PATH = os.path.join(path, f'tested_on_{ENV_KWARGS["env_name"]}')

with open(os.path.join(path, 'inputs.json')) as f:
    inputs = json.load(f)

train_rewards = []
test_rewards = []

# Iterate over subdirectories
for subdir in os.listdir(path):
    subfolder_path = os.path.join(path, subdir)
    if os.path.isdir(subfolder_path):
        saved_model = os.path.join(subfolder_path, 'model.zip')
        env_path = os.path.join(subfolder_path, 'env.pkl')

        with open(os.path.join(subfolder_path, 'output.json')) as f:
            outputs = json.load(f)

        train_rewards.append(outputs['reward_sum'])

        test_rew = test_saved_model(
            agent=agent,
            env=ENV,
            inputs=inputs,
            save_path=SAVE_PATH if SAVE_PATH is not None else None,  # Where to save test results with loaded agent
            run_id=subdir,
            test_env_kwargs=ENV_KWARGS,
            saved_model_path=saved_model,
            saved_env_path=env_path,
        )

        test_rewards.append(test_rew)

# print(train_rewards, test_rewards)
print(
    f'Train reward sum mean: {statistics.mean(train_rewards):,.2f} | Std-dev.: {statistics.stdev(train_rewards):,.2f}')
print(f'Test reward sum mean: {statistics.mean(test_rewards):,.2f} | Std-dev.: {statistics.stdev(test_rewards):,.2f}')
