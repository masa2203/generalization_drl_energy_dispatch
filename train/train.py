import os
import time
import json
import copy
from typing import Optional

from stable_baselines3 import PPO, SAC, DDPG, DQN, A2C, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure

from utils.callbacks import ProgressBarManager
from utils.make_env import make_env
from utils.logger import get_env_log_data
from utils.scheduler import linear_scheduler_sb3

AGENTS = {
    'ppo': PPO,
    'sac': SAC,
    'a2c': A2C,
    'dqn': DQN,
    'ddpg': DDPG,
    'td3': TD3,
}


def train_rl_agent(
        agent: str,
        run: int,
        path: Optional[str],
        exp_params: dict,
        env_id: str,
        env_kwargs: dict,
        rl_params: dict,
        verbose: int = 0,
        discrete_actions: Optional[list] = None,
        logger_type: Optional[list] = None,
        save_model_and_env: Optional[bool] = False,
):
    """
    Trains a reinforcement learning agent.

    :param agent: A string that represents the name of the reinforcement learning agent.
    :param run: An integer that represents the run number.
    :param path: A string that represents the path to the directory where the data will be saved.
    :param exp_params: A dictionary that contains the experiment parameters.
    :param env_id: A string that represents the ID of the environment.
    :param env_kwargs: A dictionary that contains the keyword arguments for the environment.
    :param rl_params: A dictionary that contains the reinforcement learning parameters.
    :param verbose: An integer that represents the verbosity level.
    :param discrete_actions: A list of discrete actions.
    :param logger_type: A list that represents the logger type.
    :param save_model_and_env: If True, model and environment are saved after training.
    """

    agent = AGENTS[agent]

    if logger_type is None:
        logger_type = ['csv']  # Default: save progress.csv file if path is not None

    tt = int(exp_params['n_episodes'] * env_kwargs['modeling_period_h'] / env_kwargs['resolution_h'])  # total timesteps

    # Create path for logging
    run_path = os.path.join(path, f'run_{run}') if path is not None else None
    if path is not None:
        os.makedirs(run_path, exist_ok=True)

    # Update seed
    seed = exp_params['seed'] + run
    print('|| Run #{} | Seed #{} ||'.format(run, seed))

    start = time.time()

    # For envs with P2G system, check for delayed cost assignment
    train_env_kwargs = copy.deepcopy(env_kwargs)
    if 'p2g' in train_env_kwargs:
        train_env_kwargs['p2g']['assign_cost_at_discharge'] = exp_params.get('p2g_cost_assign_discharge', False)
        train_env_kwargs['p2g']['assign_lost_profits_at_discharge'] = exp_params.get(
            'p2g_lost_profits_assign_discharge', False)

    # GET ENV CONFIG
    minmax_scaling = exp_params.get('minmax_scaling', False)
    use_predefined_action_wrapper = exp_params.get(
        'use_predefined_action_wrapper',  # New argument
        exp_params.get('use_predefined_discrete_actions', None)  # Fallback to old argument
    )
    # CREATE ENVIRONMENT
    env = make_env(env=env_id,
                   env_kwargs=train_env_kwargs,
                   path=os.path.join(run_path, "train_monitor.csv") if path is not None else None,
                   minmax_scaling=minmax_scaling,
                   use_predefined_action_wrapper=use_predefined_action_wrapper,
                   p2g_soc_penalty=exp_params.get('p2g_soc_penalty', None),
                   p2g_inactivity_penalty=exp_params.get('p2g_inactivity_penalty', None),
                   flatten_obs=exp_params['flatten_obs'],
                   discrete_actions=discrete_actions,
                   frame_stack=exp_params['frame_stack'],
                   norm_obs=exp_params['norm_obs'],
                   norm_reward=exp_params['norm_reward'],
                   gamma=rl_params['gamma'],
                   verbose=bool(verbose),
                   )

    # Handling of schedulers (remove from dict and apply function)
    if 'use_lr_sched' in rl_params:
        use_lr_sched = rl_params.pop('use_lr_sched')
        if use_lr_sched:
            rl_params['learning_rate'] = linear_scheduler_sb3(rl_params['learning_rate'])
    if 'use_cr_sched' in rl_params:  # only PPO
        use_cr_sched = rl_params.pop('use_cr_sched')
        if use_cr_sched:
            rl_params['clip_range'] = linear_scheduler_sb3(rl_params['clip_range'])
    if 'use_cr_vf_sched' in rl_params:  # only PPO
        use_cr_vf_sched = rl_params.pop('use_cr_vf_sched')
        if use_cr_vf_sched and rl_params['clip_range_vf'] is not None:
            rl_params['clip_range_vf'] = linear_scheduler_sb3(rl_params['clip_range_vf'])

    # DEFINE MODEL
    model = agent(env=env, verbose=verbose, seed=seed, **rl_params)
    if path is not None:
        logger = configure(run_path, logger_type)
        model.set_logger(logger)
    with ProgressBarManager(total_timesteps=tt) as callback:
        # Evaluation during training
        if exp_params['eval_while_training']:
            eval_env = make_env(env=env_id,
                                env_kwargs=env_kwargs,
                                path=os.path.join(run_path, 'eval_monitor.csv') if path is not None else None,
                                minmax_scaling=minmax_scaling,
                                use_predefined_action_wrapper=use_predefined_action_wrapper,
                                flatten_obs=exp_params['flatten_obs'],
                                discrete_actions=discrete_actions,
                                frame_stack=exp_params['frame_stack'],
                                norm_obs=exp_params['norm_obs'],
                                norm_reward=exp_params['norm_reward'],
                                gamma=rl_params['gamma'],
                                )

            evaluate_policy(model, eval_env, n_eval_episodes=1)  # test this to add untrained agent's stats

            eval_callback = EvalCallback(eval_env=eval_env,
                                         n_eval_episodes=1,
                                         eval_freq=exp_params['eval_freq'],
                                         deterministic=True,
                                         # best_model_save_path=run_path,
                                         best_model_save_path=None,
                                         verbose=verbose)

            model.learn(total_timesteps=tt, callback=[eval_callback, callback])
        # Evaluation only after training
        else:
            model.learn(total_timesteps=tt, callback=callback)

    env.training = False
    env.norm_reward = False
    # env.env_method('start_tracking')  # Start tracking env variables for evaluation
    env.unwrapped.envs[0].unwrapped.start_tracking()  # Start tracking env variables for evaluation
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)

    print(f"mean_reward: {mean_reward:,.2f} +/- {std_reward:,.2f}")

    if run_path is not None:
        if save_model_and_env:
            model.save(os.path.join(run_path, 'model'))  # Save best model instead through callback
            env.save(os.path.join(run_path, 'env.pkl'))
        log_data = get_env_log_data(env=env, mean_reward=mean_reward, start_time=start)
        with open(os.path.join(run_path, 'output.json'), 'w') as f:
            json.dump(log_data, f)

    env.close()

    print()
    print(f'Execution time = {time.time() - start}s')
