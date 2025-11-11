from typing import Optional, Dict, Any, Union

from gymnasium.wrappers import FlattenObservation
import stable_baselines3.common.vec_env
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from envs.p2g import P2G
from utils.wrappers import *


# Define a mapping of wrapper names to their classes
predefined_action_wrappers = {
    "on_default": PreDefinedDiscreteActions,
    "ab_default": ABPreDefinedDiscreteActions,
    "ab_small": ABSmallPreDefinedDiscreteActions,
}


def make_env(env,
             env_kwargs: dict,
             tracking: bool = False,
             allow_early_resets: bool = True,
             path: Optional[str] = None,
             use_random_episodes: Dict[str, Any] | bool = False,
             update_plant_config: list[dict] | bool = False,
             minmax_scaling: bool = False,
             use_predefined_action_wrapper: Optional[Union[str, bool]] = None,
             p2g_soc_penalty: Optional[tuple] = None,
             p2g_inactivity_penalty: Optional[dict] = None,
             flatten_obs: bool = True,
             use_rollout_info_wrapper: bool = False,
             discrete_actions: Optional[list] = None,
             frame_stack: Optional[int] = None,
             use_vec_env: bool = True,
             norm_obs: bool = True,
             norm_reward: bool = True,
             gamma: float = 0.99,
             verbose: Optional[bool] = False,
             ):
    """
    Creates a gym environment and applies a set of wrappers.

    :param env: A subclass of gym.Env that represents the environment.
    :param env_kwargs: A dictionary that represents the keyword arguments to pass to the environment.
    :param tracking: A boolean that indicates whether to track the variables. Default is False.
    :param allow_early_resets: A boolean that indicates whether to allow early resets. Default is True.
    :param path: A string that represents the path to save the monitor. Default is None.
    :param use_random_episodes: Whether to randomly sample episodes from provided dataset. Default is False.
    :param update_plant_config: Whether to update the plant configuration after each episode. Default is False.
                            If None, the wrapper is not applied. Default is None.
    :param minmax_scaling: A boolean that indicates whether observations are scaled to [0,1] range. Default is False.
    :param use_predefined_action_wrapper: A string that indicates which predefined discrete action wrapper
                                            is used. Default is None, i.e. the original action space is used.
    :param p2g_soc_penalty: A tuple (penalty weight, max SOC) for the low P2G SOC penalty. Default is None.
    :param p2g_inactivity_penalty: A dict with the parameters of the P2G inactivity penalty. Default is None.
    :param flatten_obs: A boolean that indicates whether to flatten the observation. Default is True.
    :param use_rollout_info_wrapper: A boolean that indicates whether to use imitation's RolloutInfoWrapper.
    :param discrete_actions: A list that represents the discrete actions. Default is None.
    :param frame_stack: An integer that represents the number of frames to stack. Default is None.
    :param use_vec_env: A boolean that indicates whether to use VecNormalize. Default is True.
                        If False, no normalization is done.
    :param norm_obs: A boolean that indicates whether to normalize the observation. Default is True.
    :param norm_reward: A boolean that indicates whether to normalize the reward. Default is True.
    :param gamma: A float that represents the gamma value. Default is 0.99.
    :param verbose: A boolean that represents whether the environment is verbose or not.
    :return: A stable_baselines3.common.vec_env.VecNormalize object that represents the wrapped environment.
    """
    e = Monitor(env=env(**env_kwargs, tracking=tracking, verbose=verbose),
                allow_early_resets=allow_early_resets,  # allow finish rollout for PPO -> throws error otherwise
                filename=path)

    if use_random_episodes:
        e = RandomEpisodesWrapper(e, mode=use_random_episodes['mode'], num=use_random_episodes['num'])

    if update_plant_config:
        if hasattr(e.unwrapped, "p2g") and isinstance(e.get_wrapper_attr("p2g"), P2G):
            e = UpdateEnvConfigP2G(e, configs=update_plant_config)
        else:
            e = UpdateEnvConfig(e, configs=update_plant_config)

    if minmax_scaling:
        # Safety check: Don't choose two normalization techniques simultaneously.
        assert not (use_vec_env and norm_obs), "Both Minmax-Scaling and observation norm based on running stats chosen."
        e = MinMaxScaler(e)

    # HANDLING OF PREDEFINED DISCRETE ACTIONS
    # --------------------------------------------------------------------------------------
    # Default wrapper selection based on the environment
    on_envs = ['on_3y_train', 'on_3y_test', 'on_24h', 'on_168h', 'on_2015', 'on_2016',
                'on_2017', 'on_2018', 'on_2019', 'on_2020']
    if use_predefined_action_wrapper is True:  # Backward compatibility: Choose default wrapper for the environment
        if env_kwargs.get("env_name") in on_envs:
            use_predefined_action_wrapper = "on_default"
        elif hasattr(e.unwrapped, "p2g") and isinstance(e.get_wrapper_attr("p2g"), P2G):
            use_predefined_action_wrapper = "ab_default"
        else:
            raise ValueError(f"Cannot determine a default discrete action wrapper for the environment: "
                             f"{env_kwargs.get('env_name')}")

    # Apply the selected wrapper if valid and specified
    if isinstance(use_predefined_action_wrapper, str):  # Check if a specific wrapper name is provided
        if use_predefined_action_wrapper in predefined_action_wrappers:
            wrapper_class = predefined_action_wrappers[use_predefined_action_wrapper]
            e = wrapper_class(e)
        else:
            raise ValueError(f"Invalid discrete_action_wrapper: {use_predefined_action_wrapper}. "  
                             f"Valid options are: {list(predefined_action_wrappers.keys())}")
    elif use_predefined_action_wrapper is None:
        # Rescale action space only if no discrete action wrapper is applied.
        # Rescale from [0,1] to [-1,1], applies only to first dimension (GT).
        # Must rescale before discretization (if applicable) as RescaleActionSpace doesn't support discrete spaces.
        e = RescaleActionSpace(e)
    # --------------------------------------------------------------------------------------

    if p2g_soc_penalty is not None:
        e = P2GSOCPenalty(e, penalty=p2g_soc_penalty[0], max_soc=p2g_soc_penalty[1])

    if p2g_inactivity_penalty is not None:
        e = P2GInactivityPenalty(e, **p2g_inactivity_penalty)

    if flatten_obs:
        e = FlattenObservation(e)

    if use_rollout_info_wrapper:
        e = RolloutInfoWrapper(e)

    # Add discrete action wrapper
    if discrete_actions is not None:
        e = DiscreteActions(e, discrete_actions)

    e = DummyVecEnv([lambda: e])

    # Stack observation
    if frame_stack is not None:
        e = stable_baselines3.common.vec_env.VecFrameStack(e, n_stack=frame_stack)

    if use_vec_env:
        e = VecNormalize(e, norm_obs=norm_obs, norm_reward=norm_reward, gamma=gamma)

    return e
