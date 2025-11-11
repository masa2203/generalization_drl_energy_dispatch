import torch
import torch.nn as nn
import gymnasium
from gymnasium import spaces
from imitation.rewards.reward_nets import BasicShapedRewardNet, BasicRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


net_arch_dict = {
    "tiny": {"pi": [64], "vf": [64], "qf": [64]},
    'small': {"pi": [64, 64], "vf": [64, 64], "qf": [64, 64]},
    'medium': {"pi": [128, 128], "vf": [128, 128], "qf": [128, 128]},
    'large': {"pi": [128, 256, 128], "vf": [128, 256, 128], "qf": [128, 256, 128]},
    'extra_large': {"pi": [256, 512, 512, 256], "vf": [256, 512, 512, 256], "qf": [256, 512, 512, 256]},
    'ddpg': {"pi": [400, 300], "vf": [400, 300], "qf": [400, 300]},
    'td3': {"pi": [400, 300], "vf": [400, 300], "qf": [400, 300]},
    'sac': {"pi": [256, 256], "vf": [256, 256], "qf": [256, 256]},
}

activation_fn_dict = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
}

imitation_nets = {
    'tiny': (32, ),
    'small': (128, ),
    'medium': (32, 32),
    'large': (128, 128),
    'extra_large': (256, 512, 256),
}

"""
Defaults:

PPO:
dict(pi=[64, 64], vf=[64, 64])
nn.Tanh

DDPG: 
[400, 300] for actor and critic(s)
nn.ReLU

TD3: 
[400, 300] for actor and critic(s)
nn.ReLU, final tanh for actor

DQN:
[64, 64]
nn.ReLU

SAC:
[256, 256]
nn.ReLU

A2C:
dict(pi=[64, 64], vf=[64, 64])
nn.Tanh
"""


class CustomLSTM(BaseFeaturesExtractor):
    """
    LSTM on frame dimension.
    """

    def __init__(
            self,
            observation_space: spaces.Box,
            frame_stack: int = 4,
            dropout: float = 0.0,
            lstm_layer_size: int = 32,
            lstm_num_layer: int = 2,
            ann_net_shape: list = [64],
            activation: str = 'relu',
            batch_norm: bool = False,  # Only applied to linear layers
    ):
        super().__init__(observation_space, ann_net_shape[-1])
        activation = activation_fn_dict[activation]()

        self.frame_stack = frame_stack
        self.features = int(observation_space.shape[0] / frame_stack)

        # LSTM with batch_first=True needs (batch, sequence, features)
        self.lstm = nn.LSTM(input_size=self.features,
                            hidden_size=lstm_layer_size,
                            num_layers=lstm_num_layer,
                            batch_first=True)

        # ANN
        ann_layers = [nn.Linear(in_features=lstm_layer_size,
                                out_features=ann_net_shape[0])]
        if batch_norm:
            ann_layers.append(nn.BatchNorm1d(ann_net_shape[0]))

        ann_layers.append(activation)
        ann_layers.append(nn.Dropout(dropout))

        for i in range(len(ann_net_shape) - 1):
            ann_layers.append(nn.Linear(ann_net_shape[i], ann_net_shape[i + 1]))
            if batch_norm:
                ann_layers.append(nn.BatchNorm1d(ann_net_shape[i + 1]))
            ann_layers.append(activation)
            ann_layers.append(nn.Dropout(dropout))

        # No output layer (handled by 'net_arch'-argument of SB3

        # Wrap with sequential module
        self.ann = nn.Sequential(*ann_layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.view(-1, self.frame_stack, self.features)
        x, _ = self.lstm(x)
        x = self.ann(x[:, -1, :])
        return x


class CustomANN(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: spaces.Box,
            ann_net_shape: list = [64],
            activation: str = 'relu',
            batch_norm: bool = False,
            dropout: float = 0.0
    ):
        super().__init__(observation_space, ann_net_shape[-1])
        activation = activation_fn_dict[activation]()

        ann_layers = [nn.Linear(in_features=observation_space.shape[0],
                                out_features=ann_net_shape[0])]
        if batch_norm:
            ann_layers.append(nn.BatchNorm1d(ann_net_shape[0]))

        ann_layers.append(activation)
        ann_layers.append(nn.Dropout(dropout))

        for i in range(len(ann_net_shape) - 1):
            ann_layers.append(nn.Linear(ann_net_shape[i], ann_net_shape[i + 1]))
            if batch_norm:
                ann_layers.append(nn.BatchNorm1d(ann_net_shape[i + 1]))
            ann_layers.append(activation)
            ann_layers.append(nn.Dropout(dropout))

        # No output layer (handled by 'net_arch'-argument of SB3

        # Wrap with sequential module
        self.ann = nn.Sequential(*ann_layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.ann(observations)


class CustomCNN(BaseFeaturesExtractor):
    """
    Applies one or more convolutional layers (1D) along the frame dimension.

    NatureCNN (for reference):
    1) 32 filters, 8x8, stride 4, Relu
    2) 64 filters, 4x4, stride 2, Relu
    3) FC with 512 neurons, Relu
    """

    def __init__(
            self,
            observation_space: spaces.Box,
            frame_stack: int,
            cnn_net_shape: list = [32],
            cnn_kernel_size: int = 3,
            cnn_stride: int = 1,
            ann_net_shape: list = [64],
            activation: str = 'relu',
            batch_norm: bool = False,
            dropout: float = 0.0
    ):
        super().__init__(observation_space, ann_net_shape[-1])
        activation = activation_fn_dict[activation]()

        self.frame_stack = frame_stack
        self.features = int(observation_space.shape[0] / frame_stack)

        # 1D CNN needs (batch, features, sequence)
        # CNN input layer
        cnn_layers = [nn.Conv1d(in_channels=self.features,
                                out_channels=cnn_net_shape[0],
                                kernel_size=cnn_kernel_size,
                                stride=cnn_stride)]

        if batch_norm:
            cnn_layers.append(nn.BatchNorm1d(cnn_net_shape[0]))

        cnn_layers.append(activation)
        cnn_layers.append(nn.Dropout(dropout))

        for i in range(len(cnn_net_shape) - 1):
            cnn_layers.append(nn.Conv1d(in_channels=cnn_net_shape[i],
                                        out_channels=cnn_net_shape[i + 1],
                                        kernel_size=cnn_kernel_size,
                                        stride=cnn_stride))
            if batch_norm:
                cnn_layers.append(nn.BatchNorm1d(cnn_net_shape[i + 1]))
            cnn_layers.append(activation)
            cnn_layers.append(nn.Dropout(dropout))

        # Wrap with sequential module
        self.cnn = nn.Sequential(*cnn_layers)

        # ANN
        # Compute size of CNN output after flattening
        in_features = ((frame_stack - cnn_kernel_size) // cnn_stride) + 1
        for i in range(len(cnn_net_shape) - 1):
            in_features = ((in_features - cnn_kernel_size) // cnn_stride) + 1
        in_features *= cnn_net_shape[-1]

        ann_layers = [nn.Linear(in_features=in_features,
                                out_features=ann_net_shape[0])]
        if batch_norm:
            ann_layers.append(nn.BatchNorm1d(ann_net_shape[0]))

        ann_layers.append(activation)
        ann_layers.append(nn.Dropout(dropout))

        for i in range(len(ann_net_shape) - 1):
            ann_layers.append(nn.Linear(ann_net_shape[i], ann_net_shape[i + 1]))
            if batch_norm:
                ann_layers.append(nn.BatchNorm1d(ann_net_shape[i + 1]))
            ann_layers.append(activation)
            ann_layers.append(nn.Dropout(dropout))

        # No output layer (handled by 'net_arch'-argument of SB3

        # Wrap with sequential module
        self.ann = nn.Sequential(*ann_layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.view(-1, self.frame_stack, self.features).transpose(1, 2)
        x = self.cnn(x)
        x = x.flatten(1)
        x = self.ann(x)
        return x


class CustomFeatureCNN(BaseFeaturesExtractor):
    """
    Convolutional operation with 1D CNN layers on the feature dimension.
    """
    def __init__(self,
                 observation_space: spaces.Box,
                 features_dim: int = 128,
                 dropout: float = 0.1,
                 batch_norm: bool = False,
                 frame_stack: int = 4):
        super().__init__(observation_space, features_dim=features_dim)

        self.frame_stack = frame_stack
        self.features = int(observation_space.shape[0] / frame_stack)
        cnn_net_shape = [16, 32]
        kernel_size = 3
        stride = 1

        # 1D CNN needs (batch, features, sequence)
        # CNN input layer
        cnn_layers = [
            nn.Conv1d(in_channels=self.frame_stack,
                      out_channels=cnn_net_shape[0],
                      kernel_size=kernel_size,
                      stride=stride)
        ]
        if batch_norm:
            cnn_layers.append(nn.BatchNorm1d(cnn_net_shape[0]))

        cnn_layers += [nn.ReLU(), nn.Dropout(dropout)]

        for i in range(len(cnn_net_shape) - 1):
            cnn_layers.append(nn.Conv1d(in_channels=cnn_net_shape[i],
                                        out_channels=cnn_net_shape[i + 1],
                                        kernel_size=kernel_size,
                                        stride=stride))
            if batch_norm:
                cnn_layers.append(nn.BatchNorm1d(cnn_net_shape[i + 1]))
            cnn_layers += [nn.ReLU(), nn.Dropout(dropout)]

        # Wrap with sequential module
        self.cnn = nn.Sequential(*cnn_layers)

        # ANN
        # Compute size of CNN output after flattening
        in_features = ((self.features - kernel_size) // stride) + 1
        for i in range(len(cnn_net_shape) - 1):
            in_features = ((in_features - kernel_size) // stride) + 1
        in_features *= cnn_net_shape[-1]

        self.linear = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.view(-1, self.frame_stack, self.features)
        x = self.cnn(x)
        x = x.flatten(1)
        x = self.linear(x)
        return x


def create_reward_net(
        observation_space: gymnasium.Space,
        action_space: gymnasium.Space,
        is_shaped: bool = False,
        norm_input: bool = False,
        reward_hid_sizes=None,
        potential_hid_sizes=None,
        gamma=None,
):
    """
    Creates a reward network based on the provided parameters.

    Args:
        observation_space: The observation space of the environment.
        action_space: The action space of the environment.
        is_shaped (bool): Whether to use a shaped network (i.e. with potential).
        norm_input (bool): Whether to normalize inputs using RunningNorm.
        reward_hid_sizes (tuple, optional): Hidden layer sizes for the reward network. Only used for 'shaped'.
        potential_hid_sizes (tuple, optional): Hidden layer sizes for the potential network. Only used for 'shaped'.
        gamma (float, optional): Discount factor for shaping. Only used for 'shaped'.

    Returns:
        Reward network instance.

    Raises:
        ValueError: If required parameters for 'shaped' are missing or if an unsupported type is provided.
    """
    normalize_input_layer = RunningNorm if norm_input else None

    if is_shaped:
        # Validate that required parameters for shaped reward net are provided
        if reward_hid_sizes is None or potential_hid_sizes is None or gamma is None:
            raise ValueError(
                "For shaped net, 'reward_hid_sizes', 'potential_hid_sizes', and 'gamma' must be provided.")

            # Create a shaped reward net
        return BasicShapedRewardNet(
            observation_space=observation_space,
            action_space=action_space,
            normalize_input_layer=normalize_input_layer,
            reward_hid_sizes=reward_hid_sizes,
            potential_hid_sizes=potential_hid_sizes,
            discount_factor=gamma
        )
    else:
        # Create a basic reward net
        return BasicRewardNet(
            observation_space=observation_space,
            action_space=action_space,
            normalize_input_layer=normalize_input_layer
        )