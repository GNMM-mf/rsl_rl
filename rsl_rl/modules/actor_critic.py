#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        use_tanh_output=False,  # 是否在输出层使用tanh激活函数，将动作限制在[-1, 1]
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
                # 如果启用tanh输出层，在最后一层后添加tanh激活函数
                if use_tanh_output:
                    actor_layers.append(nn.Tanh())
                    print(f"[INFO] Actor网络输出层使用tanh激活函数, 动作将被限制在[-1, 1]范围内")
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise - 使用传入的 init_noise_std 参数（如果提供），否则使用默认值
        # 注意：init_noise_std 是 std 值，需要转换为 log_std
        if init_noise_std is not None and init_noise_std > 0:
            init_log_std = torch.log(torch.tensor(init_noise_std, dtype=torch.float32))
        else:
            # 默认值：-1.0 对应 std ≈ 0.37
            init_log_std = -1.0
        self.std = nn.Parameter(init_log_std * torch.ones(num_actions))
        print(f"[INFO] Action noise initialized: log_std={init_log_std:.4f}, std={torch.exp(init_log_std):.4f}")
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # 注册 hook 来监控和限制 std 参数的值（防止在多进程环境中变成 NaN/Inf）
        def std_hook(grad):
            if grad is not None:
                # 检查梯度是否有问题
                if torch.any(torch.isnan(grad)) or torch.any(torch.isinf(grad)):
                    print(f"[WARNING] std 参数的梯度有 NaN 或 Inf 值！")
                    grad = torch.where(torch.isnan(grad) | torch.isinf(grad), 
                                     torch.zeros_like(grad), 
                                     grad)
                # 限制梯度大小，防止梯度爆炸
                grad = torch.clamp(grad, min=-1.0, max=1.0)
            return grad
        
        # 注册梯度 hook（如果支持）
        if hasattr(self.std, 'register_hook'):
            self.std.register_hook(std_hook)

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        
        # 使用log_std参数，并添加严格的clamp限制
        # 首先直接修复 self.std 参数本身（如果它是 NaN/Inf）
        if torch.any(torch.isnan(self.std)) or torch.any(torch.isinf(self.std)):
            print(f"[WARNING] 检测到 self.std 参数有 NaN 或 Inf 值，直接修复参数本身！")
            print(f"[WARNING] self.std 范围: min={self.std.min().item() if not torch.isnan(self.std.min()) else 'NaN'}, max={self.std.max().item() if not torch.isnan(self.std.max()) else 'NaN'}")
            # 直接修复参数本身（使用 no_grad 避免影响梯度计算）
            with torch.no_grad():
                self.std.data = torch.where(
                    torch.isnan(self.std.data) | torch.isinf(self.std.data),
                    torch.full_like(self.std.data, -1.0),  # 默认 log_std = -1.0 (std ≈ 0.37)
                    self.std.data
                )
                # 限制范围
                self.std.data = torch.clamp(self.std.data, min=-4.0, max=0.5)
        
        # 使用修复后的参数
        log_std = self.std
        
        # 限制log_std的范围，防止std过大或过小
        log_std = torch.clamp(log_std, min=-4.0, max=0.5)  # std范围: [0.018, 1.65]
        
        # 转换为std
        std = torch.exp(log_std) + 1e-6
        
        # 处理NaN和Inf值（双重保险）
        std = torch.where(torch.isnan(std) | torch.isinf(std), 
                         torch.full_like(std, 1e-6), 
                         std)
        # 确保std >= 0（防止负值，虽然理论上不应该发生）
        std = torch.clamp(std, min=1e-6, max=2.0)  # 进一步限制std上限
        
        # 最终安全检查：确保所有元素都 >= 0
        std = torch.maximum(std, torch.full_like(std, 1e-6))
        
        # 检查是否有任何负值（调试用）
        if torch.any(std < 0):
            print(f"[ERROR] 检测到负的std值！min(std)={std.min().item()}, max(std)={std.max().item()}")
            print(f"[ERROR] log_std范围: min={log_std.min().item()}, max={log_std.max().item()}")
            print(f"[ERROR] self.std参数范围: min={self.std.min().item()}, max={self.std.max().item()}")
            std = torch.clamp(std, min=1e-6, max=2.0)
        
        # 创建分布前再次验证
        scale = mean * 0.0 + std
        if torch.any(scale < 0):
            print(f"[ERROR] 创建分布前检测到负的scale值！强制修复...")
            scale = torch.maximum(scale, torch.full_like(scale, 1e-6))
        
        # 最终验证：确保 scale 中没有 NaN/Inf
        if torch.any(torch.isnan(scale)) or torch.any(torch.isinf(scale)):
            print(f"[ERROR] 创建分布前检测到 scale 有 NaN 或 Inf 值！强制修复...")
            scale = torch.where(torch.isnan(scale) | torch.isinf(scale),
                               torch.full_like(scale, 1e-6),
                               scale)
        
        self.distribution = Normal(mean, scale, validate_args=False)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        # 最终安全检查：确保分布有效
        if self.distribution is None:
            raise RuntimeError("Distribution is None after update_distribution")
        # 检查 stddev 是否有效
        if torch.any(self.distribution.stddev < 0):
            print(f"[ERROR] Distribution has negative stddev! min={self.distribution.stddev.min().item()}")
            # 强制修复：重新创建分布
            mean = self.distribution.mean
            std = torch.maximum(self.distribution.stddev, torch.full_like(self.distribution.stddev, 1e-6))
            self.distribution = Normal(mean, std, validate_args=False)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        # 处理字典格式的观察（推理时从normalized_policy传入）
        if isinstance(observations, dict):
            policy_obs = observations.get("policy", observations)
            actions_mean = self.actor(policy_obs)
        else:
            # 向后兼容：如果传入的是张量，直接使用
            actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
