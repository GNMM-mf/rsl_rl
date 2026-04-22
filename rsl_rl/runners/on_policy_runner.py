#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque
from importlib import import_module
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

import rsl_rl
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, EmpiricalNormalization
from rsl_rl.utils import store_code_state

# 确保使用本地的 ActorCritic 和 PPO 类（通过文件系统路径直接导入）
# 这样可以避免被 Isaac Lab 的版本覆盖
import importlib.util
import sys

# 获取当前文件的目录
_current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取 rsl_rl 模块的根目录（当前文件在 rsl_rl/runners/ 下，所以上一级就是 rsl_rl）
_rsl_rl_root = os.path.dirname(_current_dir)
# 将 rsl_rl 根目录添加到 sys.path，确保依赖项能正确导入
if _rsl_rl_root not in sys.path:
    sys.path.insert(0, _rsl_rl_root)

# 直接导入本地的 ActorCritic 类
_actor_critic_path = os.path.join(_rsl_rl_root, "modules", "actor_critic.py")
_actor_critic_recurrent_path = os.path.join(_rsl_rl_root, "modules", "actor_critic_recurrent.py")

# 导入 ActorCritic（需要先导入，因为 ActorCriticRecurrent 依赖它）
spec = importlib.util.spec_from_file_location("rsl_rl.modules.actor_critic", _actor_critic_path)
local_actor_critic_module = importlib.util.module_from_spec(spec)
# 设置模块的 __package__ 和 __name__，确保相对导入能正常工作
local_actor_critic_module.__package__ = "rsl_rl.modules"
local_actor_critic_module.__name__ = "rsl_rl.modules.actor_critic"
# 在执行模块之前，先将其注册到 sys.modules，这样其他模块导入时会使用这个版本
sys.modules["rsl_rl.modules.actor_critic"] = local_actor_critic_module
spec.loader.exec_module(local_actor_critic_module)
LocalActorCritic = local_actor_critic_module.ActorCritic

# 确保 rsl_rl.modules 也在 sys.modules 中
if "rsl_rl.modules" not in sys.modules:
    import types
    sys.modules["rsl_rl.modules"] = types.ModuleType("rsl_rl.modules")
    sys.modules["rsl_rl.modules"].__path__ = [os.path.join(_rsl_rl_root, "modules")]

# 导入 ActorCriticRecurrent（依赖 ActorCritic）
spec = importlib.util.spec_from_file_location("rsl_rl.modules.actor_critic_recurrent", _actor_critic_recurrent_path)
local_actor_critic_recurrent_module = importlib.util.module_from_spec(spec)
# 设置模块属性，确保依赖项能正确导入
local_actor_critic_recurrent_module.__package__ = "rsl_rl.modules"
local_actor_critic_recurrent_module.__name__ = "rsl_rl.modules.actor_critic_recurrent"
# 在执行模块之前，先将其注册到 sys.modules
sys.modules["rsl_rl.modules.actor_critic_recurrent"] = local_actor_critic_recurrent_module
# 导入 unpad_trajectories（从 rsl_rl.utils）
try:
    from rsl_rl.utils import unpad_trajectories
except ImportError:
    # 如果导入失败，尝试从文件系统导入
    _utils_path = os.path.join(_rsl_rl_root, "utils", "__init__.py")
    if os.path.exists(_utils_path):
        spec_utils = importlib.util.spec_from_file_location("rsl_rl.utils", _utils_path)
        utils_module = importlib.util.module_from_spec(spec_utils)
        utils_module.__package__ = "rsl_rl.utils"
        utils_module.__name__ = "rsl_rl.utils"
        sys.modules["rsl_rl.utils"] = utils_module
        spec_utils.loader.exec_module(utils_module)
        unpad_trajectories = utils_module.unpad_trajectories

spec.loader.exec_module(local_actor_critic_recurrent_module)
LocalActorCriticRecurrent = local_actor_critic_recurrent_module.ActorCriticRecurrent

# 确保使用本地的 RolloutStorage 类（通过文件系统路径直接导入）
# 先创建 rsl_rl.storage 包模块
if "rsl_rl.storage" not in sys.modules:
    import types
    storage_pkg = types.ModuleType("rsl_rl.storage")
    storage_pkg.__path__ = [os.path.join(_rsl_rl_root, "storage")]
    sys.modules["rsl_rl.storage"] = storage_pkg

_storage_path = os.path.join(_rsl_rl_root, "storage", "rollout_storage.py")
spec_storage = importlib.util.spec_from_file_location("rsl_rl.storage.rollout_storage", _storage_path)
storage_module = importlib.util.module_from_spec(spec_storage)
storage_module.__package__ = "rsl_rl.storage"
storage_module.__name__ = "rsl_rl.storage.rollout_storage"
# 设置依赖项
try:
    from rsl_rl.utils import split_and_pad_trajectories
    storage_module.split_and_pad_trajectories = split_and_pad_trajectories
except ImportError:
    # 如果导入失败，尝试从文件系统导入
    _utils_path = os.path.join(_rsl_rl_root, "utils", "__init__.py")
    if os.path.exists(_utils_path):
        spec_utils = importlib.util.spec_from_file_location("rsl_rl.utils", _utils_path)
        utils_module = importlib.util.module_from_spec(spec_utils)
        utils_module.__package__ = "rsl_rl.utils"
        utils_module.__name__ = "rsl_rl.utils"
        if "rsl_rl.utils" not in sys.modules:
            sys.modules["rsl_rl.utils"] = utils_module
        spec_utils.loader.exec_module(utils_module)
        storage_module.split_and_pad_trajectories = utils_module.split_and_pad_trajectories
# 在执行模块之前，先将其注册到 sys.modules，这样其他模块导入时会使用这个版本
sys.modules["rsl_rl.storage.rollout_storage"] = storage_module
spec_storage.loader.exec_module(storage_module)
LocalRolloutStorage = storage_module.RolloutStorage
# 确保 rsl_rl.storage 包也能导出 RolloutStorage（这样 from rsl_rl.storage import RolloutStorage 才能工作）
sys.modules["rsl_rl.storage"].RolloutStorage = LocalRolloutStorage
# 同时确保 rsl_rl.storage.__init__ 也能导出（模拟 __init__.py 的行为）
_storage_init_path = os.path.join(_rsl_rl_root, "storage", "__init__.py")
if os.path.exists(_storage_init_path):
    spec_storage_init = importlib.util.spec_from_file_location("rsl_rl.storage", _storage_init_path)
    storage_init_module = importlib.util.module_from_spec(spec_storage_init)
    storage_init_module.__package__ = "rsl_rl.storage"
    storage_init_module.__name__ = "rsl_rl.storage"
    storage_init_module.RolloutStorage = LocalRolloutStorage
    sys.modules["rsl_rl.storage"] = storage_init_module
    spec_storage_init.loader.exec_module(storage_init_module)

# 确保使用本地的 PPO 类（通过文件系统路径直接导入）
# PPO 依赖 ActorCritic 和 RolloutStorage
_ppo_path = os.path.join(_rsl_rl_root, "algorithms", "ppo.py")
spec_ppo = importlib.util.spec_from_file_location("rsl_rl.algorithms.ppo", _ppo_path)
ppo_module = importlib.util.module_from_spec(spec_ppo)
ppo_module.__package__ = "rsl_rl.algorithms"
ppo_module.__name__ = "rsl_rl.algorithms.ppo"
# 设置依赖项
ppo_module.ActorCritic = LocalActorCritic
ppo_module.RolloutStorage = LocalRolloutStorage
# 在执行模块之前，先将其注册到 sys.modules
if "rsl_rl.algorithms" not in sys.modules:
    import types
    sys.modules["rsl_rl.algorithms"] = types.ModuleType("rsl_rl.algorithms")
    sys.modules["rsl_rl.algorithms"].__path__ = [os.path.join(_rsl_rl_root, "algorithms")]
sys.modules["rsl_rl.algorithms.ppo"] = ppo_module
spec_ppo.loader.exec_module(ppo_module)
PPO = ppo_module.PPO

try:
    from tensordict import TensorDict as _ObservationTensorDict
except ImportError:
    _ObservationTensorDict = None  # type: ignore[misc, assignment]


def _is_plain_observation_tensor(t) -> bool:
    """与 ``nn.Linear`` 兼容的观测张量。

    新版 ``tensordict.TensorDict`` 在部分 PyTorch 下为 Tensor 子类，``isinstance(td, torch.Tensor)`` 可能为真，
    但绝不能整包传入 Actor；必须按 key 取出叶子张量。
    """
    if t is None:
        return False
    if _ObservationTensorDict is not None and isinstance(t, _ObservationTensorDict):
        return False
    return isinstance(t, torch.Tensor) and t.numel() > 0 and len(t.shape) >= 2


def _maybe_unwrap_obs_container(x):
    """将 ``step`` / ``get_observations`` 顶层的 TensorDict 转为 ``dict``；已是 dict 或普通张量则原样返回。"""
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    if _ObservationTensorDict is not None and isinstance(x, _ObservationTensorDict):
        return {k: x[k] for k in x.keys()}
    # 无 tensordict 时 duck-type：带 batch_size 与 keys 的容器
    if getattr(x, "batch_size", None) is not None and hasattr(x, "keys") and hasattr(x, "__getitem__"):
        try:
            return {k: x[k] for k in x.keys()}
        except Exception:
            pass
    return x


def _ensure_obs_tensor(obs, role_key: str, fallback: torch.Tensor = None):
    """从 dict/TensorDict/张量 中按 role_key 取出观测张量，保证 critic 不用 policy 的张量。
    role_key: "policy" 或 "critic"
    """
    obs = _maybe_unwrap_obs_container(obs)

    def _fallback_tensor(fb):
        if fb is None:
            return None
        u = _maybe_unwrap_obs_container(fb)
        if isinstance(u, dict):
            t = u.get(role_key)
            if t is None:
                t = u.get("policy")
            return t if _is_plain_observation_tensor(t) else None
        return fb if _is_plain_observation_tensor(fb) else None

    fb_tensor = _fallback_tensor(fallback)

    if obs is None:
        return fb_tensor
    if _is_plain_observation_tensor(obs):
        return obs
    if not isinstance(obs, dict):
        return fb_tensor
    if role_key in obs:
        val = obs[role_key]
        if _is_plain_observation_tensor(val):
            return val
    for k in ("observations", "obs", "observation", "state"):
        if k not in obs:
            continue
        sub = _maybe_unwrap_obs_container(obs[k])
        if isinstance(sub, dict) and role_key in sub:
            v = sub[role_key]
            if _is_plain_observation_tensor(v):
                return v
    return fb_tensor


def _get_policy_critic_obs(env, get_observations_returns_dict: bool = None):
    """统一从环境获取 policy 与 critic 观测，保证两者按 key 区分，避免 critic 误用 policy 张量。
    兼容 API：
    (1) dict，且含 "policy" / "critic"；
    (2) tensordict.TensorDict（Isaac Lab 新版 RslRlVecEnvWrapper.get_observations），键同 dict；
    (3) (obs_tensor, extras)，且 extras["observations"] 含 "policy"/"critic"；
    (4) 更长 tuple：首元素为 obs，其中某一元素为含 observations 的 dict。
    返回 (policy_obs_tensor, critic_obs_tensor)，均为 2D tensor。
    """
    out = env.get_observations()
    policy_fallback = None
    critic_fallback = None
    obs_for_keys = None

    if isinstance(out, dict):
        policy_fallback = out.get("policy")
        critic_fallback = out.get("critic", out.get("policy"))
        obs_for_keys = out
    elif getattr(out, "batch_size", None) is not None and hasattr(out, "keys"):
        # Isaac Lab RslRlVecEnvWrapper：get_observations() -> TensorDict(observation_manager.compute())
        try:
            obs_for_keys = {k: out[k] for k in out.keys()}
        except Exception as e:
            raise RuntimeError(
                f"无法将 get_observations() 的 TensorDict 转为观测 dict: {type(out).__name__}: {e}"
            ) from e
        policy_fallback = obs_for_keys.get("policy")
        critic_fallback = obs_for_keys.get("critic", policy_fallback)
    elif isinstance(out, tuple):
        if len(out) == 2:
            obs_tensor, extras = out[0], out[1]
        elif len(out) >= 3:
            obs_tensor = out[0]
            extras = {}
            for item in reversed(out):
                if isinstance(item, dict):
                    extras = item
                    break
        else:
            raise RuntimeError("get_observations() 返回空元组，无法解析 policy/critic 观测。")
        policy_fallback = obs_tensor
        critic_fallback = obs_tensor
        if isinstance(extras, dict) and "observations" in extras:
            obs_for_keys = extras["observations"]
    else:
        raise RuntimeError(
            f"get_observations() 返回类型不受支持: {type(out)}。"
            " 期望 dict、TensorDict（含 batch_size）或 (obs, extras) 元组。"
        )

    policy_obs = _ensure_obs_tensor(obs_for_keys, "policy", policy_fallback)
    critic_obs = _ensure_obs_tensor(obs_for_keys, "critic", critic_fallback)
    if policy_obs is None:
        policy_obs = policy_fallback
    if critic_obs is None:
        critic_obs = critic_fallback
    if policy_obs is None or critic_obs is None:
        raise RuntimeError("critic 张量不应使用 policy 的：无法从环境中解析出 policy 与 critic 观测，请检查 get_observations() 返回格式。")
    return policy_obs, critic_obs


class OnPolicyRunner:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        # 从顶层配置中提取 use_tanh_output（如果存在）
        # 这个参数可能在配置类的顶层，而不是在 policy_cfg 中
        self.use_tanh_output = train_cfg.get("use_tanh_output", False)
        self.device = device
        self.env = env
        # 按 key 区分 policy/critic，避免部署时 critic 误用 policy 张量（兼容 dict 或 (obs, extras) 两种返回格式）
        policy_obs_tensor, critic_obs_tensor = _get_policy_critic_obs(self.env)
        policy_obs_tensor = policy_obs_tensor.to(self.device)
        critic_obs_tensor = critic_obs_tensor.to(self.device)
        num_obs = policy_obs_tensor.shape[1]
        num_critic_obs = critic_obs_tensor.shape[1]

        # Resolve obs_groups from policy cfg, with a safe default
        obs_groups_cfg = self.policy_cfg.get("obs_groups")
        if not isinstance(obs_groups_cfg, dict):
            obs_groups_cfg = {"policy": ["policy"], "critic": ["critic"]}
        # Remove obs_groups from kwargs passed to the network constructor
        if "obs_groups" in self.policy_cfg:
            self.policy_cfg.pop("obs_groups")

        # Construct ActorCritic with correct parameters
        # ActorCritic expects: (num_actor_obs, num_critic_obs, num_actions, **kwargs)
        # 确保使用本地的 ActorCritic 类，而不是通过 eval() 解析（可能解析到 Isaac Lab 的版本）
        class_name = self.policy_cfg.pop("class_name", "ActorCritic")
        
        # 处理可能的完整模块路径（如 "rsl_rl.modules.ActorCritic" 或 "ActorCritic"）
        # 提取类名（最后一个点后的部分）
        if "." in class_name:
            class_name = class_name.split(".")[-1]
        
        # 验证本地 ActorCritic 类是否正确（检查签名）
        import inspect
        local_actor_critic_sig = inspect.signature(LocalActorCritic.__init__)
        local_params = set(local_actor_critic_sig.parameters.keys()) - {"self"}
        if "obs" in local_params or "obs_groups" in local_params:
            raise RuntimeError(
                f"错误：本地的 ActorCritic 类签名不正确！\n"
                f"本地的 ActorCritic 需要 'obs' 和 'obs_groups' 参数，这是不应该的。\n"
                f"当前签名: {local_actor_critic_sig}"
            )
        
        # 使用本地的 ActorCritic 类映射（通过绝对导入路径）
        actor_critic_class_map = {
            "ActorCritic": LocalActorCritic,
            "ActorCriticRecurrent": LocalActorCriticRecurrent,
        }
        if class_name not in actor_critic_class_map:
            print(f"[WARNING] 未知的 ActorCritic 类名: {class_name}，使用默认的 ActorCritic")
            print(f"[WARNING] 支持的类名: {list(actor_critic_class_map.keys())}")
            class_name = "ActorCritic"  # 使用默认值
        
        # 强制使用本地的 ActorCritic 类，忽略配置中的 class_name
        # 这样可以确保始终使用正确的类，即使配置指向了错误的类
        actor_critic_class = LocalActorCritic
        print(f"[INFO] 使用本地的 ActorCritic 类（忽略配置中的 class_name: {class_name}）")
        
        # Extract observation dimensions
        num_actor_obs = policy_obs_tensor.shape[1]
        num_critic_obs = critic_obs_tensor.shape[1]
        
        # 从 policy_cfg 或顶层配置中提取 use_tanh_output（如果存在）
        # 注意：RslRlPpoActorCriticCfg 可能不支持这个参数，所以我们需要手动提取
        # 首先尝试从 policy_cfg 中获取，如果没有则使用顶层配置中的值
        use_tanh_output = self.policy_cfg.pop("use_tanh_output", self.use_tanh_output)
        
        # Create ActorCritic with correct parameter order
        actor_critic: LocalActorCritic | LocalActorCriticRecurrent = actor_critic_class(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=self.env.num_actions,
            use_tanh_output=use_tanh_output,  # 显式传递 use_tanh_output
            **self.policy_cfg  # 传递其他配置参数
        ).to(self.device)
        # 确保使用本地的 PPO 类，而不是通过 eval() 解析（可能解析到 Isaac Lab 的版本）
        class_name = self.alg_cfg.pop("class_name", "PPO")
        # 处理可能的完整模块路径
        if "." in class_name:
            class_name = class_name.split(".")[-1]
        
        # 强制使用本地的 PPO 类，忽略配置中的 class_name
        alg_class = PPO
        print(f"[INFO] 使用本地的 PPO 类（忽略配置中的 class_name: {class_name}）")
        
        # 验证 PPO 类的 init_storage 方法签名
        import inspect
        init_storage_sig = inspect.signature(alg_class.init_storage)
        init_storage_params = set(init_storage_sig.parameters.keys()) - {"self"}
        if "actor_obs_shape" not in init_storage_params:
            raise RuntimeError(
                f"错误：导入的 PPO 类 init_storage 方法签名不正确！\n"
                f"本地的 PPO.init_storage 应该接受 'actor_obs_shape' 参数，但当前签名是: {init_storage_sig}\n"
                f"这可能意味着导入的是 Isaac Lab 的 PPO 版本。"
            )
        
        # Filter out unsupported parameters from alg_cfg
        # Get the signature of PPO.__init__ to know which parameters are supported
        ppo_init_signature = inspect.signature(alg_class.__init__)
        supported_params = set(ppo_init_signature.parameters.keys())
        # Remove 'self' from supported params
        supported_params.discard('self')
        
        # Filter alg_cfg to only include supported parameters
        filtered_alg_cfg = {k: v for k, v in self.alg_cfg.items() if k in supported_params}
        
        # Warn about unsupported parameters
        unsupported_params = set(self.alg_cfg.keys()) - supported_params
        if unsupported_params:
            print(f"[WARNING] PPO不支持以下参数，将被忽略: {unsupported_params}")
        
        self.alg: PPO = alg_class(actor_critic, device=self.device, **filtered_alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
        # init storage and model
        # PPO.init_storage expects: (num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape)
        # Extract observation shapes (remove batch dimension)
        actor_obs_shape = policy_obs_tensor.shape[1:]  # Remove batch dimension, keep feature dimensions
        critic_obs_shape = critic_obs_tensor.shape[1:]  # Remove batch dimension, keep feature dimensions
        action_shape = (self.env.num_actions,)  # Action shape as tuple
        
        self.alg.init_storage(
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.num_steps_per_env,
            actor_obs_shape=actor_obs_shape,
            critic_obs_shape=critic_obs_shape,
            action_shape=action_shape,
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]
        # 课程学习 update：首次解析并缓存，避免每 iter import_module
        self._cached_curriculum_update_fn = None
        self._curriculum_fn_resolve_done = False

    def _resolve_curriculum_update_fn(self, env_obj):
        """Return update_curriculum_progress_per_iteration or None (resolve once)."""
        task_module_path = None
        if hasattr(env_obj, "cfg") and hasattr(env_obj.cfg, "__class__"):
            cfg_module = env_obj.cfg.__class__.__module__
            if "tasks." in cfg_module:
                parts = cfg_module.split(".")
                for i, part in enumerate(parts):
                    if part == "tasks" and i + 1 < len(parts):
                        task_name = parts[i + 1]
                        task_module_path = f"isaacLab.manipulation.tasks.{task_name}.cart_control.mdp.curriculums"
                        break

        fallback_curriculum_modules = [
            "isaacLab.manipulation.tasks.Cart_hands_trajcmd.cart_control.mdp.curriculums",
            "isaacLab.manipulation.tasks.Cart_hands.cart_control.mdp.curriculums",
            "isaacLab.manipulation.tasks.Cart_simplehands.cart_control.mdp.curriculums",
            "isaacLab.manipulation.tasks.Cart_dex3hands.cart_control.mdp.curriculums",
        ]
        if task_module_path is None:
            task_modules = list(fallback_curriculum_modules)
        else:
            task_modules = [task_module_path] + [m for m in fallback_curriculum_modules if m != task_module_path]
            seen = set()
            unique_task_modules = []
            for m in task_modules:
                if m not in seen:
                    seen.add(m)
                    unique_task_modules.append(m)
            task_modules = unique_task_modules

        for task_module in task_modules:
            try:
                curriculums_module = import_module(task_module)
                if hasattr(curriculums_module, "update_curriculum_progress_per_iteration"):
                    return getattr(curriculums_module, "update_curriculum_progress_per_iteration")
            except (ImportError, AttributeError, ModuleNotFoundError):
                continue
        try:
            from isaacLab.manipulation.tasks.Cart_hands_trajcmd.cart_control.mdp.curriculums import (
                update_curriculum_progress_per_iteration,
            )

            return update_curriculum_progress_per_iteration
        except (ImportError, AttributeError, ModuleNotFoundError):
            try:
                from isaacLab.manipulation.tasks.Cart_hands.cart_control.mdp.curriculums import (
                    update_curriculum_progress_per_iteration,
                )

                return update_curriculum_progress_per_iteration
            except (ImportError, AttributeError, ModuleNotFoundError):
                return None

    def _apply_curriculum_update(self, env_obj):
        if not self._curriculum_fn_resolve_done:
            self._cached_curriculum_update_fn = self._resolve_curriculum_update_fn(env_obj)
            self._curriculum_fn_resolve_done = True
        if self._cached_curriculum_update_fn is not None:
            self._cached_curriculum_update_fn(env_obj)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        # 与 __init__ 一致：按 key 取 policy/critic，避免 critic 误用 policy 张量
        policy_obs, critic_obs = _get_policy_critic_obs(self.env)
        obs = {"policy": policy_obs.to(self.device), "critic": critic_obs.to(self.device)}
        self.train_mode()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            if os.environ.get("RSL_RL_TRAIN_HEARTBEAT", "0") == "1":
                print(f"[train heartbeat] iter {it}/{tot_iter - 1} rollout start", flush=True)
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    # Extract policy and critic observations from dict
                    policy_obs_tensor = obs["policy"]
                    critic_obs_tensor = obs["critic"]
                    # 与 get_inference_policy / test 脚本一致：policy 观测经 obs_normalizer 再进 actor。
                    # 若不经过此处，rollout 从未调用 EmpiricalNormalization.forward，保存的 obs_norm_state_dict
                    # 会永远停在初值 (mean=0,std=1)，且训练时 actor 吃的是「未归一化」观测，与部署/录包不一致。
                    if self.empirical_normalization:
                        with torch.no_grad():
                            policy_obs_actor = self.obs_normalizer(policy_obs_tensor)
                    else:
                        policy_obs_actor = policy_obs_tensor
                    # PPO.act() 首参为进入 actor 的观测（已归一化则与 ONNX 输入口径一致）
                    actions = self.alg.act(policy_obs_actor, critic_obs_tensor)
                    
                    # 调试：默认关闭。每 iter 打印会海量写 stdout；SSH/无消费者时管道塞满后 print 阻塞，表现为长时间训练「卡死」无报错。
                    # 需要时: RSL_RL_DEBUG_ACTION_PRINT=1 python ... train.py
                    if (
                        os.environ.get("RSL_RL_DEBUG_ACTION_PRINT", "0") == "1"
                        and i == 0
                        and actions.shape[-1] >= 14
                    ):
                        a0 = actions[0, :14].detach().cpu().numpy()
                        print(f"[train] iter={it} step0 env0 action(14) [raw policy]: {a0.tolist()}")
                    
                    obs_tensor, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    # 新版 RslRlVecEnvWrapper：obs_tensor 可能为 TensorDict；须解包后再喂 Actor（不能整包当 Tensor）
                    obs_step = _maybe_unwrap_obs_container(obs_tensor)
                    obs_dict = infos.get("observations", None)
                    if obs_dict is not None:
                        obs_dict = _maybe_unwrap_obs_container(obs_dict)
                    elif isinstance(obs_step, dict):
                        obs_dict = obs_step
                    else:
                        obs_dict = {"policy": obs_tensor, "critic": obs_tensor}
                    policy_obs = _ensure_obs_tensor(obs_dict, "policy", obs_tensor)
                    critic_obs = _ensure_obs_tensor(obs_dict, "critic", obs_tensor)
                    obs = {"policy": policy_obs.to(self.device), "critic": critic_obs.to(self.device)}
                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                    # process the step - PPO.process_env_step expects (rewards, dones, infos)
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        # note: we changed logging to use "log" instead of "episode" to avoid confusion with
                        # different types of logging data (rewards, curriculum, etc.)
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                # compute_returns expects last_critic_obs as a tensor
                last_critic_obs = obs["critic"]
                self.alg.compute_returns(last_critic_obs)

            # PPO.update() returns (mean_value_loss, mean_surrogate_loss, mean_entropy) tuple
            mean_value_loss, mean_surrogate_loss, mean_entropy = self.alg.update()
            # Convert to dict format expected by log() method
            loss_dict = {
                "value_function": mean_value_loss,
                "surrogate": mean_surrogate_loss,
                "entropy": mean_entropy,
            }
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            # 将当前 learning iteration 和 mean_episode_length 存储到环境对象，供课程学习使用
            # 计算最近完成的episode的平均长度（与 Train/mean_episode_length 一致）
            if len(lenbuffer) > 0:
                mean_ep_len = statistics.mean(lenbuffer)
            else:
                mean_ep_len = 0.0
            if hasattr(self.env, 'unwrapped'):
                self.env.unwrapped.current_learning_iteration = it
                self.env.unwrapped.mean_episode_length = mean_ep_len
                self.env.unwrapped.num_steps_per_env = self.num_steps_per_env  # 存储rollout长度（用于EMA alpha计算）
            else:
                self.env.current_learning_iteration = it
                self.env.mean_episode_length = mean_ep_len
                self.env.num_steps_per_env = self.num_steps_per_env  # 存储rollout长度（用于EMA alpha计算）
            
            # 在每个 iteration 结束时更新课程 progress（首次解析模块并缓存，避免每 iter import）
            env_obj = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
            self._apply_curriculum_update(env_obj)

            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()
            if it == start_iter:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # losses from latest PPO.update()
        loss_dict = locs.get("loss_dict", {})
        value_loss_v = loss_dict.get("value_function")
        surrogate_loss_v = loss_dict.get("surrogate")
        entropy_v = loss_dict.get("entropy")

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        # Compatible with latest PPO API where network is stored as `policy`
        ac_module = getattr(self.alg, "policy", None) or getattr(self.alg, "actor_critic", None)
        mean_std = ac_module.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        if value_loss_v is not None:
            self.writer.add_scalar("Loss/value_function", value_loss_v, locs["it"])
        if surrogate_loss_v is not None:
            self.writer.add_scalar("Loss/surrogate", surrogate_loss_v, locs["it"])
        if entropy_v is not None:
            self.writer.add_scalar("Loss/entropy", entropy_v, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        header_str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{header_str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {value_loss_v if value_loss_v is not None else float('nan'):.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {surrogate_loss_v if surrogate_loss_v is not None else float('nan'):.4f}\n"""
                f"""{'Entropy:':>{pad}} {entropy_v if entropy_v is not None else float('nan'):.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{header_str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {value_loss_v if value_loss_v is not None else float('nan'):.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {surrogate_loss_v if surrogate_loss_v is not None else float('nan'):.4f}\n"""
                f"""{'Entropy:':>{pad}} {entropy_v if entropy_v is not None else float('nan'):.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string, flush=True)
        if self.writer is not None and hasattr(self.writer, "flush"):
            self.writer.flush()

    def save(self, path, infos=None):
        # Resolve actor-critic module (compat between newer/older PPO APIs)
        ac_module = getattr(self.alg, "policy", None) or getattr(self.alg, "actor_critic", None)
        saved_dict = {
            "model_state_dict": ac_module.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, weights_only=False)  # weights_only=False for compatibility with older checkpoints
        ac_module = getattr(self.alg, "policy", None) or getattr(self.alg, "actor_critic", None)
        ac_module.load_state_dict(loaded_dict["model_state_dict"]) 
        if self.empirical_normalization:
            # 使用 strict=False 来忽略不匹配的键（如 "count"），因为推理时不需要这些统计信息
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"], strict=False)
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"], strict=False)
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        ac_module = getattr(self.alg, "policy", None) or getattr(self.alg, "actor_critic", None)
        if device is not None:
            ac_module.to(device)
        policy = ac_module.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            # 归一化只应用于policy观察（张量），而不是整个字典
            # act_inference期望接收字典格式的观察
            def normalized_policy(obs_dict):
                if isinstance(obs_dict, dict):
                    # 归一化policy观察
                    normalized_policy_obs = self.obs_normalizer(obs_dict["policy"])
                    # 构建新的字典，保持critic观察不变（如果需要归一化critic，应该使用critic_obs_normalizer）
                    normalized_obs = {"policy": normalized_policy_obs}
                    if "critic" in obs_dict:
                        normalized_obs["critic"] = obs_dict["critic"]
                    return ac_module.act_inference(normalized_obs)
                else:
                    # 如果传入的是张量（向后兼容）
                    return ac_module.act_inference(self.obs_normalizer(obs_dict))
            policy = normalized_policy
        return policy

    def train_mode(self):
        # Align with latest PPO API which stores network in `policy`
        (getattr(self.alg, "policy", None) or self.alg.actor_critic).train()
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def eval_mode(self):
        (getattr(self.alg, "policy", None) or self.alg.actor_critic).eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)
