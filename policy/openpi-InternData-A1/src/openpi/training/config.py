"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import os
import pathlib
from typing import Any, Literal, Protocol, TypeAlias, Dict

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.real_lift2_policy as real_lift2_policy
import openpi.policies.real_a2d_policy as real_a2d_policy
import openpi.policies.sim_split_aloha_policy as sim_split_aloha_policy
import openpi.policies.sim2real_split_aloha_policy as sim2real_split_aloha_policy
import openpi.policies.sim_franka_policy as sim_franka_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
import openpi.shared.normalize as normalize
from natsort import natsorted
from glob import glob
ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter

from pdb import set_trace

@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None


@dataclasses.dataclass(frozen=True)
class MultiDataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_dir: str | None = None
    # Directory within the assets directory containing the data assets.
    task_id: str | None = None
    subtask_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None
    # weight
    weight: float = 1.0
    use_gripper_aug: bool = False
    gripper_aug_config: dict[str, Any] | None = None
    stats_dir: str = tyro.MISSING
    data_ratio: float = 1.0
    # asset_id for saving norm stats
    asset_id: str = tyro.MISSING
    downsample_ratio: float = 1.0


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            # logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
            raise FileNotFoundError(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None

@dataclasses.dataclass(frozen=True)
class MultiDataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_dir: str = tyro.MISSING
    # Determines how the assets will be loaded.
    task_id: str | None = None
    # Determines how the assets will be loaded.
    # Base config that will be updated by the factory.
    asset_id: str = tyro.MISSING
    base_config: tyro.conf.Suppress[MultiDataConfig | None] = None
    weight: float = 1.0
    use_gripper_aug: bool = False
    gripper_aug_config: dict[str, Any] | None = None
    stats_dir: str = tyro.MISSING
    data_ratio: float = 1.0
    fixed_stats_dir: str | None = None
    downsample_ratio: float = 1.0
    robot_name: str = tyro.MISSING
    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> MultiDataConfig:
        """Create a data config."""

    def create_base_config(self, repo_dir, task_id, subtask_id, stats_dir, model_config: _model.BaseModelConfig) -> MultiDataConfig:
        robot_name = repo_dir.split('/')[-1]
        assert robot_name == self.robot_name, f"robot_name mismatch: {robot_name} != {self.robot_name}"
        task_category = repo_dir.split('/')[-2]
        return dataclasses.replace(
            self.base_config or MultiDataConfig(),
            repo_dir=repo_dir,
            task_id=task_id,
            subtask_id=subtask_id,
            norm_stats=self._load_norm_stats(epath.Path(stats_dir), task_category, robot_name, task_id, subtask_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
            weight=self.weight,
            use_gripper_aug=self.use_gripper_aug,
            gripper_aug_config=self.gripper_aug_config,
            stats_dir=self.stats_dir,
            data_ratio=self.data_ratio,
            asset_id=self.asset_id,
            downsample_ratio=self.downsample_ratio,
        )

    def _load_norm_stats(self, stats_dir: epath.Path, task_category, robot_name, task_id, subtask_id) -> dict[str, _transforms.NormStats] | None:
        try:
            if self.fixed_stats_dir is not None:
                data_assets_dir = self.fixed_stats_dir
                logging.info("Loaded from fixed stats dir")
            else:
                data_assets_dir = str(stats_dir / task_category / robot_name / task_id / subtask_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, check stats_dir config.")
            # raise FileNotFoundError(f"Norm stats not found in {data_assets_dir}, check stats_dir config.")

@dataclasses.dataclass(frozen=True)
class MultiLeRobotRealArxLift2DataConfig(MultiDataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    default_prompt: str | None = None
    # If provided, will be injected into the input data if the "prompt" key is not present.
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = False
    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": 
                        {
                            "cam_high": "images.rgb.head", 
                            "cam_left_wrist": "images.rgb.hand_left", 
                            "cam_right_wrist": "images.rgb.hand_right"
                        },
                        "state_dict": {
                            "left_joint": "states.left_joint.position", 
                            "right_joint": "states.right_joint.position", 
                            "left_gripper": "states.left_gripper.position", 
                            "right_gripper": "states.right_gripper.position"
                        },
                        "action_dict": {
                            "left_joint": "actions.left_joint.position", 
                            "right_joint": "actions.right_joint.position", 
                            "left_gripper": "actions.left_gripper.position", 
                            "right_gripper": "actions.right_gripper.position"
                        },
                        "prompt": "task"
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("actions.left_joint.position","actions.right_joint.position","actions.left_gripper.position","actions.right_gripper.position")

    @override
    def create(self, model_config: _model.BaseModelConfig, global_norm_stats: Dict[str, normalize.NormStats] = None) -> list[MultiDataConfig]:
        data_configs = []
        data_transforms = _transforms.Group(
            inputs=[real_lift2_policy.RealLift2Inputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[real_lift2_policy.RealLift2Outputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        if isinstance(self.repo_dir, str) and "*" not in self.repo_dir:
            local_paths = [self.repo_dir]
        elif isinstance(self.repo_dir, str) and "*" in self.repo_dir:
            all_local_paths = natsorted(glob(self.repo_dir))
            if self.task_id is not None:
                local_paths = [
                    p for p in all_local_paths
                    if any(str(task_id) in p for task_id in self.task_id)
                ]
            else:
                local_paths = all_local_paths

        for local_path in local_paths:
            if not check_lerobot_repo(local_path):
                continue
            robot_names = self.robot_name
            parts = local_path.split("/")

            robot_idx = next((i for i, p in enumerate(parts) if p == robot_names), None)
            if robot_idx is None:
                raise ValueError(
                    f"Cannot find robot name in path. Expected {robot_names}, "
                    f"but got path: {local_path}"
                )

            repo_dir = "/".join(parts[:robot_idx + 1])

            if robot_idx + 1 >= len(parts):
                raise ValueError(
                    f"Path ends at robot name '{parts[robot_idx]}', cannot determine task_name: {local_path}"
                )

            task_name = parts[robot_idx + 1]
            subtask_name = parts[robot_idx + 2] if robot_idx + 2 < len(parts) else ""
            if global_norm_stats is None:
                data_configs.append(dataclasses.replace(
                    self.create_base_config(repo_dir, task_name, subtask_name, self.stats_dir, model_config),
                    repack_transforms=self.repack_transforms,
                    data_transforms=data_transforms,
                    model_transforms=model_transforms,
                    action_sequence_keys=self.action_sequence_keys,
                    weight=self.weight,
                    use_gripper_aug=self.use_gripper_aug,
                    gripper_aug_config=self.gripper_aug_config,
                    stats_dir=self.stats_dir,
                ))
            else:
                data_configs.append(dataclasses.replace(
                    self.create_base_config(repo_dir, task_name, subtask_name, self.stats_dir, model_config),
                    repack_transforms=self.repack_transforms,
                    data_transforms=data_transforms,
                    model_transforms=model_transforms,
                    action_sequence_keys=self.action_sequence_keys,
                    weight=self.weight,
                    use_gripper_aug=self.use_gripper_aug,
                    gripper_aug_config=self.gripper_aug_config,
                    stats_dir=self.stats_dir,
                    norm_stats=global_norm_stats,
                ))
            

        return data_configs


@dataclasses.dataclass(frozen=True)
class MultiLeRobotReala2dDataConfig(MultiDataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    default_prompt: str | None = None
    # If provided, will be injected into the input data if the "prompt" key is not present.
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = False
    weight: float = 1.0
    use_gripper_aug: bool = False
    gripper_aug_config: dict[str, Any] | None = None
    stats_dir: str = tyro.MISSING
    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": 
                        {
                            "cam_high": "observation.images.head", 
                            "cam_left_wrist": "observation.images.hand_left", 
                            "cam_right_wrist": "observation.images.hand_right"
                        },
                        "state_dict": {
                            "joint": "observation.states.joint.position", 
                            "gripper": "observation.states.effector.position", 
                        },
                        "action_dict": {
                            "joint": "actions.joint.position", 
                            "gripper": "actions.effector.position", 
                        },
                        "prompt": "task"
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("actions.joint.position","actions.effector.position")

    @override
    def create(self, model_config: _model.BaseModelConfig, global_norm_stats: Dict[str, normalize.NormStats] = None) -> list[MultiDataConfig]:
        data_configs = []
        data_transforms = _transforms.Group(
            inputs=[real_a2d_policy.Reala2dInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[real_a2d_policy.Reala2dOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(7, -1, 7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        if isinstance(self.repo_dir, str) and "*" not in self.repo_dir:
            local_paths = [self.repo_dir]
        elif isinstance(self.repo_dir, str) and "*" in self.repo_dir:
            all_local_paths = natsorted(glob(self.repo_dir))
            if self.task_id is not None:
                local_paths = [
                    p for p in all_local_paths
                    if any(str(task_id) in p for task_id in self.task_id)
                ]
            else:
                local_paths = all_local_paths

        for local_path in local_paths:
            if not check_lerobot_repo(local_path):
                continue
            robot_names = self.robot_name
            parts = local_path.split("/")

            robot_idx = next((i for i, p in enumerate(parts) if p == robot_names), None)
            if robot_idx is None:
                raise ValueError(
                    f"Cannot find robot name in path. Expected {robot_names}, "
                    f"but got path: {local_path}"
                )

            repo_dir = "/".join(parts[:robot_idx + 1])

            if robot_idx + 1 >= len(parts):
                raise ValueError(
                    f"Path ends at robot name '{parts[robot_idx]}', cannot determine task_name: {local_path}"
                )

            task_name = parts[robot_idx + 1]
            subtask_name = parts[robot_idx + 2] if robot_idx + 2 < len(parts) else ""
            if global_norm_stats is None:
                data_configs.append(dataclasses.replace(
                    self.create_base_config(repo_dir, task_name, subtask_name, self.stats_dir, model_config),
                    repack_transforms=self.repack_transforms,
                    data_transforms=data_transforms,
                    model_transforms=model_transforms,
                    action_sequence_keys=self.action_sequence_keys,
                    weight=self.weight,
                    use_gripper_aug=self.use_gripper_aug,
                    gripper_aug_config=self.gripper_aug_config,
                    stats_dir=self.stats_dir,
                ))
            else:
                data_configs.append(dataclasses.replace(
                    self.create_base_config(repo_dir, task_name, subtask_name, self.stats_dir, model_config),
                    repack_transforms=self.repack_transforms,
                    data_transforms=data_transforms,
                    model_transforms=model_transforms,
                    action_sequence_keys=self.action_sequence_keys,
                    weight=self.weight,
                    use_gripper_aug=self.use_gripper_aug,
                    gripper_aug_config=self.gripper_aug_config,
                    stats_dir=self.stats_dir,
                    norm_stats=global_norm_stats,
                ))
            

        return data_configs

@dataclasses.dataclass(frozen=True)
class MultiSimGenieDataConfig(MultiDataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    default_prompt: str | None = None
    # If provided, will be injected into the input data if the "prompt" key is not present.
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = False
    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": 
                        {
                            "cam_high": "images.rgb.head", 
                            "cam_left_wrist": "images.rgb.hand_left", 
                            "cam_right_wrist": "images.rgb.hand_right"
                        },
                        "state_dict": {
                            "left_joint": "states.left_joint.position", 
                            "right_joint": "states.right_joint.position", 
                            "left_gripper": "states.left_gripper.position", 
                            "right_gripper": "states.right_gripper.position"
                        },
                        "action_dict": {
                            "left_joint": "actions.left_joint.position", 
                            "right_joint": "actions.right_joint.position", 
                            "left_gripper": "actions.left_gripper.position", 
                            "right_gripper": "actions.right_gripper.position",
                            "left_gripper_openness": "master_actions.left_gripper.openness",
                            "right_gripper_openness": "master_actions.right_gripper.openness"
                        },
                        "prompt": "task"
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("actions.left_joint.position","actions.right_joint.position","actions.left_gripper.position","actions.right_gripper.position","master_actions.left_gripper.openness","master_actions.right_gripper.openness")

    @override
    def create(self, model_config: _model.BaseModelConfig,global_norm_stats: Dict[str, normalize.NormStats] = None) -> list[MultiDataConfig]:
        data_configs = []
        data_transforms = _transforms.Group(
            inputs=[sim_split_aloha_policy.SimSplitAlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[sim_split_aloha_policy.SimSplitAlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(7, -1, 7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        if isinstance(self.repo_dir, str) and "*" not in self.repo_dir:
            local_paths = [self.repo_dir]
        elif isinstance(self.repo_dir, str) and "*" in self.repo_dir:
            all_local_paths = natsorted(glob(self.repo_dir))
            if self.task_id is not None:
                local_paths = [
                    p for p in all_local_paths
                    if any(str(task_id) in p for task_id in self.task_id)
                ]
            else:
                local_paths = all_local_paths
        for local_path in local_paths:
            if not check_lerobot_repo(local_path):
                continue
            robot_names = self.robot_name
            parts = local_path.split("/")

            robot_idx = next((i for i, p in enumerate(parts) if p == robot_names), None)
            if robot_idx is None:
                raise ValueError(
                    f"Cannot find robot name in path. Expected {robot_names}, "
                    f"but got path: {local_path}"
                )

            repo_dir = "/".join(parts[:robot_idx + 1])

            if robot_idx + 1 >= len(parts):
                raise ValueError(
                    f"Path ends at robot name '{parts[robot_idx]}', cannot determine task_name: {local_path}"
                )

            task_name = parts[robot_idx + 1]
            subtask_name = parts[robot_idx + 2] if robot_idx + 2 < len(parts) else ""
            if global_norm_stats is None:
                data_configs.append(dataclasses.replace(
                    self.create_base_config(repo_dir, task_name, subtask_name, self.stats_dir, model_config),
                    repack_transforms=self.repack_transforms,
                    data_transforms=data_transforms,
                    model_transforms=model_transforms,
                    action_sequence_keys=self.action_sequence_keys,
                    weight=self.weight,
                    use_gripper_aug=self.use_gripper_aug,
                    gripper_aug_config=self.gripper_aug_config,
                    stats_dir=self.stats_dir,
                ))
            else:
                data_configs.append(dataclasses.replace(
                    self.create_base_config(repo_dir, task_name, subtask_name, self.stats_dir, model_config),
                    repack_transforms=self.repack_transforms,
                    data_transforms=data_transforms,
                    model_transforms=model_transforms,
                    action_sequence_keys=self.action_sequence_keys,
                    weight=self.weight,
                    use_gripper_aug=self.use_gripper_aug,
                    gripper_aug_config=self.gripper_aug_config,
                    stats_dir=self.stats_dir,
                    norm_stats=global_norm_stats,
                ))
            
        return data_configs

@dataclasses.dataclass(frozen=True)
class MultiSim2RealGenieDataConfig(MultiDataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    default_prompt: str | None = None
    # If provided, will be injected into the input data if the "prompt" key is not present.
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = False
    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": 
                        {
                            "cam_high": "images.rgb.head", 
                            "cam_left_wrist": "images.rgb.hand_left", 
                            "cam_right_wrist": "images.rgb.hand_right"
                        },
                        "state_dict": {
                            "left_joint": "states.left_joint.position", 
                            "right_joint": "states.right_joint.position", 
                            "left_gripper": "states.left_gripper.position", 
                            "right_gripper": "states.right_gripper.position"
                        },
                        "action_dict": {
                            "left_joint": "actions.left_joint.position", 
                            "right_joint": "actions.right_joint.position", 
                            "left_gripper": "actions.left_gripper.position", 
                            "right_gripper": "actions.right_gripper.position",
                            "left_gripper_openness": "master_actions.left_gripper.openness",
                            "right_gripper_openness": "master_actions.right_gripper.openness"
                        },
                        "prompt": "task"
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("actions.left_joint.position","actions.right_joint.position","actions.left_gripper.position","actions.right_gripper.position","master_actions.left_gripper.openness","master_actions.right_gripper.openness")

    @override
    def create(self, model_config: _model.BaseModelConfig,global_norm_stats: Dict[str, normalize.NormStats] = None) -> list[MultiDataConfig]:
        data_configs = []
        data_transforms = _transforms.Group(
            inputs=[sim2real_split_aloha_policy.Sim2RealSplitAlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[sim2real_split_aloha_policy.Sim2RealSplitAlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(7, -1, 7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        if isinstance(self.repo_dir, str) and "*" not in self.repo_dir:
            local_paths = [self.repo_dir]
        elif isinstance(self.repo_dir, str) and "*" in self.repo_dir:
            all_local_paths = natsorted(glob(self.repo_dir))
            if self.task_id is not None:
                local_paths = [
                    p for p in all_local_paths
                    if any(str(task_id) in p for task_id in self.task_id)
                ]
            else:
                local_paths = all_local_paths

        for local_path in local_paths:
            if not check_lerobot_repo(local_path):
                continue
            robot_names = self.robot_name
            parts = local_path.split("/")

            robot_idx = next((i for i, p in enumerate(parts) if p == robot_names), None)
            if robot_idx is None:
                raise ValueError(
                    f"Cannot find robot name in path. Expected {robot_names}, "
                    f"but got path: {local_path}"
                )

            repo_dir = "/".join(parts[:robot_idx + 1])

            if robot_idx + 1 >= len(parts):
                raise ValueError(
                    f"Path ends at robot name '{parts[robot_idx]}', cannot determine task_name: {local_path}"
                )

            task_name = parts[robot_idx + 1]
            subtask_name = parts[robot_idx + 2] if robot_idx + 2 < len(parts) else ""
            if global_norm_stats is None:
                data_configs.append(dataclasses.replace(
                    self.create_base_config(repo_dir, task_name, subtask_name, self.stats_dir, model_config),
                    repack_transforms=self.repack_transforms,
                    data_transforms=data_transforms,
                    model_transforms=model_transforms,
                    action_sequence_keys=self.action_sequence_keys,
                    weight=self.weight,
                    use_gripper_aug=self.use_gripper_aug,
                    gripper_aug_config=self.gripper_aug_config,
                    stats_dir=self.stats_dir,
                ))
            else:
                data_configs.append(dataclasses.replace(
                    self.create_base_config(repo_dir, task_name, subtask_name, self.stats_dir, model_config),
                    repack_transforms=self.repack_transforms,
                    data_transforms=data_transforms,
                    model_transforms=model_transforms,
                    action_sequence_keys=self.action_sequence_keys,
                    weight=self.weight,
                    use_gripper_aug=self.use_gripper_aug,
                    gripper_aug_config=self.gripper_aug_config,
                    stats_dir=self.stats_dir,
                    norm_stats=global_norm_stats,
                ))    

        return data_configs


@dataclasses.dataclass(frozen=True)
class MultiSimSplitAlohaDataConfig(MultiDataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    default_prompt: str | None = None
    # If provided, will be injected into the input data if the "prompt" key is not present.
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = False

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": 
                        {
                            "cam_high": "images.rgb.head", 
                            "cam_left_wrist": "images.rgb.hand_left", 
                            "cam_right_wrist": "images.rgb.hand_right"
                        },
                        "state_dict": {
                            "left_joint": "states.left_joint.position", 
                            "right_joint": "states.right_joint.position", 
                            "left_gripper": "states.left_gripper.position", 
                            "right_gripper": "states.right_gripper.position"
                        },
                        "action_dict": {
                            "left_joint": "actions.left_joint.position", 
                            "right_joint": "actions.right_joint.position", 
                            "left_gripper": "actions.left_gripper.position", 
                            "right_gripper": "actions.right_gripper.position",
                            "left_gripper_openness": "master_actions.left_gripper.openness",
                            "right_gripper_openness": "master_actions.right_gripper.openness"
                        },
                        "prompt": "task"
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("actions.left_joint.position","actions.right_joint.position","actions.left_gripper.position","actions.right_gripper.position","master_actions.left_gripper.openness","master_actions.right_gripper.openness")

    @override
    def create(self, model_config: _model.BaseModelConfig,global_norm_stats: Dict[str, normalize.NormStats] = None) -> list[MultiDataConfig]:
        data_configs = []
        data_transforms = _transforms.Group(
            inputs=[sim_split_aloha_policy.SimSplitAlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[sim_split_aloha_policy.SimSplitAlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        if isinstance(self.repo_dir, str) and "*" not in self.repo_dir:
            local_paths = [self.repo_dir]
        elif isinstance(self.repo_dir, str) and "*" in self.repo_dir:
            all_local_paths = natsorted(glob(self.repo_dir))
            if self.task_id is not None:
                local_paths = [
                    p for p in all_local_paths
                    if any(str(task_id) in p for task_id in self.task_id)
                ]
            else:
                local_paths = all_local_paths

        for local_path in local_paths:
            if not check_lerobot_repo(local_path):
                continue
            robot_names = self.robot_name
            parts = local_path.split("/")

            robot_idx = next((i for i, p in enumerate(parts) if p == robot_names), None)
            if robot_idx is None:
                raise ValueError(
                    f"Cannot find robot name in path. Expected {robot_names}, "
                    f"but got path: {local_path}"
                )

            repo_dir = "/".join(parts[:robot_idx + 1])

            if robot_idx + 1 >= len(parts):
                raise ValueError(
                    f"Path ends at robot name '{parts[robot_idx]}', cannot determine task_name: {local_path}"
                )

            task_name = parts[robot_idx + 1]
            subtask_name = parts[robot_idx + 2] if robot_idx + 2 < len(parts) else ""
            if global_norm_stats is None:
                data_configs.append(dataclasses.replace(
                    self.create_base_config(repo_dir, task_name, subtask_name, self.stats_dir, model_config),
                    repack_transforms=self.repack_transforms,
                    data_transforms=data_transforms,
                    model_transforms=model_transforms,
                    action_sequence_keys=self.action_sequence_keys,
                    weight=self.weight,
                    use_gripper_aug=self.use_gripper_aug,
                    gripper_aug_config=self.gripper_aug_config,
                    stats_dir=self.stats_dir,
                ))
            else:
                data_configs.append(dataclasses.replace(
                    self.create_base_config(repo_dir, task_name, subtask_name, self.stats_dir, model_config),
                    repack_transforms=self.repack_transforms,
                    data_transforms=data_transforms,
                    model_transforms=model_transforms,
                    action_sequence_keys=self.action_sequence_keys,
                    weight=self.weight,
                    use_gripper_aug=self.use_gripper_aug,
                    gripper_aug_config=self.gripper_aug_config,
                    stats_dir=self.stats_dir,
                    norm_stats=global_norm_stats,
                ))
        return data_configs


@dataclasses.dataclass(frozen=True)
class MultiSim2RealSplitAlohaDataConfig(MultiDataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    default_prompt: str | None = None
    # If provided, will be injected into the input data if the "prompt" key is not present.
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = False
    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": 
                        {
                            "cam_high": "images.rgb.head", 
                            "cam_left_wrist": "images.rgb.hand_left", 
                            "cam_right_wrist": "images.rgb.hand_right"
                        },
                        "state_dict": {
                            "left_joint": "states.left_joint.position", 
                            "right_joint": "states.right_joint.position", 
                            "left_gripper": "states.left_gripper.position", 
                            "right_gripper": "states.right_gripper.position"
                        },
                        "action_dict": {
                            "left_joint": "actions.left_joint.position", 
                            "right_joint": "actions.right_joint.position", 
                            "left_gripper": "actions.left_gripper.position", 
                            "right_gripper": "actions.right_gripper.position",
                            "left_gripper_openness": "master_actions.left_gripper.openness",
                            "right_gripper_openness": "master_actions.right_gripper.openness"
                        },
                        "prompt": "task"
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("actions.left_joint.position","actions.right_joint.position","actions.left_gripper.position","actions.right_gripper.position","master_actions.left_gripper.openness","master_actions.right_gripper.openness")

    @override
    def create(self, model_config: _model.BaseModelConfig,global_norm_stats: Dict[str, normalize.NormStats] = None) -> list[MultiDataConfig]:
        data_configs = []
        data_transforms = _transforms.Group(
            inputs=[sim2real_split_aloha_policy.Sim2RealSplitAlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[sim2real_split_aloha_policy.Sim2RealSplitAlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        if isinstance(self.repo_dir, str) and "*" not in self.repo_dir:
            local_paths = [self.repo_dir]
        elif isinstance(self.repo_dir, str) and "*" in self.repo_dir:
            all_local_paths = natsorted(glob(self.repo_dir))
            if self.task_id is not None:
                local_paths = [
                    p for p in all_local_paths
                    if any(str(task_id) in p for task_id in self.task_id)
                ]
            else:
                local_paths = all_local_paths

        for local_path in local_paths:
            if not check_lerobot_repo(local_path):
                continue
            robot_names = self.robot_name
            parts = local_path.split("/")

            robot_idx = next((i for i, p in enumerate(parts) if p == robot_names), None)
            if robot_idx is None:
                raise ValueError(
                    f"Cannot find robot name in path. Expected {robot_names}, "
                    f"but got path: {local_path}"
                )

            repo_dir = "/".join(parts[:robot_idx + 1])

            if robot_idx + 1 >= len(parts):
                raise ValueError(
                    f"Path ends at robot name '{parts[robot_idx]}', cannot determine task_name: {local_path}"
                )

            task_name = parts[robot_idx + 1]
            subtask_name = parts[robot_idx + 2] if robot_idx + 2 < len(parts) else ""
            if global_norm_stats is None:
                data_configs.append(dataclasses.replace(
                    self.create_base_config(repo_dir, task_name, subtask_name, self.stats_dir, model_config),
                    repack_transforms=self.repack_transforms,
                    data_transforms=data_transforms,
                    model_transforms=model_transforms,
                    action_sequence_keys=self.action_sequence_keys,
                    weight=self.weight,
                    use_gripper_aug=self.use_gripper_aug,
                    gripper_aug_config=self.gripper_aug_config,
                    stats_dir=self.stats_dir,
                ))
            else:
                data_configs.append(dataclasses.replace(
                    self.create_base_config(repo_dir, task_name, subtask_name, self.stats_dir, model_config),
                    repack_transforms=self.repack_transforms,
                    data_transforms=data_transforms,
                    model_transforms=model_transforms,
                    action_sequence_keys=self.action_sequence_keys,
                    weight=self.weight,
                    use_gripper_aug=self.use_gripper_aug,
                    gripper_aug_config=self.gripper_aug_config,
                    stats_dir=self.stats_dir,
                    norm_stats=global_norm_stats,
                ))
        return data_configs



@dataclasses.dataclass(frozen=True)
class MultiSimFrankaDataConfig(MultiDataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    default_prompt: str | None = None
    # If provided, will be injected into the input data if the "prompt" key is not present.
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = False

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": 
                        {
                            "cam_high": "images.rgb.head", 
                            "cam_left_wrist": "images.rgb.hand", 
                        },
                        "state_dict": {
                            "joint_position": "states.joint.position", 
                            "gripper_pose": "states.gripper.pose", 
                            "gripper_position": "states.gripper.position", 
                        },
                        "action_dict": {
                            "gripper_pose": "actions.gripper.pose", 
                            "gripper_position": "actions.gripper.position", 
                            "gripper_openness": "actions.gripper.openness", 
                        },
                        "prompt": "task"
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("actions.gripper.pose","actions.gripper.position","actions.gripper.openness")

    @override
    def create(self, model_config: _model.BaseModelConfig,global_norm_stats: Dict[str, normalize.NormStats] = None) -> list[MultiDataConfig]:
        data_configs = []
        data_transforms = _transforms.Group(
            inputs=[sim_franka_policy.SimFrankaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[sim_franka_policy.SimFrankaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActionsPose(delta_action_mask)],
                outputs=[_transforms.AbsoluteActionsPose(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        if isinstance(self.repo_dir, str) and "*" not in self.repo_dir:
            local_paths = [self.repo_dir]
        elif isinstance(self.repo_dir, str) and "*" in self.repo_dir:
            all_local_paths = natsorted(glob(self.repo_dir))
            if self.task_id is not None:
                local_paths = [
                    p for p in all_local_paths
                    if any(str(task_id) in p for task_id in self.task_id)
                ]
            else:
                local_paths = all_local_paths

        for local_path in local_paths:
            if not check_lerobot_repo(local_path):
                continue
            robot_names = self.robot_name
            parts = local_path.split("/")

            robot_idx = next((i for i, p in enumerate(parts) if p == robot_names), None)
            if robot_idx is None:
                raise ValueError(
                    f"Cannot find robot name in path. Expected {robot_names}, "
                    f"but got path: {local_path}"
                )

            repo_dir = "/".join(parts[:robot_idx + 1])

            if robot_idx + 1 >= len(parts):
                raise ValueError(
                    f"Path ends at robot name '{parts[robot_idx]}', cannot determine task_name: {local_path}"
                )

            task_name = parts[robot_idx + 1]
            subtask_name = parts[robot_idx + 2] if robot_idx + 2 < len(parts) else ""
            if global_norm_stats is None:
                data_configs.append(dataclasses.replace(
                    self.create_base_config(repo_dir, task_name, subtask_name, self.stats_dir, model_config),
                    repack_transforms=self.repack_transforms,
                    data_transforms=data_transforms,
                    model_transforms=model_transforms,
                    action_sequence_keys=self.action_sequence_keys,
                    weight=self.weight,
                    use_gripper_aug=self.use_gripper_aug,
                    gripper_aug_config=self.gripper_aug_config,
                    stats_dir=self.stats_dir,
                ))
            else:
                data_configs.append(dataclasses.replace(
                    self.create_base_config(repo_dir, task_name, subtask_name, self.stats_dir, model_config),
                    repack_transforms=self.repack_transforms,
                    data_transforms=data_transforms,
                    model_transforms=model_transforms,
                    action_sequence_keys=self.action_sequence_keys,
                    weight=self.weight,
                    use_gripper_aug=self.use_gripper_aug,
                    gripper_aug_config=self.gripper_aug_config,
                    stats_dir=self.stats_dir,
                    norm_stats=global_norm_stats,
                ))
        return data_configs



@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotArxAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    default_prompt: str | None = None
    # If provided, will be injected into the input data if the "prompt" key is not present.
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = False

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": 
                        {
                            "cam_high": "images.rgb.head", 
                            "cam_left_wrist": "images.rgb.hand_left", 
                            "cam_right_wrist": "images.rgb.hand_right"
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "task"
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )



@dataclasses.dataclass(frozen=True)
class StatsDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = False
    default_prompt: str | None = None
    # If provided, will be injected into the input data if the "prompt" key is not present.
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = False

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": 
                        {
                            "cam_high": "images.rgb.head", 
                            "cam_left_wrist": "images.rgb.hand_left", 
                            "cam_right_wrist": "images.rgb.hand_right"
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "task"
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.
    # Path to the filter dictionary file.
    filter_dict_path: str | None = "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            filter_dict_path=self.filter_dict_path,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: list[DataConfigFactory] = dataclasses.field(default_factory=list)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1
    online_compute_norm_stats: bool = False

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
        # pretrain on interndata a1
        TrainConfig(
        name="pretrain-interndata-a1",
        model=pi0_config.Pi0Config(),
        data=[
            # genie1
            MultiSimGenieDataConfig(
                repo_dir='data/InternData-A1/sim/*/genie1/*/*',
                task_id=None,
                use_gripper_aug=True,
                gripper_aug_config={
                    "gripper_action_keys": ["master_actions.left_gripper.openness", "master_actions.right_gripper.openness"],
                    "gripper_dim": -1,
                    "gripper_threshold_method": "std_multiplier",
                    "gripper_threshold_multiplier": 1.0,
                    "gripper_min_threshold": 0.001,
                    "gripper_max_threshold": 1.0,
                },
                stats_dir='stats/sim',
                base_config=MultiDataConfig(
                    prompt_from_task=True,
                ),
                weight=10,
                asset_id='genie1',
                robot_name='genie1',
            ),
            MultiSimGenieDataConfig(
                repo_dir='data/InternData-A1/sim/*/genie1/*',
                task_id=None,
                use_gripper_aug=True,
                gripper_aug_config={
                    "gripper_action_keys": ["master_actions.left_gripper.openness", "master_actions.right_gripper.openness"],
                    "gripper_dim": -1,
                    "gripper_threshold_method": "std_multiplier",
                    "gripper_threshold_multiplier": 1.0,
                    "gripper_min_threshold": 0.001,
                    "gripper_max_threshold": 1.0,
                },
                stats_dir='stats/sim',
                base_config=MultiDataConfig(
                    prompt_from_task=True,
                ),
                weight=10,
                asset_id='genie1',
                robot_name='genie1',
            ),
            
            # arx_lift2
            MultiSimSplitAlohaDataConfig(
                repo_dir='data/InternData-A1/sim/*/lift2/*',
                task_id=None,
                use_gripper_aug=True,
                gripper_aug_config={
                    "gripper_action_keys": ["master_actions.left_gripper.openness", "master_actions.right_gripper.openness"],
                    "gripper_dim": -1,
                    "gripper_threshold_method": "std_multiplier",
                    "gripper_threshold_multiplier": 1.0,
                    "gripper_min_threshold": 0.001,
                    "gripper_max_threshold": 1.0,
                },
                stats_dir='stats/sim',
                base_config=MultiDataConfig(
                    prompt_from_task=True,
                ),
                weight=5,
                asset_id='lift2',
                robot_name='lift2',
            ),
            MultiSimSplitAlohaDataConfig(
                repo_dir='data/InternData-A1/sim/*/lift2/*/*',
                task_id=None,
                use_gripper_aug=True,
                gripper_aug_config={
                    "gripper_action_keys": ["master_actions.left_gripper.openness", "master_actions.right_gripper.openness"],
                    "gripper_dim": -1,
                    "gripper_threshold_method": "std_multiplier",
                    "gripper_threshold_multiplier": 1.0,
                    "gripper_min_threshold": 0.001,
                    "gripper_max_threshold": 1.0,
                },
                stats_dir='stats/sim',
                base_config=MultiDataConfig(
                    prompt_from_task=True,
                ),
                weight=5,
                asset_id='lift2',
                robot_name='lift2',
            ),
            # agilex_split_aloha
            MultiSimSplitAlohaDataConfig(
                repo_dir='data/InternData-A1/sim/*/split_aloha/*',
                task_id=None,
                use_gripper_aug=True,
                gripper_aug_config={
                    "gripper_action_keys": ["master_actions.left_gripper.openness", "master_actions.right_gripper.openness"],
                    "gripper_dim": -1,
                    "gripper_threshold_method": "std_multiplier",
                    "gripper_threshold_multiplier": 1.0,
                    "gripper_min_threshold": 0.001,
                    "gripper_max_threshold": 1.0,
                },
                stats_dir='stats/sim',
                base_config=MultiDataConfig(
                    prompt_from_task=True,
                ),
                asset_id='split_aloha',
                weight=10,
                robot_name='split_aloha',
            ),
            MultiSimSplitAlohaDataConfig(
                repo_dir='data/InternData-A1/sim/*/split_aloha/*/*',
                task_id=None,
                use_gripper_aug=True,
                gripper_aug_config={
                    "gripper_action_keys": ["master_actions.left_gripper.openness", "master_actions.right_gripper.openness"],
                    "gripper_dim": -1,
                    "gripper_threshold_method": "std_multiplier",
                    "gripper_threshold_multiplier": 1.0,
                    "gripper_min_threshold": 0.001,
                    "gripper_max_threshold": 1.0,
                },
                stats_dir='stats/sim',
                base_config=MultiDataConfig(
                    prompt_from_task=True,
                ),
                asset_id='split_aloha',
                weight=10,
                robot_name='split_aloha',
            ),
            # franka
            MultiSimFrankaDataConfig(
                repo_dir='data/InternData-A1/sim/*/franka/*',
                task_id=None,
                use_gripper_aug=True,
                gripper_aug_config={
                    "gripper_action_keys": ["actions.gripper.openness"],
                    "gripper_dim": -1,
                    "gripper_threshold_method": "std_multiplier",
                    "gripper_threshold_multiplier": 1.0,
                    "gripper_min_threshold": 0.001,
                    "gripper_max_threshold": 1.0,
                },
                stats_dir='stats/sim',
                base_config=MultiDataConfig(
                    prompt_from_task=True,
                ),
                asset_id='franka',
                weight=5,
                robot_name='franka',
            ),
            MultiSimFrankaDataConfig(
                repo_dir='data/InternData-A1/sim/*/franka/*/*',
                task_id=None,
                use_gripper_aug=True,
                gripper_aug_config={
                    "gripper_action_keys": ["actions.gripper.openness"],
                    "gripper_dim": -1,
                    "gripper_threshold_method": "std_multiplier",
                    "gripper_threshold_multiplier": 1.0,
                    "gripper_min_threshold": 0.001,
                    "gripper_max_threshold": 1.0,
                },
                stats_dir='stats/sim',
                base_config=MultiDataConfig(
                    prompt_from_task=True,
                ),
                asset_id='franka',
                weight=5,
                robot_name='franka',
            ),
        ],
        # pretrain model path
        weight_loader=weight_loaders.PaliGemmaWeightLoader("checkpoints/jax/paligemma/pt_224.npz"),
        pytorch_weight_path="", 
        num_train_steps=2_000_000,
        num_workers=12,
        batch_size=512, 
        save_interval=5000,
        lr_schedule=_optimizer.WarmupConstantSchedule(),
    ),
        # finetune on real-world tasks
        TrainConfig(
        name="finetune-a2d-pen",
        model=pi0_config.Pi0Config(),
        data=[
            MultiLeRobotReala2dDataConfig(
                repo_dir='data/InternData-A1/real/genie1/Put_the_pen_from_the_table_into_the_pen_holder/*',
                task_id=["set_0"],
                use_gripper_aug=False,
                stats_dir='',
                fixed_stats_dir='stats/real/genie1/Put_the_pen_from_the_table_into_the_pen_holder',
                base_config=MultiDataConfig(
                    prompt_from_task=True,
                ),
                asset_id='finetune-a2d-pen',
                weight=1,
                robot_name='genie1',
            ),
        ],
        # pretrain model path
        weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/20251014-sim-pretrain-470k/20251014-sim-pretrain-470k-8-node-bs512-nw12/680000/params"),
        # weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/jax/pi0_base/params"),
        pytorch_weight_path="", 
        num_train_steps=30_000,
        num_workers=32,
        batch_size=128, 
        save_interval=5000,
    ),
    TrainConfig(
        name="finetune-sim2real-lift2-sort-rubbish",
        model=pi0_config.Pi0Config(),
        data=[
            MultiSim2RealSplitAlohaDataConfig(
                repo_dir='data/InternData-A1/sim/long_horizon_tasks/lift2/sort_the_rubbish/*',
                task_id=None,   # when task_id is None, we use all collections under the repo_dir
                use_gripper_aug=False,
                stats_dir='',
                fixed_stats_dir='stats/sim2real/lift2/sort_the_rubbish',
                base_config=MultiDataConfig(
                    prompt_from_task=True,
                ),
                asset_id='finetune-sim2real-lift2-sort-rubbish',
                weight=1,
                robot_name='lift2',
            ),
        ],
        # pretrain model path
        weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/20251014-sim-pretrain-470k/20251014-sim-pretrain-470k-8-node-bs512-nw12/680000/params"),
        # weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/jax/pi0_base/params"),
        pytorch_weight_path="", 
        num_train_steps=30_000,
        num_workers=32,
        batch_size=128, 
        save_interval=5000,
    ),
    ]


if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]

def check_lerobot_repo(repo_dir: str):
    if os.path.isdir(os.path.join(repo_dir, "data")) and os.path.isdir(os.path.join(repo_dir, "meta")) and os.path.isdir(os.path.join(repo_dir, "videos")):
        return True
    else:
        return False