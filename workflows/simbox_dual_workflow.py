import glob
import json
import os
import pickle
import random
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime

import numpy as np
import yaml
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.transformations import (
    get_relative_transform,
    pose_from_tf_matrix,
)
from omni.physx import acquire_physx_interface
from tqdm import tqdm
from yaml import Loader

from deps.world_toolkit.world_recorder import WorldRecorder
from workflows.simbox.utils.task_config_parser import TaskConfigParser

from .base import NimbusWorkFlow
from .simbox.core.controllers import get_controller_cls
from .simbox.core.loggers.lmdb_logger import LmdbLogger
from .simbox.core.loggers.utils import log_dual_obs
from .simbox.core.skills import get_skill_cls
from .simbox.core.tasks import get_task_cls
from .simbox.core.utils.collision_utils import filter_collisions
from .simbox.core.utils.utils import set_random_seed


# pylint: disable=unused-argument
@NimbusWorkFlow.register("SimBoxDualWorkFlow")
class SimBoxDualWorkFlow(NimbusWorkFlow):
    def __init__(
        self,
        world,
        task_cfg_path: str,
        scene_info: str = "dining_room_scene_info",
        random_seed: int = None,
    ):
        self.scene_info = scene_info
        self.step_replay = False
        self.random_seed = random_seed
        super().__init__(world, task_cfg_path)

    def parse_task_cfgs(self, task_cfg_path: str) -> list:
        task_cfgs = TaskConfigParser(task_cfg_path).parse_tasks()
        # Merge robot configs for each task
        for task_cfg in task_cfgs:
            self._merge_robot_configs(task_cfg)
        return task_cfgs

    def _merge_robot_configs(self, task_cfg: dict):
        """Merge robot configs from robot_config_file into task_cfg['robots']."""
        robots = task_cfg.get("robots", [])

        for robot in robots:
            robot_config_file = robot.get("robot_config_file")
            if robot_config_file:
                with open(robot_config_file, "r", encoding="utf-8") as f:
                    robot_base_cfg = yaml.load(f, Loader=Loader)

                # Merge: robot_base_cfg as base, task_cfg['robots'][i] overrides
                merged_cfg = deepcopy(robot_base_cfg)
                merged_cfg.update(robot)
                robot.clear()
                robot.update(merged_cfg)

    def reset(self, need_preload: bool = True):
        # source code noted this as debug, so it could be removed later
        from omni.isaac.core.utils.viewports import set_camera_view

        set_camera_view(eye=[1.3, 0.7, 2.7], target=[0.0, 0, 1.5], camera_prim_path="/OmniverseKit_Persp")
        # Modify config
        arena_file_path = self.task_cfg.get("arena_file", None)
        with open(arena_file_path, "r", encoding="utf-8") as arena_file:
            arena = yaml.load(arena_file, Loader=Loader)

        if "involved_scenes" in arena:
            arena["involved_scenes"] = self.scene_info

        self.task_cfg["arena"] = arena

        for obj_cfg in self.task_cfg["objects"]:
            if obj_cfg["target_class"] == "ArticulatedObject":
                if obj_cfg.get("apply_randomization", False):
                    asset_root = self.task_cfg["asset_root"]
                    art_paths = glob.glob(os.path.join(asset_root, obj_cfg["art_cat"], "*"))
                    art_paths.sort()
                    path = random.choice(art_paths)
                    info_name = obj_cfg["info_name"]
                    info_path = f"{path}/Kps/{info_name}/info.json"
                    with open(info_path, "r", encoding="utf-8") as f:
                        info = json.load(f)
                    scale = info["object_scale"][:3]

                    obj_cfg["path"] = path.replace(f"{asset_root}/", "", 1) + "/instance.usd"
                    obj_cfg["category"] = path.split("/")[-2]
                    obj_cfg["obj_info_path"] = info_path.replace(f"{asset_root}/", "", 1)
                    obj_cfg["scale"] = scale
                    self.task_cfg["data"]["collect_info"] = obj_cfg["category"]

        self.task_cfg.pop("arena_file", None)
        self.task_cfg.pop("camera_file", None)
        self.task_cfg.pop("logger_file", None)
        # Modify config done
        if self.task_cfg.get("fluid", None):
            # for fluid manipulation, only gpu mode is supportive
            physx_interface = acquire_physx_interface()
            physx_interface.overwrite_gpu_setting(1)

        self.task = get_task_cls(self.task_cfg["task"])(self.task_cfg)
        self.stage = self.world.stage
        self.stage.SetDefaultPrim(self.stage.GetPrimAtPath("/World"))
        self.world.add_task(self.task)

        # # Add hidden ground plane for physics simulation
        # from omni.isaac.core.objects import GroundPlane
        # plane = GroundPlane(
        #     prim_path="/World/GroundPlane",
        #     z_position=0.0,
        #     visible=False,
        # )

        prim_paths = []  # do not collide with each other
        global_collision_paths = []  # collide with everything

        self.robots_prim_paths = []
        for robot in self.task_cfg["robots"]:
            robot_prim_path = self.task.root_prim_path + "/" + robot["name"]
            prim_paths.append(robot_prim_path)
            self.robots_prim_paths.append(robot_prim_path)
        neglect_collision_names = self.task_cfg.get("neglect_collision_names", [])
        candidates = self.task_cfg["objects"] + self.task_cfg["arena"]["fixtures"]
        for candidate in candidates:
            candidate_prim_path = self.task.root_prim_path + "/" + candidate["name"]
            global_collision_paths.append(candidate_prim_path)
            for neglect_collision_name in neglect_collision_names:
                if neglect_collision_name in candidate["name"]:
                    prim_paths.append(candidate_prim_path)
                    global_collision_paths.remove(candidate_prim_path)

        collision_root_path = "/World/collisions"
        filter_collisions(
            self.stage,
            self.world.get_physics_context().prim_path,
            collision_root_path,
            prim_paths,
            global_collision_paths,
        )
        self.world.reset()
        self.world.step(render=True)
        self.controllers = self._initialize_controllers(self.task, self.task_cfg, self.world)
        self.skills = self._initialize_skills(self.task, self.task_cfg, self.controllers, self.world)

        for _ in range(50):
            self._init_static_objects(self.task)
            self.world.step(render=False)

        self.logger = LmdbLogger(
            task_dir=self.task_cfg["data"]["task_dir"],
            language_instruction=self.task.language_instruction,
            detailed_language_instruction=self.task.detailed_language_instruction,
            collect_info=self.task_cfg["data"]["collect_info"],
            version=self.task_cfg["data"].get("version", "v1.0"),
        )
        # Motion vectors are large dense tensors; keep LMDB logging opt-in.
        self.log_motion_vectors = bool(self.task_cfg["data"].get("log_motion_vectors", False))

        if self.random_seed is not None:
            seed = self.random_seed
        else:
            seed = time.time_ns() % (2**32)
        set_random_seed(seed)

        # while True:
        #     self.world.get_observations()
        #     # self._init_static_objects(self.task)
        #     self.world.step(render=True)

    def _initialize_skills(self, task, task_cfg, controllers, world):
        draw_points = False
        if draw_points:
            from omni.isaac.debug_draw import _debug_draw

            draw = _debug_draw.acquire_debug_draw_interface()
        else:
            draw = None

        # Initialize skills for each robot.
        skills = []
        for cfg_skill_dict in task_cfg["skills"]:
            skill_dict = defaultdict(list)
            for robot_name, robot_skill_list in cfg_skill_dict.items():
                robot = task.robots[robot_name]
                controller = controllers[robot_name]

                for lr_skill_dict in robot_skill_list:
                    skill_sequence = [
                        [
                            get_skill_cls(skill_cfg["name"])(
                                robot,
                                controller[lr_name],
                                task,
                                skill_cfg,
                                world=world,
                                draw=draw,
                            )
                            for skill_cfg in lr_skill_list
                        ]
                        for lr_name, lr_skill_list in lr_skill_dict.items()
                    ]
                    skill_dict[robot_name].append(skill_sequence)
            skills.append(skill_dict)
        return skills

    def _initialize_controllers(self, task, task_cfg, world):
        """Initialize controllers for each robot."""
        controllers = {}
        for robot in task_cfg["robots"]:
            controllers[robot["name"]] = {}
            for robot_file in robot["robot_file"]:
                controller_name = "left" if "left" in robot_file else "right"
                controllers[robot["name"]][controller_name] = get_controller_cls(robot["target_class"])(
                    name=robot["name"],
                    robot_file=robot_file,
                    constrain_grasp_approach=robot.get("constrain_grasp_approach", False),
                    collision_activation_distance=robot.get("collision_activation_distance", 0.03),
                    task=task,
                    world=world,
                    ignore_substring=robot.get("ignore_substring", ["material", "Plane", "conveyor", "scene", "table"]),
                    use_batch=robot.get("use_batch", False),
                )
                controllers[robot["name"]][controller_name].reset()
        return controllers

    def _initialize_world_recorder(self):
        """
        Initialize WorldRecorder with appropriate mode based on configuration.

        Supports two modes:
        - step_replay=False: Records prim poses for fast geometric replay (compatible with old workflow)
        - step_replay=True: Uses preprocessed joint position data for physics-accurate replay (new default)
        """
        self.world_recorder = WorldRecorder(
            self.world,
            self.task.robots,
            self.task.objects | self.task.distractors | self.task.visuals,
            step_replay=self.step_replay,
        )
        self.world_recorder.reset()

    def _reset_controllers(self, controllers):
        """Reset all controllers."""
        for _, controller in controllers.items():
            for _, ctrl in controller.items():
                ctrl.reset()

    def _init_static_objects(self, task):
        for _, obj in task.objects.items():
            try:
                init_translation = obj.init_translation
                init_orientation = obj.init_orientation
                init_parent = obj.init_parent
                if init_translation and init_orientation and init_parent:
                    parent_world_pose = get_relative_transform(
                        get_prim_at_path(task.root_prim_path + "/" + init_parent), get_prim_at_path(task.root_prim_path)
                    )
                    parent_translation, _ = pose_from_tf_matrix(parent_world_pose)
                    obj.set_local_pose(
                        translation=(parent_translation + init_translation), orientation=init_orientation
                    )
                    obj.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
                    obj.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
            except Exception:
                pass

    def _randomization_layout_mem(self):
        # Reset world
        self.world.reset()

        # Individual initialize
        self.task.individual_randomize_from_mem()
        self.task.post_reset()

        self.world.step(render=False)

        # Reset controllers
        self._reset_controllers(self.controllers)

        # Reset skills
        del self.skills
        self.skills = self._initialize_skills(self.task, self.task_cfg, self.controllers, self.world)

        # Warmup
        for _ in range(20):
            self.world.get_observations()
            self._init_static_objects(self.task)
            self.world.step(render=False)

        self._initialize_world_recorder()

        self.logger.clear(
            language_instruction=self.task.language_instruction,
            detailed_language_instruction=self.task.detailed_language_instruction,
        )

        # episode_stats["current_times"] += 1

    def _randomization_layout(self):
        # Reset world
        self.world.reset()

        # Individual initialize
        self.task.individual_randomize()
        self.task.post_reset()

        self.world.step(render=False)

        # Reset controllers
        if self.task_cfg.get("fluid", None):
            # Fluid, Bug, Why !!!!!!
            # For fluid manipulation, only delete controllers and reinitialize controllers can plan successfully
            if hasattr(self, "controllers"):
                del self.controllers
            self.controllers = self._initialize_controllers(self.task, self.task_cfg, self.world)

        # del self.controllers
        # self.controllers = self._initialize_controllers(self.task, self.task_cfg, self.world)
        self._reset_controllers(self.controllers)

        # Reset skills
        if hasattr(self, "skills"):
            del self.skills

        self.skills = self._initialize_skills(self.task, self.task_cfg, self.controllers, self.world)

        # Warmup
        for _ in range(20):
            self.world.get_observations()
            self._init_static_objects(self.task)
            self.world.step(render=False)

        if self.task_cfg.get("fluid", None):
            self.task._set_fluid()
            # Fluid need additional warmup
            for _ in range(150):
                self.world.step(render=False)

        self._initialize_world_recorder()

        self.logger.clear(
            language_instruction=self.task.language_instruction,
            detailed_language_instruction=self.task.detailed_language_instruction,
        )

        # episode_stats["current_times"] += 1

    def randomization(self, layout_path=None) -> bool:
        try:
            if layout_path is None:
                # Individual Reset
                self.task.individual_reset()
                self._randomization_layout()
            else:
                with open(layout_path, "rb") as f:
                    data = pickle.load(f)
                self.data = data
                self.randomization_from_mem(data)
            return True
        except Exception as e:
            raise e

    def update_skill_states(self, skills, episode_success, should_continue):
        """Update and manage skill states."""
        current_skills = skills[0]

        # Check if any skills remain
        if not any(current_skills.values()):
            skills.pop(0)
            if skills:
                should_continue = self.plan_first_skill(skills, should_continue)
            return episode_success, should_continue

        # Update each robot's skills
        for _, skill_sequences in current_skills.items():
            if not skill_sequences:
                continue

            # Update all skills first
            for lr_skill_list in skill_sequences[0]:
                if lr_skill_list:
                    start_lr_skill = lr_skill_list[0]
                    start_lr_skill.update()  # Must update regardless of completion
                    if start_lr_skill.is_done():
                        if not start_lr_skill.is_success():
                            episode_success = False
                            should_continue = False
                        lr_skill_list.remove(start_lr_skill)

                        if lr_skill_list:
                            next_skill = lr_skill_list[0]
                            next_skill.simple_generate_manip_cmds()
                            if hasattr(next_skill, "visualize_target"):
                                next_skill.visualize_target(self.world)
                            if len(next_skill.manip_list) == 0:
                                should_continue = not next_skill.is_ready()
                    if hasattr(start_lr_skill, "visualize_target"):
                        start_lr_skill.visualize_target(self.world)

            # Remove empty skill sequences
            completed_skills = []
            for lr_skill_list in skill_sequences[0]:
                if not lr_skill_list:
                    completed_skills.append(lr_skill_list)
            for completed_skill in completed_skills:
                skill_sequences[0].remove(completed_skill)

            # Move to next sequence if current is empty
            if not skill_sequences[0]:
                skill_sequences.pop(0)
                if skill_sequences:
                    for skill in skill_sequences[0]:
                        skill[0].simple_generate_manip_cmds()
                        if len(skill[0].manip_list) == 0:
                            should_continue = not skill[0].is_ready()
        return episode_success, should_continue

    def plan_first_skill(self, skills, should_continue):
        for _, robot_skill_list in skills[0].items():
            for lr_skill_list in robot_skill_list[0]:
                lr_skill_list[0].simple_generate_manip_cmds()
                if hasattr(lr_skill_list[0], "visualize_target"):
                    lr_skill_list[0].visualize_target(self.world)
                if len(lr_skill_list[0].manip_list) == 0:
                    should_continue = not lr_skill_list[0].is_ready()
        return should_continue

    def generate_seq(self) -> list:
        end = False

        # while True:
        #     obs = self.world.get_observations()
        #     # self._init_static_objects(self.task)
        #     self.world.step(render=True)

        step_id = 0
        episode_success = True
        should_continue = True
        max_episode_length = self.task_cfg["data"]["max_episode_length"]
        episode_stats = {"succeed_times": 0, "current_times": 0}

        should_continue = self.plan_first_skill(self.skills, should_continue)

        # Warmup
        for _ in range(10):
            obs = self.world.get_observations()
            # self._init_static_objects(self.task)
            self.world.step(render=False)

        while not (step_id >= max_episode_length or (not self.skills and not episode_success) or (not should_continue)):
            obs = self.world.get_observations()
            action_dict = {}
            record_flag = True
            if self.skills and should_continue:
                # Process current skills
                current_skills = self.skills[0]
                for robot_name, skill_sequences in current_skills.items():
                    if skill_sequences and skill_sequences[0]:
                        action = [
                            skill[0].controller.forward(skill[0].manip_list[0])
                            for skill in skill_sequences[0]
                            if skill[0] and skill[0].is_ready()
                        ]

                        feasible_labels = [skill[0].is_feasible() for skill in skill_sequences[0] if skill[0]]
                        record_labels = [skill[0].is_record() for skill in skill_sequences[0] if skill[0]]

                        if False in feasible_labels:
                            should_continue = False
                        if False in record_labels:
                            record_flag = False

                        if action:
                            action_dict[robot_name] = {
                                "joint_positions": np.concatenate([a["joint_positions"] for a in action]),
                                "joint_indices": np.concatenate([a["joint_indices"] for a in action]),
                                "raw_action": action,
                            }
            elif not self.skills and episode_success:
                print("Task is successful")
                end = True
                for j_idx in range(1, 7):
                    self.world.step(render=False)
                    obs = self.world.get_observations()
                    log_dual_obs(self.logger, obs, action_dict, self.controllers, step_idx=step_id + j_idx)
                    self.world_recorder.record()

                episode_stats["succeed_times"] += 1
                should_continue = False

            if record_flag:
                log_dual_obs(self.logger, obs, action_dict, self.controllers, step_idx=step_id)
                self.world_recorder.record()
            self.task.apply_action(action_dict)
            self.world.step(render=False)

            step_id += 1
            if self.skills:
                episode_success, should_continue = self.update_skill_states(
                    self.skills, episode_success, should_continue
                )

        if end:
            if self.step_replay:
                return [None] * step_id
            else:
                # Prim poses mode: return recorded poses for compatibility
                return self.world_recorder.prim_poses
        else:
            return []

    def recover_seq(self, seq_path):
        data = self.data
        return self.recover_seq_from_mem(data)

    def _record_rgb_depth(self, step_idx: int):
        for key, value in self.task.cameras.items():
            for robot_name, _ in self.task.robots.items():
                if robot_name in key:
                    camera_obs = value.get_observations()
                    rgb_img = camera_obs["color_image"]
                    # Special processing if enabled
                    camera2env_pose = camera_obs["camera2env_pose"]
                    save_camera_name = key.replace(f"{robot_name}_", "")
                    self.logger.add_color_image(
                        robot_name, "images.rgb." + save_camera_name, rgb_img, step_idx=step_idx
                    )
                    if "depth_image" in camera_obs:
                        depth_image = camera_obs["depth_image"]
                        depth_img = np.nan_to_num(depth_img, nan=0.0, posinf=0.0, neginf=0.0)
                        self.logger.add_depth_image(
                            robot_name, "images.depth." + save_camera_name, depth_img, step_idx=step_idx
                        )
                    if "semantic_mask" in camera_obs:
                        self.logger.add_seg_image(
                            robot_name, "images.seg." + save_camera_name, seg_mask, step_idx=step_idx
                        )
                        if "semantic_mask_id2labels" in camera_obs:
                            self.logger.add_scalar_data(
                                robot_name,
                                "labels.seg." + save_camera_name,
                                camera_obs["semantic_mask_id2labels"],
                            )
                    if "bbox2d_tight" in camera_obs:
                        self.logger.add_scalar_data(
                            robot_name, "labels.bbox2d_tight." + save_camera_name, camera_obs["bbox2d_tight"]
                        )
                    if "bbox2d_tight_id2labels" in camera_obs:
                        self.logger.add_scalar_data(
                            robot_name,
                            "labels.bbox2d_tight_id2labels." + save_camera_name,
                            camera_obs["bbox2d_tight_id2labels"],
                        )
                    if "bbox2d_loose" in camera_obs:
                        self.logger.add_scalar_data(
                            robot_name, "labels.bbox2d_loose." + save_camera_name, camera_obs["bbox2d_loose"]
                        )
                    if "bbox2d_loose_id2labels" in camera_obs:
                        self.logger.add_scalar_data(
                            robot_name,
                            "labels.bbox2d_loose_id2labels." + save_camera_name,
                            camera_obs["bbox2d_loose_id2labels"],
                        )
                    if "bbox3d" in camera_obs:
                        self.logger.add_scalar_data(
                            robot_name, "labels.bbox3d." + save_camera_name, camera_obs["bbox3d"]
                        )
                    if "bbox3d_id2labels" in camera_obs:
                        self.logger.add_scalar_data(
                            robot_name,
                            "labels.bbox3d_id2labels." + save_camera_name,
                            camera_obs["bbox3d_id2labels"],
                        )
                    if self.log_motion_vectors and "motion_vectors" in camera_obs:
                        self.logger.add_scalar_data(
                            robot_name, "labels.motion_vectors." + save_camera_name, camera_obs["motion_vectors"]
                        )
                    self.logger.add_scalar_data(
                        robot_name, "camera2env_pose." + save_camera_name, camera2env_pose
                    )
                    if step_idx == 0:
                        save_camera_name = key.replace(f"{robot_name}_", "")
                        self.logger.add_json_data(
                            robot_name, f"{save_camera_name}_camera_params", camera_obs["camera_params"]
                        )

                    # depth_img = get_src(value, "depth")
                    # depth_img = np.nan_to_num(depth_img, nan=0.0, posinf=0.0, neginf=0.0)

                    # # Initialize lists for new camera keys
                    # if key not in self.rgb:
                    #     self.rgb[key] = []
                    # if key not in self.depth:
                    #     self.depth[key] = []

                    # # Append current frame to the corresponding camera's list
                    # self.rgb[key].append(rgb_img)
                    # self.depth[key].append(depth_img)

    def seq_replay(self, sequence: list) -> int:
        """
        Replay recorded sequence with mode-specific data preparation.

        Returns:
            int: Number of steps replayed
        """
        if not self.step_replay:
            self.world_recorder.prim_poses = sequence

        # warmup before replay formally
        self.world_recorder.warmup()

        # Get total steps from WorldRecorder
        total_steps = self.world_recorder.get_total_steps()
        step_idx = 0

        # Unified replay loop - WorldRecorder handles rendering internally
        with tqdm(total=total_steps, desc="Replay Progress") as pbar:
            while not self.world_recorder.replay():
                # Record RGB/depth at current step
                self._record_rgb_depth(step_idx)
                step_idx += 1
                pbar.update(1)

        self.length = total_steps
        print("Replay finished.")
        return total_steps

    def get_task_name(self):
        return self.task_cfg["task"]

    def save_seq(self, save_path: str) -> int:
        ser_bytes = self.dump_plan_info()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S_%f")
        save_path = os.path.join(save_path, "plan")
        os.makedirs(save_path, exist_ok=True)
        path = os.path.join(save_path, f"{timestamp}.pkl")
        with open(path, "wb") as f:
            f.write(ser_bytes)
        return self.world_recorder.get_total_steps()

    def save(self, save_path: str) -> int:
        os.makedirs(save_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S_%f")
        self.logger.save(save_path, timestamp, save_img=True)

        return self.length

    def plan_with_render(self):
        end = False

        step_id = 0
        length = 0
        episode_success = True
        should_continue = True
        max_episode_length = self.task_cfg["data"]["max_episode_length"]
        episode_stats = {"succeed_times": 0, "current_times": 0}

        should_continue = self.plan_first_skill(self.skills, should_continue)

        # Warmup
        for _ in range(10):
            obs = self.world.get_observations()
            # self._init_static_objects(self.task)
            self.world.step(render=True)

        # while True:
        #     obs = self.world.get_observations()
        #     # self._init_static_objects(self.task)
        #     self.world.step(render=True)

        while not (step_id >= max_episode_length or (not self.skills and not episode_success) or (not should_continue)):
            obs = self.world.get_observations()
            action_dict = {}
            record_flag = True
            if self.skills and should_continue:
                # Process current skills
                current_skills = self.skills[0]
                for robot_name, skill_sequences in current_skills.items():
                    if skill_sequences and skill_sequences[0]:
                        action = [
                            skill[0].controller.forward(skill[0].manip_list[0])
                            for skill in skill_sequences[0]
                            if skill[0] and skill[0].is_ready()
                        ]

                        feasible_labels = [skill[0].is_feasible() for skill in skill_sequences[0] if skill[0]]
                        record_labels = [skill[0].is_record() for skill in skill_sequences[0] if skill[0]]

                        if False in feasible_labels:
                            should_continue = False
                        if False in record_labels:
                            record_flag = False

                        if action:
                            action_dict[robot_name] = {
                                "joint_positions": np.concatenate([a["joint_positions"] for a in action]),
                                "joint_indices": np.concatenate([a["joint_indices"] for a in action]),
                                "raw_action": action,
                            }
            elif not self.skills and episode_success:
                print("Task is successful")
                end = True
                for j_idx in range(1, 7):
                    self.world.step(render=True)
                    obs = self.world.get_observations()
                    log_dual_obs(self.logger, obs, action_dict, self.controllers, step_idx=step_id + j_idx)
                    self._record_rgb_depth(step_id + j_idx)
                    self.world_recorder.record()
                length = step_id + 6
                episode_stats["succeed_times"] += 1
                should_continue = False

            if record_flag:
                log_dual_obs(self.logger, obs, action_dict, self.controllers, step_idx=step_id)
                self._record_rgb_depth(step_id)
            self.task.apply_action(action_dict)
            self.world.step(render=True)

            step_id += 1
            if self.skills:
                episode_success, should_continue = self.update_skill_states(
                    self.skills, episode_success, should_continue
                )

        self.length = length
        if end:
            return length
        else:
            return 0

    def _dump_task_cfg(self, task_cfg):
        task_cfg_copy = deepcopy(task_cfg)
        return pickle.dumps(task_cfg_copy)

    def dump_plan_info(self) -> bytes:
        logger_ser = self.logger.dump()
        cfg_ser = self._dump_task_cfg(self.task_cfg)
        ser = pickle.dumps((cfg_ser, self.world_recorder.dumps(), logger_ser))
        return ser

    def dedump_plan_info(self, ser_obj: bytes) -> object:
        res = pickle.loads(ser_obj)
        return res

    def randomization_from_mem(self, data) -> bool:
        try:
            cfg_ser, _, _ = data
            task_cfg = pickle.loads(cfg_ser)
            self.task_cfg = task_cfg
            self.task.cfg = task_cfg

            # Individual Reset
            self.task.individual_reset_from_mem()
            self._randomization_layout_mem()
            return True
        except Exception as e:
            raise e

    def recover_seq_from_mem(self, data) -> list:
        """
        Recover sequence from memory based on WorldRecorder mode.

        Returns:
            - step_replay=False: Returns prim_poses list
            - step_replay=True: Returns placeholder list (replay data is in WorldRecorder)
        """
        try:
            _, wr_ser, logger_ser = data
            self.logger.dedump(logger_ser)

            if wr_ser:
                self.world_recorder.loads(wr_ser)

            if self.step_replay:
                return [None] * self.world_recorder.num_steps
            else:
                return self.world_recorder.prim_poses

        except Exception as e:
            raise e
