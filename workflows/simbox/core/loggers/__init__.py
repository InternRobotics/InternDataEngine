# pylint: skip-file
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseLogger(ABC):
    def __init__(
        self,
        task_dir="Pick_up_the_object",
        language_instruction="Pick up the object.",
        detailed_language_instruction="Pick up the object with right gripper.",
        collect_info="set1-1_collector1_20250715",
        version="v1.0",
        tpi_initial_info={},
    ):
        self.log_num_steps = 0
        self.task_dir = task_dir
        self.language_instruction = language_instruction
        self.detailed_language_instruction = detailed_language_instruction
        self.version = version
        self.collect_info = collect_info
        self.tpi_initial_info = tpi_initial_info
        self.json_data_logger: Dict[str, List[Any]] = {}
        self.scalar_data_logger: Dict[str, List[Any]] = {}
        self.action_data_logger: Dict[str, List[Any]] = {}
        self.proprio_data_logger: Dict[str, List[Any]] = {}
        self.object_data_logger: Dict[str, List[Any]] = {}
        self.color_image_logger: Dict[str, List[Any]] = {}
        self.depth_image_logger: Dict[str, List[Any]] = {}
        self.seg_image_logger: Dict[str, List[Any]] = {}
        self.color_image_step_logger: Dict[str, List[Any]] = {}
        self.depth_image_step_logger: Dict[str, List[Any]] = {}
        self.seg_image_step_logger: Dict[str, List[Any]] = {}

    def update_tpi_initial_info(self, tpi_initial_info):
        self.tpi_initial_info = tpi_initial_info

    def count_timestep(self):
        self.log_num_steps += 1

    def add_json_data(self, robot, key, data):
        if robot not in self.json_data_logger:
            self.json_data_logger[robot] = {}
        self.json_data_logger[robot][key] = data

    def add_proprio_data(self, robot, key, value):
        if robot not in self.proprio_data_logger:
            self.proprio_data_logger[robot] = {}
        if key not in self.proprio_data_logger[robot]:
            self.proprio_data_logger[robot][key] = []
        self.proprio_data_logger[robot][key].append(value)

    def add_action_data(self, robot, key, value):
        if robot not in self.action_data_logger:
            self.action_data_logger[robot] = {}
        if key not in self.action_data_logger[robot]:
            self.action_data_logger[robot][key] = []
        self.action_data_logger[robot][key].append(value)

    def add_object_data(self, robot, key, value):
        if robot not in self.object_data_logger:
            self.object_data_logger[robot] = {}
        if key not in self.object_data_logger[robot]:
            self.object_data_logger[robot][key] = []
        self.object_data_logger[robot][key].append(value)

    def add_scalar_data(self, robot, key, value):
        if robot not in self.scalar_data_logger:
            self.scalar_data_logger[robot] = {}
        if key not in self.scalar_data_logger[robot]:
            self.scalar_data_logger[robot][key] = []
        self.scalar_data_logger[robot][key].append(value)

    def _add_image_data(self, data_logger, step_logger, robot, key, value, step_idx=None):
        if robot not in data_logger:
            data_logger[robot] = {}
        if key not in data_logger[robot]:
            data_logger[robot][key] = []
        data_logger[robot][key].append(value)

        if robot not in step_logger:
            step_logger[robot] = {}
        if key not in step_logger[robot]:
            step_logger[robot][key] = []
        if step_idx is None:
            step_idx = len(data_logger[robot][key]) - 1
        step_logger[robot][key].append(int(step_idx))

    def add_color_image(self, robot, key, value, step_idx=None):
        self._add_image_data(
            self.color_image_logger,
            self.color_image_step_logger,
            robot,
            key,
            value,
            step_idx=step_idx,
        )

    def add_depth_image(self, robot, key, value, step_idx=None):
        self._add_image_data(
            self.depth_image_logger,
            self.depth_image_step_logger,
            robot,
            key,
            value,
            step_idx=step_idx,
        )

    def add_seg_image(self, robot, key, value, step_idx=None):
        self._add_image_data(
            self.seg_image_logger,
            self.seg_image_step_logger,
            robot,
            key,
            value,
            step_idx=step_idx,
        )

    def clear(
        self,
        language_instruction="Pick up the object.",
        detailed_language_instruction="Pick up the object with right gripper.",
    ):
        self.language_instruction = language_instruction
        self.detailed_language_instruction = detailed_language_instruction
        self.last_qpos = None
        self.last_ee_pose = None
        self.log_num_steps = 0
        self.tpi_initial_info = {}
        self.json_data_logger = {}
        self.proprio_data_logger = {}
        self.action_data_logger = {}
        self.object_data_logger = {}
        self.scalar_data_logger = {}
        self.color_image_logger = {}
        self.depth_image_logger = {}
        self.seg_image_logger = {}
        self.color_image_step_logger = {}
        self.depth_image_step_logger = {}
        self.seg_image_step_logger = {}

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def save(self):
        pass
