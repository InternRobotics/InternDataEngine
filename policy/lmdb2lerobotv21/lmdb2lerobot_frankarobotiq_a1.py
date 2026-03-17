import argparse
import json
import logging
import os
import gc
import shutil
import torchvision
import cv2
import h5py
import lmdb
import numpy as np
import pickle
import torch
import pinocchio as pin
import time
import ray
import logging
import pdb
import os
import imageio # imageio-ffmpeg

from PIL import Image
from tqdm import tqdm
from lerobot.common.datasets.compute_stats import auto_downsample_height_width, get_feature_stats, sample_indices
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import check_timestamps_sync, get_episode_data_index, validate_episode_buffer
from ray.runtime_env import RuntimeEnv
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
from concurrent.futures import ALL_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

"""
    Store both camera image and robot state as a combined observation. 
    Args:
        observation: images(camera), states (robot state)
        actions: joint, gripper, ee_pose
"""
FEATURES = {
    "images.rgb.head": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
    },
    "images.rgb.hand": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
    },
    "head_camera_intrinsics": {
        "dtype": "float32",
        "shape": (4,),
        "names": ["fx", "fy", "cx", "cy"],        
    },
    "hand_camera_intrinsics": {
        "dtype": "float32",
        "shape": (4,),
        "names": ["fx", "fy", "cx", "cy"],        
    },
    "head_camera_to_robot_extrinsics": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["position.x", "position.y", "position.z", "quaternion.w", "quaternion.x", "quaternion.y", "quaternion.z"],
    },
    "hand_camera_to_robot_extrinsics": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["position.x", "position.y", "position.z", "quaternion.w", "quaternion.x", "quaternion.y", "quaternion.z"],
    },
    "states.joint.position": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6",],
    },
    "states.gripper.position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["gripper_0",],
    },
    "states.gripper.pose": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["x", "y", "z", "roll", "pitch", "yaw",],
    },
    "states.ee_to_armbase_pose": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["position.x", "position.y", "position.z", "quaternion.w", "quaternion.x", "quaternion.y", "quaternion.z"],
    },
    "states.ee_to_robot_pose": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["position.x", "position.y", "position.z", "quaternion.w", "quaternion.x", "quaternion.y", "quaternion.z"],
    },  
    "states.tcp_to_armbase_pose": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["position.x", "position.y", "position.z", "quaternion.w", "quaternion.x", "quaternion.y", "quaternion.z"],
    },
    "states.tcp_to_robot_pose": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["position.x", "position.y", "position.z", "quaternion.w", "quaternion.x", "quaternion.y", "quaternion.z"],
    }, 
    "states.robot_to_env_pose": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["position.x", "position.y", "position.z", "quaternion.w", "quaternion.x", "quaternion.y", "quaternion.z"],
    },
    "actions.joint.position": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6",],
    },
    "actions.gripper.position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["gripper_0",],
    },
    "actions.gripper.openness": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["gripper_0",],
    },
    "actions.gripper.pose": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["x", "y", "z", "roll", "pitch", "yaw",],
    },
    "actions.ee_to_armbase_pose": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["position.x", "position.y", "position.z", "quaternion.w", "quaternion.x", "quaternion.y", "quaternion.z"],
    },
    "actions.ee_to_robot_pose": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["position.x", "position.y", "position.z", "quaternion.w", "quaternion.x", "quaternion.y", "quaternion.z"],
    },  
    "actions.tcp_to_armbase_pose": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["position.x", "position.y", "position.z", "quaternion.w", "quaternion.x", "quaternion.y", "quaternion.z"],
    },
    "actions.tcp_to_robot_pose": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["position.x", "position.y", "position.z", "quaternion.w", "quaternion.x", "quaternion.y", "quaternion.z"],
    }, 
    "master_actions.joint.position": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6",],
    },
    "master_actions.gripper.position": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["gripper_0",],
    },
    "master_actions.gripper.openness": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["gripper_0",],
    },
    "master_actions.gripper.pose": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["x", "y", "z", "roll", "pitch", "yaw",],
    },
}

class FrankaDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        download_videos: bool = True,
        local_files_only: bool = False,
        video_backend: str | None = None,
    ):
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            download_videos=download_videos,
            local_files_only=local_files_only,
            video_backend=video_backend,
        )

    def save_episode(self, episode_data: dict | None = None, videos: dict | None = None) -> None:
        if not episode_data:
            episode_buffer = self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])
        for key, ft in self.features.items():
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key]).squeeze()
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            episode_buffer[key] = str(video_path)  # PosixPath -> str
            video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(videos[key], video_path)
        ep_stats = compute_episode_stats(episode_buffer, self.features)
        self._save_episode_table(episode_buffer, episode_index)
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)
        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )
        if not episode_data:
            self.episode_buffer = self.create_episode_buffer()


    def add_frame(self, frame: dict) -> None:
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()
        features = {key: value for key, value in self.features.items() if key in self.hf_features}
        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()
        frame_index = self.episode_buffer["size"]
        timestamp = frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)

        for key in frame:
            if key == "task":
                self.episode_buffer["task"].append(frame["task"])
                continue
            if key not in self.features:
                raise ValueError(f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'.")
            self.episode_buffer[key].append(frame[key])
        self.episode_buffer["size"] += 1

# def crop_resize_no_padding(image, target_size=(480, 640)):
#     """
#     Crop and scale to target size (no padding)
#     :param image: input image (NumPy array)
#     :param target_size: target size (height, width)
#     :return: processed image
#     """
#     h, w = image.shape[:2]
#     target_h, target_w = target_size
#     target_ratio = target_w / target_h  # Target aspect ratio (e.g. 640/480=1.333)

#     # the original image aspect ratio and cropping direction
#     if w / h > target_ratio:  # Original image is wider → crop width
#         crop_w = int(h * target_ratio)  # Calculate crop width based on target aspect ratio
#         crop_h = h
#         start_x = (w - crop_w) // 2  # Horizontal center starting point
#         start_y = 0
#     else:  # Original image is higher → crop height
#         crop_h = int(w / target_ratio)  # Calculate clipping height according to target aspect ratio
#         crop_w = w
#         start_x = 0
#         start_y = (h - crop_h) // 2  # Vertical center starting point

#     # Perform centered cropping (to prevent out-of-bounds)
#     start_x, start_y = max(0, start_x), max(0, start_y)
#     end_x, end_y = min(w, start_x + crop_w), min(h, start_y + crop_h)
#     cropped = image[start_y:end_y, start_x:end_x]

#     # Resize to target size (bilinear interpolation)
#     resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
#     return resized

def tf2xyzwxyz(posetf):
    translation = posetf[:3, 3]
    orientation = R.from_matrix(posetf[:3,:3]).as_quat(scalar_first=True) # w, x, y, z
    xyzwxyz = (np.concatenate([translation, orientation])).astype("float32")

    return xyzwxyz

def load_lmdb_data(episode_path: Path, sava_path: Path, fps_factor: int, target_fps: int) -> Optional[Dict]:
    def load_image(txn, key):
        raw = txn.get(key)
        data = pickle.loads(raw)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        # Convert to RGB if necessary
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = crop_resize_no_padding(image, target_size=(480, 640))
        return image
    armbase_to_robot_pose = np.eye(4)
    rot_x = np.eye(4)
    rot_x[1][1] = -1.0
    rot_x[2][2] = -1.0
    tcp2ee_pose = np.eye(4)
    tcp2ee_pose[2, 3] = 0.145

    model = pin.buildModelFromUrdf("../assets/franka_robotiq/frankarobotiq.urdf")
    data = model.createData()

    try:
        env = lmdb.open(
            str(episode_path / "lmdb"),
            readonly=True,
            lock=False,
            max_readers=128,
            readahead=False
        )
        meta_info = pickle.load(open(episode_path/"meta_info.pkl", "rb"))
        with env.begin(write=False) as txn:
            keys = [k for k, _ in txn.cursor()]
            # import pdb; pdb.set_trace()
            qpos_keys = ['states.joint.position', 'states.gripper.position', 'states.gripper.pose',]
            master_action_keys = ['master_actions.joint.position', 'master_actions.gripper.position', 'master_actions.gripper.openness', 'master_actions.gripper.pose',]
            action_keys = ['actions.joint.position', 'actions.gripper.position', 'actions.gripper.openness', 'actions.gripper.pose',]
            image_keys = ['images.rgb.head', 'images.rgb.hand',]
            compute_qpos_keys = ['states.joint.position']
            additional_action_keys = ["actions.ee_to_armbase_pose", "actions.ee_to_robot_pose", "actions.tcp_to_armbase_pose", "actions.tcp_to_robot_pose"]
            robot2env_keys = ['robot2env_pose']
            intrinsics_keys = ['json_data']
            camera2env_keys = ["camera2env_pose.head", "camera2env_pose.hand"]
            total_steps = []
            for image_key in image_keys:
                keys_image_per_step = meta_info['keys'][image_key]
                total_steps.append(len(keys_image_per_step))
            
            state_action_dict = {}
            ### qpos
            for key in qpos_keys:
                state_action_dict[key] = pickle.loads(txn.get(key.encode()))
                state_action_dict[key] = np.stack(state_action_dict[key])
                total_steps.append(len(state_action_dict[key]))
            state_keys = list(state_action_dict.keys())
            # ### next qpos as action
            # for k in state_keys:
            #     state_action_dict[k.replace("states", "actions")] = np.concatenate([state_action_dict[k][1:, :], state_action_dict[k][-1, :][None,:]], axis=0)
            ### master action
            for key in master_action_keys:
                state_action_dict[key] = pickle.loads(txn.get(key.encode()))
                if np.isscalar(state_action_dict[key]):
                    state_action_dict[key] = np.array([state_action_dict[key]]).astype("float32")
                state_action_dict[key] = np.stack(state_action_dict[key])
                #if "openness" in key:
                if len(state_action_dict[key].shape)==1:
                    state_action_dict[key] = state_action_dict[key][:, np.newaxis]
                total_steps.append(len(state_action_dict[key]))
            ### action
            for key in action_keys:
                state_action_dict[key] = pickle.loads(txn.get(key.encode()))
                if np.isscalar(state_action_dict[key]):
                    state_action_dict[key] = np.array([state_action_dict[key]]).astype("float32")
                state_action_dict[key] = np.stack(state_action_dict[key])
                #if "openness" in key:
                if len(state_action_dict[key].shape)==1:
                    state_action_dict[key] = state_action_dict[key][:, np.newaxis]
                total_steps.append(len(state_action_dict[key]))
            ### ee & tcp pose proprio
            for compute_qpos_key in compute_qpos_keys:
                compute_qpos = pickle.loads(txn.get(compute_qpos_key.encode()))
                ee_to_armbase_poses = []
                ee_to_robot_poses = []
                tcp_to_armbase_poses = []
                tcp_to_robot_poses = []                
                for each_compute_qpos in compute_qpos:
                    q = np.zeros(model.nq)   # 关节角
                    ndim = each_compute_qpos.shape[0]
                    q[:ndim] = each_compute_qpos
                    pin.forwardKinematics(model, data, q)
                    pin.updateFramePlacements(model, data)

                    fid_a = model.getFrameId("base_link")
                    fid_b = model.getFrameId("panda_link8")

                    T_a = data.oMf[fid_a]    # world -> a
                    T_b = data.oMf[fid_b]    # world -> b
                    T_a_b = T_a.inverse() * T_b

                    ee2a_translation = T_a_b.homogeneous[:3, 3]
                    ee2a_orientation = R.from_matrix(T_a_b.homogeneous[:3,:3]).as_quat(scalar_first=True) # w, x, y, z

                    ee_to_armbase_pose = (np.concatenate([ee2a_translation, ee2a_orientation])).astype("float32")
                    ee_to_armbase_poses.append(ee_to_armbase_pose)

                    tcp_to_arm_base_posetf = T_a_b.homogeneous @ tcp2ee_pose
                    tcp_to_arm_base_translation = tcp_to_arm_base_posetf[:3, 3]
                    tcp_to_arm_base_orientation = R.from_matrix(tcp_to_arm_base_posetf[:3,:3]).as_quat(scalar_first=True) # w, x, y, z
                    tcp_to_armbase_pose = (np.concatenate([tcp_to_arm_base_translation, tcp_to_arm_base_orientation])).astype("float32")
                    tcp_to_armbase_poses.append(tcp_to_armbase_pose)
                    ee_to_robot_posetf = armbase_to_robot_pose @ T_a_b.homogeneous
                    ee2r_translation = ee_to_robot_posetf[:3, 3]
                    ee2r_orientation = R.from_matrix(ee_to_robot_posetf[:3,:3]).as_quat(scalar_first=True) # w, x, y, z

                    ee_to_robot_pose = (np.concatenate([ee2r_translation, ee2r_orientation])).astype("float32")
                    ee_to_robot_poses.append(ee_to_robot_pose)
                    
                    tcp_to_robot_posetf = ee_to_robot_posetf @ tcp2ee_pose
                    tcp_to_robot_translation = tcp_to_robot_posetf[:3, 3]
                    tcp_to_robot_orientation = R.from_matrix(tcp_to_robot_posetf[:3,:3]).as_quat(scalar_first=True) # w, x, y, z
                    tcp_to_robot_pose = (np.concatenate([tcp_to_robot_translation, tcp_to_robot_orientation])).astype("float32")
                    tcp_to_robot_poses.append(tcp_to_robot_pose)

                ee2a_key = f"states.ee_to_armbase_pose"
                ee2r_key = f"states.ee_to_robot_pose"
                tcp2a_key = f"states.tcp_to_armbase_pose"
                tcp2r_key = f"states.tcp_to_robot_pose"

                state_action_dict[ee2a_key] = np.stack(ee_to_armbase_poses)
                state_action_dict[ee2r_key] = np.stack(ee_to_robot_poses)
                state_action_dict[tcp2a_key] = np.stack(tcp_to_armbase_poses)
                state_action_dict[tcp2r_key] = np.stack(tcp_to_robot_poses)
            ### ee & tcp pose action
            for additional_action_key in additional_action_keys:
                additional_state_key = additional_action_key.replace("actions", "states")
                additional_state = state_action_dict[additional_state_key]
                additional_action = np.concatenate([additional_state[1:, :], additional_state[-1, :][None,:]], axis=0)
                state_action_dict[additional_action_key] = additional_action
            ### intrinsics pose 
            num_steps = (state_action_dict[ee2a_key]).shape[0]
            for intrinsics_key in intrinsics_keys:
                intrinsics_params = pickle.loads(txn.get(intrinsics_key.encode()))
                hand_camera_params = intrinsics_params["hand_camera_params"]
                hand_camera_params = (np.array([hand_camera_params[0][0], hand_camera_params[1][1], hand_camera_params[0][2], hand_camera_params[1][2]])).astype("float32")
                head_camera_params = intrinsics_params["head_camera_params"]
                head_camera_params = (np.array([head_camera_params[0][0], head_camera_params[1][1], head_camera_params[0][2], head_camera_params[1][2]])).astype("float32")
                if head_camera_params[2] >= 500:
                    head_camera_params /= 2
                state_action_dict["head_camera_intrinsics"] = np.stack([head_camera_params for _ in range(num_steps)])
                state_action_dict["hand_camera_intrinsics"] = np.stack([hand_camera_params for _ in range(num_steps)])
            ### robot2env pose
            for robot2env_key in robot2env_keys:
                robot2env_pose_tfs = pickle.loads(txn.get(robot2env_key.encode()))
                robot2env_pose_7ds = []
                for robot2env_pose_tf in robot2env_pose_tfs:
                    translation = robot2env_pose_tf[:3, 3]
                    orientation = R.from_matrix(robot2env_pose_tf[:3,:3]).as_quat(scalar_first=True) # w, x, y, z
                    robot2env_pose_7d = (np.concatenate([translation, orientation])).astype("float32")
                    robot2env_pose_7ds.append(robot2env_pose_7d)
                state_action_dict[robot2env_key] = np.stack(robot2env_pose_7ds)
            ### camera2env pose
            for camera2env_key in camera2env_keys:
                camera2env_pose_tfs = pickle.loads(txn.get(camera2env_key.encode()))
                camera2robot_poses = []
                for frame_idx in range(len(camera2env_pose_tfs)):
                    camera2env_posetf = camera2env_pose_tfs[frame_idx]
                    robot2env_pose_tf = robot2env_pose_tfs[frame_idx]
                    camera2robot_pose_tf = np.linalg.inv(robot2env_pose_tf) @ camera2env_posetf @ rot_x
                    camera2robot_poses.append(tf2xyzwxyz(camera2robot_pose_tf))
                
                if camera2env_key == "camera2env_pose.head":
                    state_action_dict["head_camera_to_robot_extrinsics"] = np.stack(camera2robot_poses)
                elif camera2env_key == "camera2env_pose.hand":
                     state_action_dict["head_camera_to_robot_extrinsics"] = np.stack(camera2robot_poses)

            unique_steps = list(set(total_steps))
            # import pdb; pdb.set_trace()
            print("episode_path:", episode_path)
            print("total_steps: ", total_steps)
            assert len(unique_steps) == 1 and unique_steps[0]>0, f"no data found or qpos / image steps mismatch in {episode_path}"
            assert unique_steps[0]>100, f"Episode length of {episode_path} is {unique_steps[0]}, which does not meet requirements"
            assert np.max(np.abs(state_action_dict["states.joint.position"])) < 2*np.pi
            selected_steps = [step for step in range(unique_steps[0]) if step % fps_factor == 0]
            frames = []
            image_observations = {}
            for image_key in image_keys:
                image_observations[image_key] = []
            start_time = time.time()
            for step_index, step in enumerate(selected_steps):
                step_str = f"{step:04d}"
                data_dict = {}
                for key, value in state_action_dict.items():
                    # if "forlan2robot_pose" in key:
                    #     data_dict["states.ee_to_armbase_pose"] = value[step]
                    if "robot2env_pose" in key:
                        data_dict["states.robot_to_env_pose"] = value[step]
                    else:
                        data_dict[key] = value[step]
                data_dict["task"] = meta_info['language_instruction']
                frames.append(data_dict)
                # import pdb; pdb.set_trace()
                for image_key in image_keys:
                    image_key_step_encode = f"{image_key}/{step_str}".encode()
                    if not image_key_step_encode in keys:
                        raise ValueError(f"Image key {image_key_step_encode} not found in LMDB keys.")
                    image_observations[image_key].append(load_image(txn, image_key_step_encode))
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"load image_observations of {episode_path}")
        env.close()
        if not frames:
            return None
        os.makedirs(sava_path, exist_ok=True)
        os.makedirs(sava_path/episode_path.name, exist_ok=True)
        video_paths = {}
        for image_key in image_keys:
            h_ori, w_ori =  image_observations[image_key][0].shape[:2]
            if w_ori == 1280:
                w_tgt = w_ori//2
                h_tgt = h_ori//2
            else:
                w_tgt = w_ori
                h_tgt = h_ori
            imageio.mimsave(
                sava_path/episode_path.name/f'{image_key.replace(".", "_")}.mp4', 
                image_observations[image_key], 
                fps=target_fps,
                codec="libsvtav1",
                # codec="libx264",
                ffmpeg_params=[
                    "-crf", "28",            # 画质控制（0-63，默认30）
                    "-preset", "8",          # 速度预设（0-13，值越高越快但压缩率越低）
                    # "-g", "240",             # 关键帧间隔（建议 ≥ fps 的 8 倍）
                    "-pix_fmt", "yuv420p",   # 兼容性像素格式
                    "-movflags", "+faststart", # 将元数据移到文件开头，便于网络播放
                    # "-threads", "8",           # 使用的线程数
                    "-vf", f"scale={w_tgt}:{h_tgt}",
                    "-y",                      # 覆盖已存在的输出文件
                ]
            )
            video_paths[image_key] = sava_path/episode_path.name/f'{image_key.replace(".", "_")}.mp4'
        print(f"imageio.mimsave time taken of {episode_path}")

        return {
            "frames": frames,
            "videos": video_paths,
        }

    except Exception as e:
        logging.error(f"Failed to load or process LMDB data: {e}")
        return None


def get_all_tasks(src_path: Path, output_path: Path) -> Tuple[Path, Path]:
    output_path.mkdir(exist_ok=True)
    yield (src_path, output_path)

def compute_episode_stats(episode_data: Dict[str, List[str] | np.ndarray], features: Dict) -> Dict:
    ep_stats = {}
    for key, data in episode_data.items():
        if features[key]["dtype"] == "string":
            continue
        elif features[key]["dtype"] in ["image", "video"]:
            ep_ft_array = sample_images(data)
            axes_to_reduce = (0, 2, 3)  # keep channel dim
            keepdims = True
        else:
            ep_ft_array = data  # data is already a np.ndarray
            axes_to_reduce = 0  # compute stats over the first axis
            keepdims = data.ndim == 1  # keep as np.array

        ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)
        if features[key]["dtype"] in ["image", "video"]:
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in ep_stats[key].items()
            }
    return ep_stats

def sample_images(input):
    if type(input) is str:
        video_path = input
        reader = torchvision.io.VideoReader(video_path, stream="video")
        frames = [frame["data"] for frame in reader]
        frames_array = torch.stack(frames).numpy()  # Shape: [T, C, H, W]
        sampled_indices = sample_indices(len(frames_array))
        images = None
        for i, idx in enumerate(sampled_indices):
            img = frames_array[idx]
            img = auto_downsample_height_width(img)
            if images is None:
                images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)
            images[i] = img
    elif type(input) is np.ndarray:
        frames_array = input[:, None, :, :]  # Shape: [T, C, H, W]
        sampled_indices = sample_indices(len(frames_array))
        images = None
        for i, idx in enumerate(sampled_indices):
            img = frames_array[idx]
            img = auto_downsample_height_width(img)
            if images is None:
                images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)
            images[i] = img
    return images


def load_local_dataset(episode_path: str, save_path:str, origin_fps=30, target_fps=30):
    fps_factor = origin_fps // target_fps
    # print(f"fps downsample factor: {fps_factor}")
    # logging.info(f"fps downsample factor: {fps_factor}")
    # for format_str in [f"{episode_id:07d}", f"{episode_id:06d}", str(episode_id)]:
    #     episode_path = Path(src_path) / format_str
    #     save_path = Path(save_path) / format_str
    #     if episode_path.exists():
    #         break
    # else:
    #     logging.warning(f"Episode directory not found for ID {episode_id}")
    #     return None, None
    episode_path = Path(episode_path)
    if not episode_path.exists():
        logging.warning(f"{episode_path} does not exist")
        return None, None
        
    if not (episode_path / "lmdb/data.mdb").exists():
        logging.warning(f"LMDB data not found for episode {episode_path}")
        return None, None
    
    raw_dataset = load_lmdb_data(episode_path, save_path, fps_factor, target_fps)
    if raw_dataset is None:
        return None, None
    frames = raw_dataset["frames"] # states, actions, task
    videos = raw_dataset["videos"] # image paths
    ## check the frames
    for camera_name, video_path in videos.items():
        if not os.path.exists(video_path):
            logging.error(f"Video file {video_path} does not exist.")
            print(f"Camera {camera_name} Video file {video_path} does not exist.")
            return None, None
    return frames, videos


def save_as_lerobot_dataset(task: tuple[Path, Path], repo_id, num_threads, debug, origin_fps=30,  target_fps=30, num_demos=None, robot_type="Franka", delete_downsampled_videos=True):
    src_path, save_path = task
    print(f"**Processing collected** {src_path}")
    print(f"**saving to** {save_path}")
    if save_path.exists():
        print(f"Output directory {save_path} already exists. Deleting it.")
        logging.warning(f"Output directory {save_path} already exists. Deleting it.")
        shutil.rmtree(save_path)
        # print(f"Output directory {save_path} already exists.")
        # return 

    dataset = FrankaDataset.create(
        repo_id=f"{repo_id}",
        root=save_path,
        fps=target_fps,
        robot_type=robot_type,
        features=FEATURES,
    )
    all_episode_paths = sorted([f.as_posix() for f in src_path.glob(f"*") if f.is_dir()])
    if num_demos is not None:
        all_episode_paths = all_episode_paths[:num_demos]
    # all_subdir_eids = [int(Path(path).name) for path in all_subdir]
    if debug:
        for i in range(1):
            frames, videos = load_local_dataset(episode_path=all_episode_paths[i], save_path=save_path, origin_fps=origin_fps, target_fps=target_fps)
            if frames is None or videos is None:
                print(f"Skipping episode {all_episode_paths[i]} due to missing data.")
                continue
            for frame_data in frames:
                dataset.add_frame(frame_data)
            dataset.save_episode(videos=videos)
            if delete_downsampled_videos:
                for _, video_path in videos.items():
                    parent_dir = os.path.dirname(video_path)
                    try:
                        shutil.rmtree(parent_dir)
                        # os.remove(video_path)
                        # print(f"Successfully deleted: {parent_dir}")
                        print(f"Successfully deleted: {video_path}")
                    except Exception as e:
                        pass  # Handle the case where the directory might not exist or is already deleted

    else:
        counter_episodes_uncomplete = 0
        for batch_index in range(len(all_episode_paths)//num_threads+1):
            batch_episode_paths = all_episode_paths[batch_index*num_threads:(batch_index+1)*num_threads]
            if len(batch_episode_paths) == 0:
                continue
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                for episode_path in batch_episode_paths:
                    print("starting to process episode: ", episode_path)
                    futures.append(
                        executor.submit(load_local_dataset, episode_path=episode_path, save_path=save_path, origin_fps=origin_fps, target_fps=target_fps)
                    )
                for raw_dataset in as_completed(futures):
                    frames, videos = raw_dataset.result()
                    if frames is None or videos is None:
                        counter_episodes_uncomplete += 1
                        print(f"Skipping episode {episode_path} due to missing data.")
                        continue
                    for frame_data in frames:
                        dataset.add_frame(frame_data)
                    dataset.save_episode(videos=videos)
                    gc.collect()
                    print(f"finishing processed {videos}")
                    if delete_downsampled_videos:
                        for _, video_path in videos.items():
                            # Get the parent directory of the video
                            parent_dir = os.path.dirname(video_path)
                            try:
                                shutil.rmtree(parent_dir)
                                print(f"Successfully deleted: {parent_dir}")
                            except Exception as e:
                                pass
        print("counter_episodes_uncomplete:", counter_episodes_uncomplete)

def main(src_path, save_path, repo_id, num_threads=4, debug=False, origin_fps=30, target_fps=30, num_demos=None):
    logging.info("Scanning for episodes...")
    tasks = get_all_tasks(src_path, save_path)
    if debug:
        task = next(tasks)
        save_as_lerobot_dataset(task, repo_id, num_threads=num_threads, debug=debug, origin_fps=origin_fps, target_fps=target_fps, num_demos=num_demos)
    else:
        for task in tasks:
            save_as_lerobot_dataset(task, repo_id, num_threads=num_threads, debug=debug, origin_fps=origin_fps, target_fps=target_fps, num_demos=num_demos)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert collected data from Piper to Lerobot format.")
    parser.add_argument(
        "--src_path",
        type=str,
        # required=False,
        default="/fs-computility/efm/shared/datasets/myData-A1/real/raw_data/agilex_split_aloha/",
        help="Path to the input file containing collected data in Piper format.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        # required=False,
        default="/fs-computility/efm/shared/datasets/myData-A1/real/lerobot_v2_1/agilex_split_aloha/",
        help="Path to the output file where the converted Lerobot format will be saved.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with limited episodes",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=64,
        help="Number of threads per process",
    )
    # parser.add_argument(
    #     "--task_name",
    #     type=str,
    #     required=True,
    #     default="Pick_up_the_marker_and_put_it_into_the_pen_holder",
    #     help="Name of the task to be processed. Default is 'Pick_up_the_marker_and_put_it_into_the_pen_holder'.",
    # )
    parser.add_argument(
        "--repo_id",
        type=str,
        # required=True,
        default="SplitAloha_20250714",
        help="identifier for the dataset repository.",
    )
    parser.add_argument(
        "--origin_fps",
        type=int,
        default=30,
        help="Frames per second for the obervation video. Default is 30.",
    )
    parser.add_argument(
        "--target_fps",
        type=int,
        default=30,
        help="Frames per second for the downsample video. Default is 30.",
    )
    parser.add_argument(
        "--num_demos",
        type=int,
        default=None,
        help="Demos need to transfer"
    )
    args = parser.parse_args()
    assert int(args.origin_fps) % int(args.target_fps) == 0, "origin_fps must be an integer multiple of target_fps"
    start_time = time.time()
    main(
        src_path=Path(args.src_path),
        save_path=Path(args.save_path),
        repo_id=args.repo_id,
        num_threads=args.num_threads,
        debug=args.debug,
        origin_fps=args.origin_fps,
        target_fps=args.target_fps,
        num_demos=args.num_demos,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time taken: {elapsed_time:.2f} seconds")
