import numpy as np
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.transformations import get_relative_transform
from omni.isaac.sensor import Camera


def _get_annotator(camera: Camera, annotator_name: str):
    custom_annotators = getattr(camera, "_custom_annotators", None)
    if not isinstance(custom_annotators, dict):
        return None
    return custom_annotators.get(annotator_name)


def _get_frame(frame):
    if isinstance(frame, np.ndarray) and frame.size > 0:
        return frame[:, :, :3]
    return None


def _get_depth(depth):
    if isinstance(depth, np.ndarray) and depth.size > 0:
        return depth
    return None


def _get_rgb_image(camera: Camera):
    output_mode = getattr(camera, "output_mode", "rgb")
    if output_mode == "rgb":
        return _get_frame(camera.get_rgba())
    if output_mode == "diffuse_albedo":
        annotator = _get_annotator(camera, "DiffuseAlbedo")
        if annotator is None:
            return None
        return _get_frame(annotator.get_data())
    raise NotImplementedError(f"Unsupported output mode: {output_mode}")


def _get_depth_image(camera: Camera):
    annotator = _get_annotator(camera, "distance_to_image_plane")
    if annotator is None:
        return None
    return _get_depth(annotator.get_data())


def _get_object_mask(camera: Camera):
    annotator = _get_annotator(camera, "semantic_segmentation")
    if annotator is None:
        return None
    annotation_data = annotator.get_data()
    if (
        not isinstance(annotation_data, dict)
        or "data" not in annotation_data
        or "info" not in annotation_data
    ):
        return None
    info = annotation_data["info"]
    if not isinstance(info, dict) or "idToLabels" not in info:
        return None
    mask = annotation_data["data"]
    if isinstance(mask, np.ndarray) and mask.size > 0:
        return {"mask": mask, "id2labels": info["idToLabels"]}
    return None


def _get_bbox(camera: Camera, bbox_type: str):
    annotator = _get_annotator(camera, bbox_type)
    if annotator is None:
        return None
    annotation_data = annotator.get_data()
    if (
        not isinstance(annotation_data, dict)
        or "data" not in annotation_data
        or "info" not in annotation_data
    ):
        return None
    info = annotation_data["info"]
    if not isinstance(info, dict) or "idToLabels" not in info:
        return None
    return annotation_data["data"], info["idToLabels"]


def _get_motion_vectors(camera: Camera):
    annotator = _get_annotator(camera, "motion_vectors")
    if annotator is None:
        return None
    annotation_data = annotator.get_data()
    if isinstance(annotation_data, np.ndarray) and annotation_data.size > 0:
        return annotation_data
    return None


def _get_camera2env_pose(camera: Camera):
    prim_path = getattr(camera, "prim_path", None)
    root_prim_path = getattr(camera, "root_prim_path", None)
    if not prim_path or not root_prim_path:
        return None
    return get_relative_transform(get_prim_at_path(prim_path), get_prim_at_path(root_prim_path))


def _get_camera_params(camera: Camera):
    camera_matrix = getattr(camera, "is_camera_matrix", None)
    if camera_matrix is None:
        return None
    if isinstance(camera_matrix, np.ndarray):
        return camera_matrix.tolist()
    return camera_matrix


def get_src(camera: Camera, data_type: str):
    if data_type == "rgb":
        return _get_rgb_image(camera)
    if data_type == "depth":
        return _get_depth_image(camera)
    if data_type == "seg":
        return _get_object_mask(camera)
    if data_type == "bbox2d_tight":
        return _get_bbox(camera, "bounding_box_2d_tight")
    if data_type == "bbox2d_loose":
        return _get_bbox(camera, "bounding_box_2d_loose")
    if data_type == "bbox3d":
        return _get_bbox(camera, "bounding_box_3d")
    if data_type == "motion_vectors":
        return _get_motion_vectors(camera)
    if data_type == "camera2env_pose":
        return _get_camera2env_pose(camera)
    if data_type == "camera_params":
        return _get_camera_params(camera)
    raise NotImplementedError(f"Unsupported source type: {data_type}")
