import numpy as np
import omni.replicator.core as rep
from core.cameras.base_camera import register_camera
from core.utils.camera_utils import get_src
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.transformations import (
    get_relative_transform,
    pose_from_tf_matrix,
)
from omni.isaac.sensor import Camera


@register_camera
class CustomCamera(Camera):
    """Generic pinhole RGB-D camera used in simbox tasks."""

    def __init__(self, cfg, prim_path, root_prim_path, reference_path, name, *args, **kwargs):
        """
        Args:
            cfg: Config dict with required keys:
                - params: Dict containing:
                    - pixel_size: Pixel size in microns
                    - f_number: F-number
                    - focus_distance: Focus distance in meters
                    - camera_params: [fx, fy, cx, cy] camera intrinsics
                    - resolution_width: Image width
                    - resolution_height: Image height
                - output_mode (optional): "rgb" or "diffuse_albedo"
            prim_path: Camera prim path in USD stage
            root_prim_path: Root prim path in USD stage
            reference_path: Reference prim path for camera mounting
            name: Camera name
        """
        # ===== Initialize camera =====
        super().__init__(
            prim_path=prim_path,
            name=name,
            resolution=(cfg["params"]["resolution_width"], cfg["params"]["resolution_height"]),
            *args,
            **kwargs,
        )
        self.initialize()
        self.with_distance = cfg["params"].get("with_distance", True)
        self.with_semantic = cfg["params"].get("with_semantic", False)
        self.with_bbox2d = cfg["params"].get("with_bbox2d", False)
        self.with_bbox3d = cfg["params"].get("with_bbox3d", False)
        # Motion vectors are high-volume outputs; keep default off unless explicitly enabled in config.
        self.with_motion_vector = cfg["params"].get("with_motion_vector", False)
        self.with_depth = cfg["params"].get("depth", False)

        if self.with_distance:
            self.add_distance_to_image_plane_to_frame()
        if self.with_semantic:
            self.add_semantic_segmentation_to_frame()
        if self.with_bbox2d:
            self.add_bounding_box_2d_tight_to_frame()
            self.add_bounding_box_2d_loose_to_frame()
        if self.with_bbox3d:
            self.add_bounding_box_3d_to_frame()
        if self.with_motion_vector:
            self.add_motion_vectors_to_frame()

        # ===== From cfg =====
        pixel_size = cfg["params"].get("pixel_size")
        f_number = cfg["params"].get("f_number")
        focus_distance = cfg["params"].get("focus_distance")
        fx, fy, cx, cy = cfg["params"].get("camera_params")
        width = cfg["params"].get("resolution_width")
        height = cfg["params"].get("resolution_height")
        self.output_mode = cfg.get("output_mode", "rgb")

        # ===== Compute and set camera parameters =====
        horizontal_aperture = pixel_size * 1e-3 * width
        vertical_aperture = pixel_size * 1e-3 * height
        focal_length_x = fx * pixel_size * 1e-3
        focal_length_y = fy * pixel_size * 1e-3
        focal_length = (focal_length_x + focal_length_y) / 2

        self.set_focal_length(focal_length / 10.0)
        self.set_focus_distance(focus_distance)
        self.set_lens_aperture(f_number * 100.0)
        self.set_horizontal_aperture(horizontal_aperture / 10.0)
        self.set_vertical_aperture(vertical_aperture / 10.0)
        self.set_clipping_range(0.05, 1.0e5)
        self.set_projection_type("pinhole")

        fx = width * self.get_focal_length() / self.get_horizontal_aperture()
        fy = height * self.get_focal_length() / self.get_vertical_aperture()
        self.is_camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

        self.reference_path = reference_path
        self.root_prim_path = root_prim_path
        self.parent_camera_prim_path = str(self.prim.GetParent().GetPath())
        self.parent_camera_xform = XFormPrim(self.parent_camera_prim_path)

        if self.output_mode == "diffuse_albedo":
            self.add_diffuse_albedo_to_frame()

    def add_diffuse_albedo_to_frame(self) -> None:
        """Attach the diffuse_albedo annotator to this camera."""
        if "DiffuseAlbedo" not in self._custom_annotators:
            self._custom_annotators["DiffuseAlbedo"] = rep.AnnotatorRegistry.get_annotator("DiffuseAlbedo")
            self._custom_annotators["DiffuseAlbedo"].attach([self._render_product_path])
        self._current_frame["DiffuseAlbedo"] = None

    def remove_diffuse_albedo_from_frame(self) -> None:
        if self._custom_annotators["DiffuseAlbedo"] is not None:
            self._custom_annotators["DiffuseAlbedo"].detach([self._render_product_path])
            self._custom_annotators["DiffuseAlbedo"] = None
        self._current_frame.pop("DiffuseAlbedo", None)

    def add_specular_albedo_to_frame(self) -> None:
        """Attach the specular_albedo annotator to this camera."""
        if self._custom_annotators["SpecularAlbedo"] is None:
            self._custom_annotators["SpecularAlbedo"] = rep.AnnotatorRegistry.get_annotator("SpecularAlbedo")
            self._custom_annotators["SpecularAlbedo"].attach([self._render_product_path])
        self._current_frame["SpecularAlbedo"] = None

    def remove_specular_albedo_from_frame(self) -> None:
        if self._custom_annotators["SpecularAlbedo"] is not None:
            self._custom_annotators["SpecularAlbedo"].detach([self._render_product_path])
            self._custom_annotators["SpecularAlbedo"] = None
        self._current_frame.pop("SpecularAlbedo", None)

    def add_direct_diffuse_to_frame(self) -> None:
        """Attach the direct_diffuse annotator to this camera."""
        if self._custom_annotators["DirectDiffuse"] is None:
            self._custom_annotators["DirectDiffuse"] = rep.AnnotatorRegistry.get_annotator("DirectDiffuse")
            self._custom_annotators["DirectDiffuse"].attach([self._render_product_path])
        self._current_frame["DirectDiffuse"] = None

    def remove_direct_diffuse_from_frame(self) -> None:
        if self._custom_annotators["DirectDiffuse"] is not None:
            self._custom_annotators["DirectDiffuse"].detach([self._render_product_path])
            self._custom_annotators["DirectDiffuse"] = None
        self._current_frame.pop("DirectDiffuse", None)

    def add_indirect_diffuse_to_frame(self) -> None:
        """Attach the indirect_diffuse annotator to this camera."""
        if self._custom_annotators["IndirectDiffuse"] is None:
            self._custom_annotators["IndirectDiffuse"] = rep.AnnotatorRegistry.get_annotator("IndirectDiffuse")
            self._custom_annotators["IndirectDiffuse"].attach([self._render_product_path])
        self._current_frame["IndirectDiffuse"] = None

    def remove_indirect_diffuse_from_frame(self) -> None:
        if self._custom_annotators["IndirectDiffuse"] is not None:
            self._custom_annotators["IndirectDiffuse"].detach([self._render_product_path])
            self._custom_annotators["IndirectDiffuse"] = None
        self._current_frame.pop("IndirectDiffuse", None)

    def get_observations(self):
        if self.reference_path:
            camera_mount2env_pose = get_relative_transform(
                get_prim_at_path(self.reference_path), get_prim_at_path(self.root_prim_path)
            )
            camera_mount2env_pose = pose_from_tf_matrix(camera_mount2env_pose)
            self.parent_camera_xform.set_local_pose(
                translation=camera_mount2env_pose[0],
                orientation=camera_mount2env_pose[1],
            )
        camera2env_pose = get_relative_transform(
            get_prim_at_path(self.prim_path), get_prim_at_path(self.root_prim_path)
        )

        if self.output_mode == "rgb":
            color_image = self.get_rgba()[..., :3]
        elif self.output_mode == "diffuse_albedo":
            color_image = self._custom_annotators["DiffuseAlbedo"].get_data()[..., :3]
        else:
            raise NotImplementedError

        obs = {
            "color_image": color_image,
            "camera2env_pose": camera2env_pose,
            "camera_params": self.is_camera_matrix.tolist(),
        }
        if self.with_depth:
            obs["depth_image"] = get_src(self, "depth"),

        seg_data = get_src(self, "seg")
        if seg_data is not None:
            obs["semantic_mask"] = seg_data["mask"]
            obs["semantic_mask_id2labels"] = seg_data["id2labels"]

        bbox2d_tight = get_src(self, "bbox2d_tight")
        if bbox2d_tight is not None:
            obs["bbox2d_tight"], obs["bbox2d_tight_id2labels"] = bbox2d_tight
        bbox2d_loose = get_src(self, "bbox2d_loose")
        if bbox2d_loose is not None:
            obs["bbox2d_loose"], obs["bbox2d_loose_id2labels"] = bbox2d_loose
        bbox3d = get_src(self, "bbox3d")
        if bbox3d is not None:
            obs["bbox3d"], obs["bbox3d_id2labels"] = bbox3d
        motion_vectors = get_src(self, "motion_vectors")
        if motion_vectors is not None:
            obs["motion_vectors"] = motion_vectors
        return obs
