from dataclasses import dataclass
from typing import Optional, List
from omegaconf import MISSING


@dataclass
class PoseEstimationConfig:
    image_dir: str = "/srv/galene0/sr572/palace_of_westminster/dense/images"
    output_dir: str = "/srv/galene0/sr572/palace_of_westminster/dense/output_nsamples"
    database_path: str = "database.db"

@dataclass
class COLMAPConfig(PoseEstimationConfig):
    camera_model: str = "SIMPLE_RADIAL"
    use_gpu: str = 7
    matching_method: str = "exhaustive"
    # For evaluation purposes, path to the ground truth poses
    gt_poses_dir: Optional[str] = None
    angular_error: Optional[int] = None
    translation_error: Optional[float] = None
    # TODO: move them to have a data preprocessing separate config
    seed: int = 0
    sample_size: Optional[int] = None
    image_list: Optional[List[str]] = None
    video_sample_step: Optional[int] = None

@dataclass
class OpenMVGConfig(PoseEstimationConfig):
    camera_model: str = "TEST"

@dataclass
class DataConfig:
    seed: int = 0
    sample_size: Optional[int] = None
    image_list: Optional[List[str]] = None
    video_sample_step: Optional[int] = None

@dataclass
class Config:
    pose_config: PoseEstimationConfig = MISSING
    # data_config: DataConfig = MISSING
    wandb_project: str = MISSING
    wandb_log: bool = True


