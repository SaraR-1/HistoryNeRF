from dataclasses import dataclass
from typing import Optional
from omegaconf import MISSING


@dataclass
class PoseEstimationConfig:
    image_dir: str = "/srv/galene0/sr572/palace_of_westminster/dense/images"
    output_dir: str = "/srv/galene0/sr572/palace_of_westminster/dense/output"
    database_path: str = "database.db"

@dataclass
class COLMAPConfig(PoseEstimationConfig):
    camera_model: str = "SIMPLE_RADIAL"
    use_gpu: str = 7
    matching_method: str = "exhaustive"
    seed: int = 0
    sample_size: Optional[int] = None

@dataclass
class OpenMVGConfig(PoseEstimationConfig):
    camera_model: str = "TEST"

@dataclass
class Config:
    pose_config: PoseEstimationConfig = MISSING

