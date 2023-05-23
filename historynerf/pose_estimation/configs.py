from dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class PoseEstimationConfig:
    image_dir: str = MISSING
    output_dir: str = MISSING
    database_path: str = MISSING

@dataclass
class COLMAPConfig(PoseEstimationConfig):
    camera_model: str = "SIMPLE_RADIAL"
    use_gpu: float = 1
    matching_method: str = "exhaustive"

@dataclass
class Config:
    pose_estimation: PoseEstimationConfig = MISSING

