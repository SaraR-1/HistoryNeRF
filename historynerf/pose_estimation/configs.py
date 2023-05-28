from dataclasses import dataclass
from typing import Type, Union
from omegaconf import MISSING


@dataclass
class PoseEstimationConfig:
    image_dir: str = "/Users/sr572/Documents/ScienceMuseum/Datasets/tory_trevi_fountain/images"
    output_dir: str = "/Users/sr572/Documents/ScienceMuseum/Datasets/tory_trevi_fountain/output"
    database_path: str = "database.db"

@dataclass
class COLMAPConfig(PoseEstimationConfig):
    camera_model: str = "SIMPLE_RADIAL"
    use_gpu: float = 1
    matching_method: str = "exhaustive"

@dataclass
class OpenMVGConfig(PoseEstimationConfig):
    camera_model: str = "TEST"

@dataclass
class Config:
    pose_config: PoseEstimationConfig = MISSING

