from dataclasses import dataclass
from typing import Optional, List
from omegaconf import MISSING

@dataclass
class SamplingConfig:
    rnd_seed: Optional[int] = None
    sample_size: Optional[int] = None
    image_list: Optional[List[str]] = None
    video_sample_step: Optional[int] = None

@dataclass
class NoiseConfig:
    pass

@dataclass
class PreprocessingConfig:
    input_dir: str
    output_dir: str
    overwrite_output: bool = False
    sampling: SamplingConfig = MISSING
    noise: NoiseConfig = MISSING

@dataclass
class COLMAPConfig:
    pass

@dataclass
class NeRFConfig:
    pass

@dataclass
class EvaluationConfig:
    pass

@dataclass
class Config:
    preprocessing: PreprocessingConfig = MISSING
    colmap: COLMAPConfig = MISSING
    nerf: NeRFConfig = MISSING
    evaluation: EvaluationConfig = MISSING
    wandb_project: str = MISSING
    wandb_log: bool = True