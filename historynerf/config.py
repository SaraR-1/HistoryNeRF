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
class DataPreparationConfig:
    """
    Add noise and undersample

    Args:
        input_dir: Either path to a folder containing images or path to an mp4 video
    """
    input_dir: str
    output_dir: str
    overwrite_output: bool = False
    sampling: SamplingConfig = MISSING
    # noise: NoiseConfig = MISSING

@dataclass
class PoseEstimationConfig:
    sfm_tool: str = MISSING # {any,colmap,hloc}  Colmap will use sift features, hloc can use many modern methods such as superpoint features and superglue matcher (default: any) 
    camera_type: Optional[str] = MISSING # {perspective,fisheye,equirectangular} - default: perspective
    matching_method: Optional[str] = MISSING # Vocab tree is recommended for a balance of speed and accuracy. Exhaustive is slower but more accurate. Sequential is faster but should only be used for videos. (default: vocab_tree) 
    feature_type: Optional[str] = MISSING # {any,sift,superpoint,superpoint_aachen,superpoint_max,superpoint_inloc,r2d2,d2net-ss,sosnet,disk} (default: any) 
    matcher_type: Optional[str] = MISSING # {any,NN,superglue,superglue-fast,NN-superpoint,NN-ratio,NN-mutual,adalam} Matching algorithm. (default: any)
    num_downscales: Optional[int] = MISSING   # Number of times to downscale the images. Downscales by 2 each time. For example a value of 3 will downscale the images by 2x, 4x, and 8x. (default: 3) 
    skip_colmap_flag: Optional[bool] = MISSING # If True, skips COLMAP and generates transforms.json if possible. (default: False)     
    skip_image_processing_flag: Optional[bool] = MISSING # If True, skips copying and downscaling of images and only runs COLMAP if possible and enabled (default: False) 
    colmap_model_path: Optional[str] = MISSING # Optionally sets the path of the colmap model. Used only when --skip-colmap is set to True. The path is relative to the output directory. (default: colmap/sparse/0) 
    gpu_flag: Optional[bool] = MISSING # If True, use GPU. (default: True)                                                                                
    use_sfm_depth_flag: Optional[bool] = MISSING # If True, export and use depth maps induced from SfM points. (default: False)
    include_depth_debug_flag: Optional[bool] = MISSING # If --use-sfm-depth and this flag is True, also export debug images showing Sf overlaid upon input images. (default: False) 

@dataclass
class NeRFConfig:
    method_name: str = MISSING
    vis: Optional[str] = MISSING # {viewer,wandb,tensorboard,viewer+wandb,viewer+tensorboard} (default: viewer)
    # steps-per-save: Optional[int] = MISSING # Number of steps between saves. (default: 2000)
    # steps-per-eval-batch: Optional[int] = MISSING # Number of steps between randomly sampled batches of rays. (default: 500)
    # steps-per-eval-image: Optional[int] = MISSING # Number of steps between single eval images. (default: 500)
    # steps-per-eval-all-images: Optional[int] = MISSING # Number of steps between all eval images. (default: 25000)
    # max-num-iterations: Optional[int] = MISSING # Maximum number of iterations. (default: 30000)

    # machine.num-gpus: Optional[int] = MISSING # Number of GPUs to use. (default: 1)
    # pipeline.datamanager.train-num-rays-per-batch # int Number of rays per batch to use per training iteration. (default: 4096)
    # pipeline.datamanager.train-num-images-to-sample-from # Number of images to sample during training iteration. (default: -1, i.e. all images)
    # pipeline.datamanager.eval-num-rays-per-batch # int Number of rays per batch to use per eval iteration. (default: 4096)
    # pipeline.datamanager.eval-num-images-to-sample-from # Number of images to sample during eval iteration. (default: -1, i.e. all images)


    # def __post_init__(self) -> None:
    #     allowed_methods = ("depth-nerfacto" ,"dnerf", "instant-ngp", "instant-ngp-bounded", "mipnerf", "nerfacto", "nerfacto-big", 
    #     "nerfplayer-nerfacto", "nerfplayer-ngp", "neus", "neus-facto", "vanilla-nerf", "volinga", "in2n", "in2n-small", "in2n-tiny", 
    #     "kplanes", "kplanes-dynamic", "lerf", "lerf-big", "lerf-lite", "tetra-nerf", "tetra-nerf-original")
    #     error_message = f"Wrong NeRF method name. Allowed methods: {allowed_methods}"
    #     assert self.method_name in allowed_methods, error_message

@dataclass
class EvaluationConfig:
    pass

@dataclass
class Config:
    data_preparation: DataPreparationConfig = MISSING
    pose_estimation: PoseEstimationConfig = MISSING
    nerf: NeRFConfig = MISSING
    # evaluation: EvaluationConfig = MISSING
    # project_name: str = MISSING # Wandb project name
    # experiment_name: str = MISSING # Wandb experiment name
    wandb_entity: Optional[str] = MISSING
    wandb_project: Optional[str] = MISSING
    wandb_log: bool = True
