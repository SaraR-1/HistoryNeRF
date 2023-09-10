from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple
from omegaconf import MISSING
from historynerf.evaluation.alignment import NormTypes, KeypointDetectorProtocol 

@dataclass
class SamplingConfig:
    """Configurations for sampling images.
    
    Args:
        rnd_seed: Random seed for sampling.
        sample_size: Number of samples to take.
        image_list: List of specific image names to sample.
        sequential_sample_step: Step between sequential samples.
        sequential_sample_start: Starting index for sequential sampling.
    """
    rnd_seed: Optional[int] = None
    sample_size: Optional[int] = None
    image_list: Optional[List[str]] = None
    sequential_sample_step: Optional[int] = None
    sequential_sample_start: Optional[int] = 0

@dataclass
class DataPreparationConfig:
    """Configurations for data preparation.
    
    Args:
        input_dir: Path to the folder containing images or an mp4 video.
        output_dir: Output directory.
        overwrite_output: Whether to overwrite existing output.
        resize: Dimensions to resize images to.
        sampling: Sampling configurations.
    """
    input_dir: str
    output_dir: str
    overwrite_output: bool = False
    resize: Optional[List[int]] = None
    sampling: SamplingConfig = MISSING

@dataclass
class PoseEstimationConfig:
    """Configurations for Pose Estimation.
    
    Args:
        sfm_tool: Structure from Motion tool to use.
        camera_type: Type of camera.
        matching_method (Optional[str]): Feature matching method.
        # ...
        
    """
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
class NeRFCameraOptimizerConfig:
    """Configurations for NeRF Camera Optimizer.
    
    Args:
        mode (Optional[str]): Optimization strategy.
    """
    mode: Optional[str] = MISSING # {off,SO3xR3,SE3} Pose optimization strategy to use. If enabled, we recommend SO3xR3.
                         
@dataclass
class NeRFDataManagerConfig:
    """Configurations for NeRF Data Manager.
    
    Args:
        train_num_rays_per_batch (Optional[int]): Number of rays per training batch.
        # ... 
    """
    train_num_rays_per_batch: Optional[int] = MISSING # Number of rays per batch to use per training iteration. (default: 4096)
    train_num_images_to_sample_from: Optional[int] = MISSING # Number of images to sample during training iteration. (default: -1, i.e. all images)
    eval_num_rays_per_batch: Optional[int] = MISSING # Number of rays per batch to use per eval iteration. (default: 4096)
    eval_num_images_to_sample_from: Optional[int] = MISSING # Number of images to sample during eval iteration. (default: -1, i.e. all images)
    camera_optimizer: NeRFCameraOptimizerConfig = MISSING

@dataclass
class NeRFModelConfig:
    """Configurations for the NeRF Model.
    
    Args:
        use_gradient_scaling (Optional[bool]): Whether to scale gradients by ray distance.
    """
    use_gradient_scaling: Optional[bool] = MISSING # Scale gradients by the ray distance to the pixel as suggested in Radiance Field Gradient Scaling for Unbiased Near-Camera Training paper (default: False)

@dataclass
class NeRFPipelineConfig:
    """Configurations for the NeRF Pipeline.
    
    Args:
        datamanager (NeRFDataManagerConfig): Data manager configurations.
        model (NeRFModelConfig): Model configurations.
    """
    datamanager: NeRFDataManagerConfig
    model: NeRFModelConfig

@dataclass
class MachineConfig:
    """Machine specific configurations.
    
    Args:
        num_gpus (Optional[int]): Number of GPUs to use.
    """
    num_gpus: Optional[int] = MISSING # Number of GPUs to use. (default: 1)

@dataclass
class NeRFConfig:
    """Main configurations for NeRF.
    
    Args:
        method_name (str): Name of the NeRF method to use.
        # ... 
    """
    method_name: str = MISSING
    dataparser_name: Optional[str] = MISSING # default nerfstudio-data
    train_split_fraction: Optional[float] = MISSING # The fraction of images to use for training. The remaining images are for eval. (default: 0.9)
    vis: Optional[str] = MISSING # {viewer,wandb,tensorboard,viewer+wandb,viewer+tensorboard} (default: viewer)
    steps_per_save: Optional[int] = MISSING # Number of steps between saves. (default: 2000)
    steps_per_eval_batch: Optional[int] = MISSING # Number of steps between randomly sampled batches of rays. (default: 500)
    steps_per_eval_image: Optional[int] = MISSING # Number of steps between single eval images. (default: 500)
    steps_per_eval_all_images: Optional[int] = MISSING # Number of steps between all eval images. (default: 25000)
    max_num_iterations: Optional[int] = MISSING # Maximum number of iterations. (default: 30000)

    disable_scene_scale: Optional[bool] = MISSING

    pipeline: NeRFPipelineConfig = MISSING
    machine: MachineConfig = MISSING

    def __post_init__(self) -> None:
        allowed_methods = ("depth-nerfacto" ,"dnerf", "instant-ngp", "instant-ngp-bounded", "mipnerf", "nerfacto", "nerfacto-big", 
        "nerfplayer-nerfacto", "nerfplayer-ngp", "neus", "neus-facto", "vanilla-nerf", "volinga", "in2n", "in2n-small", "in2n-tiny", 
        "kplanes", "kplanes-dynamic", "lerf", "lerf-big", "lerf-lite", "tetra-nerf", "tetra-nerf-original")
        error_message = f"Wrong NeRF method name. Allowed methods: {allowed_methods}"
        assert self.method_name in allowed_methods, error_message
        
@dataclass
class AlignmentEvaluationConfig:
    """Configurations for alignment evaluation.
    
    Args:
        flag (Optional[bool]): Whether to perform alignment evaluation.
        # ... 
    """
    flag: Optional[bool] = MISSING
    keypoint_detector: Optional[str] = MISSING
    matcher_distance: Optional[str] = MISSING
    match_filter: Optional[float] = MISSING
    matched_keypoints_threshold: Optional[int]= MISSING

@dataclass
class EvaluationConfig:
    """Configurations for evaluation.
    
    Args:
        camera_pose_path_train (Optional[str]): Path to camera poses for train set.
        # ... (Continue with other fields)
    """
    camera_pose_path_train: Optional[str] = MISSING # Gold Standard camera poses file of the train set, transforms.json
    camera_pose_path_test: Optional[str] = MISSING # Gold Standard camera poses file of the test set, transforms.json
    gt_images_dir: Optional[str] = MISSING # test folder
    config_path: Optional[str] = MISSING # only used when running run_evaluation, already provided when running the entire pipeline
    output_dir: Optional[str] = MISSING # only used when running run_evaluation, already provided when running the entire pipeline
    
    alignment: AlignmentEvaluationConfig = MISSING

@dataclass
class Config:
    """Main configuration class for run.py.
    
    Args:
        data_preparation (DataPreparationConfig): Data preparation configurations.
        pose_estimation (PoseEstimationConfig): Pose estimation configurations.
        nerf (NeRFConfig): NeRF configurations.
        evaluation (EvaluationConfig): Evaluation configurations.
        # ... (Continue with other fields)
    """
    data_preparation: DataPreparationConfig = MISSING
    pose_estimation: PoseEstimationConfig = MISSING
    nerf: NeRFConfig = MISSING
    evaluation: EvaluationConfig = MISSING
    wandb_entity: Optional[str] = MISSING
    wandb_project: Optional[str] = MISSING
    wandb_log: bool = True


@dataclass
class SplitDataConfig:
    """Main configuration class for run_splitdata.py"""
    camera_path: Path = MISSING
    n: int = MISSING
    images_dir: Path = MISSING
    output_dir: Path = MISSING


@dataclass
class GoldStandardConfig:
    """Main configuration class for run_goldstandard.py"""
    input_dir: str = MISSING
    output_dir: str = MISSING
    pose_estimation: PoseEstimationConfig = MISSING