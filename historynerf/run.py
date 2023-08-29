from coolname import generate_slug
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pathlib import Path
import wandb

from historynerf.config import Config, DataPreparationConfig, PoseEstimationConfig, NeRFConfig, EvaluationConfig, SamplingConfig
from historynerf.data_preparation import DataPreparation
from historynerf.colmap_load import ColmapLoader
from historynerf.nerfstudio_wrapper import NSWrapper
from historynerf.evaluation import NerfEvaluator, evaluate_compare_poses, evaluate_and_visualize_alignment


root_dir = Path(__file__).parents[1]

cs = ConfigStore.instance()
cs.store(name="base_parent", node=Config)
cs.store(group="data_preparation", name="base_data", node=DataPreparationConfig)
cs.store(group="data_preparation/sampling", name="base_sampling", node=SamplingConfig)
cs.store(group="pose_estimation", name="base_pose_estimation", node=PoseEstimationConfig)
cs.store(group="nerf", name="base_nerf", node=NeRFConfig)
cs.store(group="evaluation", name="base_evaluation", node=EvaluationConfig)


@hydra.main(config_path=str(root_dir / "configs"), config_name="parent", version_base="1.1")
def main(cfg: Config) -> None:
    cfg_obj = OmegaConf.to_object(cfg)
    # print(cfg_obj)
    # Randomly generate a name for the experiment
    experiment_name = generate_slug(2)
    
    # Simple example to run data_preparation, here only with undersampling
    data_obj = DataPreparation(cfg.data_preparation)
    data_obj.save_images()

    if cfg.pose_estimation.colmap_model_path:
        colmap_loader = ColmapLoader(
            recon_dir=Path(cfg.pose_estimation.colmap_model_path), 
            output_dir=Path(data_obj.config.output_dir).parent / "processed_data_fixedcolmap", 
            imgs_dir=Path(data_obj.config.input_dir) if data_obj.skip_save else Path(data_obj.config.output_dir))
        colmap_loader.sample_colmap()

        # Update cfg.colmap_model_path and data_obj.config.input_dir
        cfg.pose_estimation["colmap_model_path"], data_obj.config.input_dir = colmap_loader.update_path(current_colmap=cfg.pose_estimation["colmap_model_path"], current_input=data_obj.config.input_dir)
            
    nerf_obj = NSWrapper(
        input_dir=data_obj.config.input_dir if data_obj.skip_save else None,
        output_dir=data_obj.config.output_dir,
        pose_estimation_config=cfg.pose_estimation, 
        nerf_config=cfg.nerf,
        wandb_project=cfg.wandb_project,
        experiment_name=experiment_name,)
    nerf_obj.run()

    # Get the experiment id from the name
    api = wandb.Api()
    # breakpoint()
    experiment_id = api.runs(f"{cfg_obj.wandb_entity}/{cfg_obj.wandb_project}", filters={"config.experiment_name": experiment_name})[0].id
    print("Resume W&B.")
    # Add a flag to disable wandb from the config file
    if cfg_obj.wandb_log:
        wandb.init(
            project=cfg_obj.wandb_project,
            id=experiment_id,
            resume=True,
            config=OmegaConf.to_container(cfg, resolve=True, enum_to_str=True,),)
    else:
        wandb.init(project=cfg_obj.wandb_project, mode="disabled")

    # Log number of images used for training, read number of images in input_dir in cfg_obj
    wandb.log({"Training Sample Size": len(list(Path(nerf_obj.input_dir).glob("*.jpg")))})

    # Evaluate and log results
    output_camera_path = Path(nerf_obj.output_dir).parent / "processed_data" / "transforms.json"

    output_path_nerf = Path(nerf_obj.output_dir).parent / "nerf" / experiment_name / cfg_obj.nerf.method_name / "default"
    output_config_path = output_path_nerf / "config.yml"
    evaluation_output_dir = output_path_nerf / "evaluation"
    evaluation_output_dir.mkdir(exist_ok=True)

    # Log output_path_nerf and output_config_path in the wandb config
    wandb.config.update({"output_path_nerf": str(output_path_nerf), "output_config_path": str(output_config_path)})

    nerfevaluator = NerfEvaluator(
        config_path=output_config_path, 
        camera_path_test=Path(cfg_obj.evaluation.camera_pose_path_test),
        gt_images_dir=cfg_obj.evaluation.gt_images_dir, 
        output_dir=evaluation_output_dir,
        )
    nerfevaluator.save_rendered()
    nerfevaluator.compute_metrics()
    
    if cfg_obj.evaluation.alignment.flag:
        # Running the function on the synthetic images to get the visualizations
        output_dir = evaluation_output_dir / "alignment"
        output_dir = Path(nerf_obj.output_dir).parent / "alignment"
        output_dir.mkdir(exist_ok=True)

        evaluate_and_visualize_alignment(
            image_directory=Path(data_obj.config.input_dir) if data_obj.skip_save else Path(data_obj.config.output_dir),
            output_directory=output_dir, 
            keypoint_detector=cfg_obj.evaluation.alignment.keypoint_detector,  
            matcher_distance=cfg_obj.evaluation.alignment.matcher_distance, 
            match_filter=cfg_obj.evaluation.alignment.match_filter, 
            matched_keypoints_threshold=cfg_obj.evaluation.alignment.matched_keypoints_threshold
            )

    # evaluate_compare_poses(
    #     camera_path1=Path(cfg_obj.evaluation.camera_pose_path_train), 
    #     camera_path2=output_camera_path, 
    #     angular_error_max_dist=15, 
    #     translation_error_max_dist=0.25, 
    #     output_dir=evaluation_output_dir)

    wandb.finish()

if __name__ == "__main__":
    main()
