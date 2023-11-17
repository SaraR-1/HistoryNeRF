from coolname import generate_slug
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pathlib import Path
import wandb

from historynerf.config import (Config, DataPreparationConfig, PoseEstimationConfig,
                               NeRFConfig, EvaluationConfig, SamplingConfig)
from historynerf.data_preparation import DataPreparation
from historynerf.colmap_load import ColmapLoader
from historynerf.nerfstudio_wrapper import NSWrapper
from historynerf.evaluation import NerfEvaluator, evaluate_and_visualize_alignment
from historynerf.utils import register_configs

root_dir = Path(__file__).parents[1]

CONFIGURATIONS = [
    ('base', 'base_parent', Config),
    ('data_preparation', 'base_data', DataPreparationConfig),
    ('data_preparation/sampling', 'base_sampling', SamplingConfig),
    ('pose_estimation', 'base_pose_estimation', PoseEstimationConfig),
    ('nerf', 'base_nerf', NeRFConfig),
    ('evaluation', 'base_evaluation', EvaluationConfig)
]


def initialize_wandb(cfg_obj: dict, experiment_name: str) -> None:
    """Initialize the Weights and Biases logging."""
    api = wandb.Api()
    experiment_id = api.runs(f"{cfg_obj.wandb_entity}/{cfg_obj.wandb_project}", filters={"config.experiment_name": experiment_name})[0].id
    if cfg_obj.wandb_log:
        print("Resume W&B.")
        wandb.init(
            project=cfg_obj.wandb_project, 
            id=experiment_id, 
            resume=True, 
            config=cfg_obj
            )
    else:
        wandb.init(project=cfg_obj.wandb_project, mode="disabled")

register_configs(CONFIGURATIONS)
@hydra.main(config_path=str(root_dir / "configs"), config_name="parent_run", version_base="1.1")
def main(cfg: Config) -> None:
    """Main function to run the pipeline."""
    cfg_obj = OmegaConf.to_object(cfg)

    experiment_name = generate_slug(2)

    data_obj = DataPreparation(cfg.data_preparation)
    data_obj.save_images()

    if cfg.pose_estimation.colmap_model_path:
        colmap_loader = ColmapLoader(
            recon_dir=Path(cfg.pose_estimation.colmap_model_path),
            output_dir=Path(data_obj.config.output_dir).parent / "processed_data_fixedcolmap",
            imgs_dir=Path(data_obj.config.input_dir) if data_obj.skip_save else Path(data_obj.config.output_dir)
            )
        colmap_loader.undersample()
        cfg.pose_estimation["colmap_model_path"], data_obj.config.input_dir = colmap_loader.update_path(
            current_colmap=cfg.pose_estimation["colmap_model_path"], current_input=data_obj.config.input_dir)

    nerf_obj = NSWrapper(
        input_dir=data_obj.config.input_dir if data_obj.skip_save else None,
        output_dir=data_obj.config.output_dir,
        pose_estimation_config=cfg.pose_estimation,
        nerf_config=cfg.nerf,
        wandb_project=cfg.wandb_project,
        experiment_name=experiment_name
    )
    nerf_obj.run()

    initialize_wandb(cfg_obj, experiment_name)

    output_path_nerf = Path(nerf_obj.output_dir).parent / "nerf" / experiment_name / cfg_obj.nerf.method_name / "default"
    evaluation_output_dir = output_path_nerf / "evaluation"
    evaluation_output_dir.mkdir(exist_ok=True)

    nerf_obj.render(config_path=output_path_nerf / "config.yml", output_dir=evaluation_output_dir)

    wandb.log({"Training Sample Size": len(list(Path(nerf_obj.input_dir).glob("*.jpg")))})
    wandb.config.update({"output_path_nerf": str(output_path_nerf), "output_config_path": str(output_path_nerf / "config.yml")})

    nerfevaluator = NerfEvaluator(
        config_path=output_path_nerf / "config.yml",
        camera_path_test=Path(cfg_obj.evaluation.camera_pose_path_test),
        gt_images_dir=cfg_obj.evaluation.gt_images_dir,
        output_dir=evaluation_output_dir
        )
    nerfevaluator.save_rendered_images()
    nerfevaluator.compute_metrics()

    if cfg_obj.evaluation.alignment.flag:
        output_dir = Path(nerf_obj.output_dir).parent / "alignment"
        output_dir.mkdir(exist_ok=True)
        evaluate_and_visualize_alignment(
            image_directory=Path(data_obj.config.input_dir) if data_obj.skip_save else Path(data_obj.config.output_dir),
            output_directory=output_dir,
            keypoint_detector=cfg_obj.evaluation.alignment.keypoint_detector,
            matcher_distance=cfg_obj.evaluation.alignment.matcher_distance,
            match_filter=cfg_obj.evaluation.alignment.match_filter,
            matched_keypoints_threshold=cfg_obj.evaluation.alignment.matched_keypoints_threshold)

    wandb.finish()


if __name__ == "__main__":
    main()
