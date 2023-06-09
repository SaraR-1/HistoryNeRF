from hydra.core.config_store import ConfigStore
from historynerf.pose_estimation.configs import COLMAPConfig, Config, OpenMVGConfig
from historynerf.pose_estimation import COLMAPPoseEstimator, OpenMVGEstimator, PoseEstimator

import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import wandb


root_dir = Path(__file__).parents[2]

cs = ConfigStore.instance()
cs.store(name="base_parent", node=Config)
cs.store(group="pose_config", name="base_colmap", node=COLMAPConfig)
cs.store(group="pose_config", name="base_openmvg", node=OpenMVGConfig)

estimation_map = {
    COLMAPConfig: COLMAPPoseEstimator,
    OpenMVGConfig: OpenMVGEstimator
}

@hydra.main(config_path=str(root_dir / "configs"), config_name="parent", version_base="1.1")
def main(cfg: Config) -> None:
    cfg_obj = OmegaConf.to_object(cfg)
    print("Starting W&B.")
    # Add a flag to disable wandb from the config file
    if cfg_obj.wandb_log:
        wandb.init(
            project=cfg_obj.wandb_project, 
            config={
                **cfg['pose_config'],
                "method": estimation_map[type(cfg_obj.pose_config)].__name__,
            })
    else:
        wandb.init(project=cfg_obj.wandb_project, mode="disabled")
    print("W&B started.")
    estimator: PoseEstimator = estimation_map[type(cfg_obj.pose_config)](cfg_obj.pose_config)

    print("Start COLMAP pose estimation")
    estimator.estimate_poses()
    wandb.config.update({"output_dir": cfg_obj.pose_config.output_dir}, allow_val_change=True)
    print(f"COLMAP pose estimation finished. Output directory: {cfg_obj.pose_config.output_dir}")

    # Evaluate the estimated poses
    if cfg_obj.pose_config.gt_poses_dir:
        print("Start pose evaluation")
        aggregate_angular_error, aggregate_l2_translation = estimator.evaluate_poses()
        print("Pose evaluation finished")
        wandb.log({
            "Angular Error": aggregate_angular_error,
            "L2 Translation Error": aggregate_l2_translation
        })

    wandb.finish()

if __name__ == "__main__":
    main()