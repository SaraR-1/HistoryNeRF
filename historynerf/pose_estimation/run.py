from hydra.core.config_store import ConfigStore
from historynerf.pose_estimation.configs import COLMAPConfig, Config, OpenMVGConfig
from historynerf.pose_estimation import COLMAPPoseEstimator, OpenMVGEstimator, PoseEstimator

import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf


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
    estimator: PoseEstimator = estimation_map[type(cfg_obj.pose_config)](cfg_obj.pose_config)
    print("Start COLMAP pose estimation")
    estimator.estimate_poses()

if __name__ == "__main__":
    main()