import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pathlib import Path
import wandb

from historynerf.config import Config, DataPreparationConfig, PoseEstimationConfig, NeRFConfig, EvaluationConfig, SamplingConfig
from historynerf.data_preparation import DataPreparation

root_dir = Path(__file__).parents[1]

cs = ConfigStore.instance()
cs.store(name="base_parent", node=Config)
cs.store(group="data_preparation", name="base_data", node=DataPreparationConfig)
cs.store(group="data_preparation/sampling", name="base_sampling", node=SamplingConfig)
cs.store(group="pose_estimation", name="base_pose_estimation", node=PoseEstimationConfig)
cs.store(group="nerf", name="base_nerf", node=NeRFConfig)
# cs.store(group="evaluation_config", name="base_evaluation", node=EvaluationConfig)

@hydra.main(config_path=str(root_dir / "configs"), config_name="parent", version_base="1.1")
def main(cfg: Config) -> None:
    cfg_obj = OmegaConf.to_object(cfg)
    # print(cfg_obj)
    
    # Simple example to run data_preparation, here only with undersampling
    data_obj = DataPreparation(cfg.data_preparation)
    data_obj.save_images()

if __name__ == "__main__":
    main()


# data preparation