from hydra.core.config_store import ConfigStore
from historynerf.configs import Config, PreprocessingConfig, COLMAPConfig, NeRFConfig, EvaluationConfig

import hydra
from pathlib import Path
from omegaconf import OmegaConf
import wandb

root_dir = Path(__file__).parents[1]

cs = ConfigStore.instance()
cs.store(name="base_parent", node=Config)
cs.store(group="data_config", name="base_data", node=PreprocessingConfig)
cs.store(group="pose_estimation_config", name="base_pose_estimation", node=COLMAPConfig)
cs.store(group="nerf_config", name="base_nerf", node=NeRFConfig)
cs.store(group="evaluation_config", name="base_evaluation", node=EvaluationConfig)

@hydra.main(config_path=str(root_dir / "configs"), config_name="parent", version_base="1.1")
def main(cfg: Config) -> None:
    cfg_obj = OmegaConf.to_object(cfg)
    print(cfg_obj)