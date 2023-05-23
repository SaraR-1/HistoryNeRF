from hydra.core.config_store import ConfigStore
from historynerf.pose_estimation.configs import PoseEstimationConfig, COLMAPConfig, Config
import hydra
from pathlib import Path

root_dir = Path(__file__).parents[2]

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="pose_estimation", name="colmap", node=COLMAPConfig)

@hydra.main(config_path=str(root_dir / "configs"), config_name="defaults", version_base="1.1")
def main(cfg: Config) -> None:
    # get instance of type PoseEstimator
    pose_estimator = hydra.utils.instantiate(cfg.pose_estimation)
    breakpoint()

if __name__ == "__main__":
    main()