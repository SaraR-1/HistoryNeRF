import hydra
from hydra.core.config_store import ConfigStore
from pathlib import Path
from historynerf.config import GoldStandardConfig, PoseEstimationConfig
from historynerf.data_preparation import video_preparation
from historynerf.nerfstudio_wrapper import NSWrapper

root_dir = Path(__file__).parents[1]
cs = ConfigStore.instance()
cs.store(name="base_goldstandard", node=GoldStandardConfig)
cs.store(group="pose_estimation", name="base_pose_estimation", node=PoseEstimationConfig)

@hydra.main(config_path=str(root_dir / "configs"), config_name="parent_goldstandard", version_base="1.1")
def main(cfg: GoldStandardConfig) -> None:
    print(cfg)
    if not Path(cfg.input_dir).is_dir():
        video_preparation(input_dir=cfg.input_dir, output_dir=cfg.output_dir, overwrite_output=False)
        cfg.input_dir = Path(cfg.output_dir) / "frames"
        
    ns_object = NSWrapper(
        pose_estimation_config=cfg.pose_estimation, 
        output_dir=cfg.output_dir, 
        input_dir=cfg.input_dir
        )
    
    # Run colmap on all data
    ns_object.process_data()


if __name__ == "__main__":
    main()
    