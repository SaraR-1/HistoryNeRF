import hydra
from hydra.core.config_store import ConfigStore
from pathlib import Path
from historynerf.config import GoldStandardConfig, PoseEstimationConfig
from historynerf.data_preparation import video_preparation
from historynerf.nerfstudio_wrapper import NSWrapper
from historynerf.utils import register_configs

root_dir = Path(__file__).parents[1]

CONFIGURATIONS = [
    ('base', 'base_goldstandard', GoldStandardConfig),
    ('pose_estimation', 'base_pose_estimation', PoseEstimationConfig),
]

register_configs(CONFIGURATIONS)
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
    