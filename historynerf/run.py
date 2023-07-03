from coolname import generate_slug
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pathlib import Path
import wandb

from historynerf.config import Config, DataPreparationConfig, PoseEstimationConfig, NeRFConfig, EvaluationConfig, SamplingConfig
from historynerf.data_preparation import DataPreparation
from historynerf.nerfstudio_wrapper import NSWrapper

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
    # Randomly generate a name for the experiment
    experiment_name = generate_slug(2)
    
    # Simple example to run data_preparation, here only with undersampling
    data_obj = DataPreparation(cfg.data_preparation)
    data_obj.save_images()

    nerf_obj = NSWrapper(
        input_dir=data_obj.config.output_dir,
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
            resume=True,)
    else:
        wandb.init(project=cfg_obj.wandb_project, mode="disabled")
    wandb.log({"prova": 5})
    wandb.config.update(cfg)
    wandb.finish()

if __name__ == "__main__":
    main()
