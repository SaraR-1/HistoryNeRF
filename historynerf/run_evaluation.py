import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pathlib import Path
import wandb

from nerfstudio.utils.eval_utils import eval_setup

from historynerf.config import EvaluationConfig
from historynerf.evaluation.nerf import NerfEvaluator 


root_dir = Path(__file__).parents[1]

cs = ConfigStore.instance()
cs.store(name="base_evaluation", node=EvaluationConfig)

@hydra.main(config_path=str(root_dir / "configs" / "evaluation"), config_name="evaluation", version_base="1.1")
def main(cfg: EvaluationConfig) -> None:
    cfg_obj = OmegaConf.to_object(cfg)

    config, pipeline, _, _ = eval_setup(config_path=Path(cfg_obj.config_path), eval_num_rays_per_chunk=None, test_mode="test")

    # Get the experiment id from the name
    api = wandb.Api()
    wandb_entity = "sara"
    experiment_id = api.runs(f"{wandb_entity}/{config.project_name}", filters={"config.experiment_name": config.experiment_name})[0].id
    print("Resume W&B.")
    # Add a flag to disable wandb from the config file
    wandb.init(
        project=config.project_name,
        id=experiment_id,
        resume=True,)

    Path(cfg_obj.output_dir).rename(Path(cfg_obj.output_dir).parent / "evaluation_old")

    nerfevaluator = NerfEvaluator(
        config_path=Path(cfg_obj.config_path), 
        camera_path_test=Path(cfg_obj.camera_pose_path_test),
        gt_images_dir=Path(cfg_obj.gt_images_dir), 
        output_dir=Path(cfg_obj.output_dir),
        )
    nerfevaluator.save_rendered()
    nerfevaluator.compute_metrics()

    wandb.finish()

if __name__ == "__main__":
    main()


