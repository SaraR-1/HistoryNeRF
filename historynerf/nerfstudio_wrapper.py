import os
from pathlib import Path

from historynerf.config import PoseEstimationConfig, NeRFConfig
from omegaconf.dictconfig import DictConfig
from typing import Union, Optional


def argument_name_parser(argument: str) -> str:
    """Convert underscores in the argument to dashes."""
    return argument.replace("_", "-")

def flag_to_argument(flag_name: str, flag_value: bool) -> str:
    """Convert a flag name and value to a command-line argument.  Remove 'flag' from the flag name and add 'no-' if the flag value is False."""
    flag_name = flag_name.replace("-flag", "")
    return flag_name if flag_value else f"no-{flag_name}"

def dict_to_arg_string(dictionary: DictConfig[str, Union[DictConfig, str, bool, None]], prefix: str = '') -> str:
    '''
    Convert a dictionary to a string with the arguments to be used in the command line.
    '''
    result = []
    for key, value in dictionary.items():
        if value is None:
            continue
        if isinstance(value, DictConfig):
            result.append(dict_to_arg_string(value, prefix=f'{prefix}{key}.'))
        else:
            key = argument_name_parser(key)
            if 'flag' in key:
                flag_name = flag_to_argument(key, value)
                result.append(f'--{prefix}{flag_name}')
            else:
                result.append(f'--{prefix}{key} {value}')
    return ' '.join(result)


class NSWrapper:
    def __init__(self,
                 pose_estimation_config: PoseEstimationConfig,
                 nerf_config: NeRFConfig,
                 wandb_project: str,
                 experiment_name: str,
                 output_dir: str,
                 input_dir: Optional[str] = None):
        """
        Initialize the NSWrapper class.        
        """
        self.input_dir = Path(output_dir if input_dir is None else input_dir)
        self.output_dir = Path(output_dir)
        self.pose_estimation_config = pose_estimation_config
        self.nerf_config = nerf_config
        self.wandb_project = wandb_project
        self.experiment_name = experiment_name
        self.initialize()

    def initialize(self) -> None:
        """Create output directories within the specified directory."""
        condition = not self.pose_estimation_config.skip_colmap_flag
        output_dir_processed_data = self.output_dir.parents[0] / "processed_data" if condition else Path(self.pose_estimation_config.colmap_model_path).parents[2]
        output_dir_processed_data.mkdir(exist_ok=True)
        self.output_dir_processed_data = output_dir_processed_data

        output_dir_nerf = self.output_dir.parents[0] / "nerf"
        output_dir_nerf.mkdir(exist_ok=True)
        self.output_dir_nerf = output_dir_nerf

    def process_data(self) -> None:
        """Process images into a nerfstudio dataset, calculates the camera poses for each image using `COLMAP <https://colmap.github.io/>`_."""
        command = f"ns-process-data images --data {self.input_dir} --output-dir {self.output_dir_processed_data}"
        command += f" {dict_to_arg_string(self.pose_estimation_config)}"
        print(f"Running: \n{command}")
        os.system(command)

    def train(self) -> None:
        """Train the NeRF model."""
        base_command = f"ns-train {self.nerf_config.method_name} --data {self.output_dir_processed_data} --output-dir {self.output_dir_nerf} --timestamp 'default'"
        nerf_config_filtered = {k: v for k, v in self.nerf_config.items() if k not in ["method_name", "dataparser_name", "train_split_fraction", "disable_scene_scale"]}
        command = f"{base_command} {dict_to_arg_string(nerf_config_filtered)}"

        if "wandb" in self.nerf_config.vis:
            command += f" --project_name {self.wandb_project} --experiment-name {self.experiment_name}"
            
        command += f" {self.nerf_config.dataparser_name} --train-split-fraction {self.nerf_config.train_split_fraction}"

        if self.nerf_config.disable_scene_scale:
            command += " --orientation-method none --center-method none --auto-scale-poses False"

        print(f"Running: \n{command}")
        os.system(command)

    def render(self, config_path: str, output_dir: Path) -> None:
        """Save rendered video based on the training set."""
        command = f"ns-render interpolate --load-config {config_path} --output-path {output_dir / 'output.mp4'} --pose-source train"
        print(f"Running: \n{command}")
        os.system(command)

    def run(self) -> None:
        """Run the entire pipeline."""
        self.process_data()
        self.train()
