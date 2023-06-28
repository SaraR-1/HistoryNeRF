import os
from pathlib import Path
import json

from historynerf.config import PoseEstimationConfig, NeRFConfig
from omegaconf.dictconfig import DictConfig


def argument_name_parser(argument):
    return argument.replace("_", "-")

def flag_to_argument(flag_name, flag_value):
    '''
    Convert a flag to an argument. Remove "flag" from the flag name and add "no-" if the flag value is False.
    '''
    flag_name = flag_name.replace("-flag", "")
    return flag_name if flag_value else f"no-{flag_name}"

def dict_to_arg_string(dictionary, prefix=''):
    '''
    Convert a dictionary to a string with the arguments to be used in the command line.
    '''
    result = ""
    for key, value in dictionary.items():
        if value is None:
            continue
        if isinstance(value, DictConfig):
            result += dict_to_arg_string(value, prefix=f'{prefix}{key}.')
        else:
            key = argument_name_parser(key)
            if 'flag' in key:
                flag_name = flag_to_argument(key, value)
                result += f'--{prefix}{flag_name} '
            else:
                result += f'--{prefix}{key} {value} '
    return result


class NSWrapper:
    def __init__(
            self,
            input_dir: str,
            pose_estimation_config: PoseEstimationConfig,
            nerf_config: NeRFConfig, 
            wandb_project: str,
            experiment_name: str,):
        
        self.input_dir = input_dir
        self.pose_estimation_config = pose_estimation_config
        self.nerf_config = nerf_config
        self.wandb_project = wandb_project
        self.experiment_name = experiment_name

        self.initialize()
    
    def initialize(self):
        '''
        Create output directories within the specified input directory.
        '''
        output_dir_processed_data = Path(self.input_dir).parents[0] / "processed_data"
        output_dir_processed_data.mkdir(exist_ok=True)

        output_dir_nerf = Path(self.input_dir).parents[0] / "nerf"
        output_dir_nerf.mkdir(exist_ok=True)

        self.output_dir_processed_data = output_dir_processed_data
        self.output_dir_nerf = output_dir_nerf

    def process_data(self):
        '''
        Process images into a nerfstudio dataset.
        1. Scales images to a specified size.
        2. Calculates the camera poses for each image using `COLMAP <https://colmap.github.io/>`_.

        Run example:
        ns-process-data images --data ../data/bridge_of_sighs/output_temp/every5frames/images --output-dir ../data/bridge_of_sighs/output_temp/every5frames/processed_data
        
        ns-train nerfacto --data ../data/bridge_of_sighs/output_temp/every5frames_II/processed_data --viewer.websocket-port 8501
        
        '''
        base_command = f"ns-process-data images --data {self.input_dir} --output-dir {self.output_dir_processed_data}"

        arg_string = dict_to_arg_string(self.pose_estimation_config)
        command = f"{base_command} {arg_string}"

        os.system(command)

    def train(self):
        # Use --vis {wandb, tensorboard, viewer+wandb, viewer+tensorboard} to run with eval.
        base_command = f"ns-train {self.nerf_config.method_name} --data {self.output_dir_processed_data} --output-dir {self.output_dir_nerf}"
        
        # Parse arguments, exclude self.nerf_config.method_name
        nerf_config = {k:v for k, v in self.nerf_config.items() if k != "method_name"}
        arg_string = dict_to_arg_string(nerf_config)
        command = f"{base_command} {arg_string}"

        if self.nerf_config.vis == "wandb":
            command += f" --project_name {self.wandb_project}"
            command += f" --experiment-name {self.experiment_name}"
            
        os.system(command)


    def render(self):
        # For rendering the video, the camera path must first be created and extracted manually using viewer
        pass

    def run(self):
        self.process_data()
        self.train()
