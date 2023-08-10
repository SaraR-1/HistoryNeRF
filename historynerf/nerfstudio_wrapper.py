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
            pose_estimation_config: PoseEstimationConfig,
            nerf_config: NeRFConfig, 
            wandb_project: str,
            experiment_name: str,
            output_dir: str,
            input_dir: str=None):
        
        self.input_dir = output_dir if input_dir is None else input_dir
        self.output_dir = output_dir
        self.pose_estimation_config = pose_estimation_config
        self.nerf_config = nerf_config
        self.wandb_project = wandb_project
        self.experiment_name = experiment_name

        self.initialize()
    
    def initialize(self):
        '''
        Create output directories within the specified input directory.
        '''
        # Create colmap output dir if the flag for skip colmap is false
        if not self.pose_estimation_config.skip_colmap_flag:
            output_dir_processed_data = Path(self.output_dir).parents[0] / "processed_data"
            output_dir_processed_data.mkdir(exist_ok=True)
        else:
            output_dir_processed_data = Path(self.pose_estimation_config.colmap_model_path).parents[2]
            
        self.output_dir_processed_data = output_dir_processed_data

        output_dir_nerf = Path(self.output_dir).parents[0] / "nerf"
        output_dir_nerf.mkdir(exist_ok=True)
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
        base_command = f"ns-train {self.nerf_config.method_name} --data {self.output_dir_processed_data} --output-dir {self.output_dir_nerf} --timestamp 'default'"
        nerf_config = {k:v for k, v in self.nerf_config.items() if k not in ["method_name", "dataparser_name", "train_split_fraction", "disable_scene_scale"]}
        arg_string = dict_to_arg_string(nerf_config)
        command = f"{base_command} {arg_string}"

        if "wandb" in self.nerf_config.vis:
            command += f" --project_name {self.wandb_project}"
            command += f" --experiment-name {self.experiment_name}"

        command += f" {self.nerf_config.dataparser_name} --train-split-fraction {self.nerf_config.train_split_fraction}"
        if self.nerf_config.disable_scene_scale:
            command += " --orientation-method none --center-method none --auto-scale-poses False"
        # command += f" " 
        print(command)
        os.system(command)



    def render(self):
        # For rendering the video, the camera path must first be created and extracted manually using viewer
        pass
        # ns-render spiral --load-config=/workspace/data/bridge_of_sighs/output/gold_standard/nerf/slick-swan/nerfacto/2023-07-04_141837/config.yml --output-path=/workspace/data/bridge_of_sighs/output/gold_standard/nerf/slick-swan/nerfacto/2023-07-04_141837/output_spiral.mp4


    def run(self):
        self.process_data()
        self.train()
