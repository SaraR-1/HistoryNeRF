import os
from pathlib import Path

from historynerf.config import PoseEstimationConfig, NeRFConfig

def argument_name_parser(argument):
    return argument.replace("_", "-")

def flag_to_argument(flag_name, flag_value):
    """
    Convert a flag to an argument. Remove "flag" from the flag name and add "no-" if the flag value is False.
    """
    flag_name = flag_name.replace("-flag", "")
    return flag_name if flag_value else f"no-{flag_name}"

def parse_arguments(arguments):
    """
    Parse the arguments to be used in the command line.

    Parameters
    ----------
    arguments : dict
        Dictionary with the arguments to be parsed.
    
    Returns
    -------
    arguments_command : str
        String with the arguments to be used in the command line.
    """
    arguments_command = ""
    for argument_name, argument_value in arguments.items():
        # Only add the argument if it is not None
        if argument_value:
            argument_name = argument_name_parser(argument_name)
            if "flag" in argument_name:
                argument_command = flag_to_argument(argument_name, argument_value)
            else:
                argument_command = f"{argument_name} {argument_value}"
            arguments_command += f" --{argument_command}"
    return arguments_command

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
        
        ns-train nerfacto --data ../data/bridge_of_sighs/output_temp/every5frames/processed_data --viewer.websocket-port 8501
        
        '''
        base_command = f"ns-process-data images --data {self.input_dir} --output-dir {self.output_dir_processed_data}"
        arguments_command = parse_arguments(self.pose_estimation_config)
        command = f"{base_command} {arguments_command}"
        os.system(command)

    def train(self):
        # Use --vis {wandb, tensorboard, viewer+wandb, viewer+tensorboard} to run with eval.
        base_command = f"ns-train {self.nerf_config.method_name} --data {self.output_dir_processed_data} --output-dir {self.output_dir_nerf}"
        # Parse arguments, exclude self.nerf_config.method_name
        nerf_config = {k:v for k, v in self.nerf_config.items() if k != "method_name"}
        arguments_command = parse_arguments(nerf_config)
        command = f"{base_command} {arguments_command}"
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
