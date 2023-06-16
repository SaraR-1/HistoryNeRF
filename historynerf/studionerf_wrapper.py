import os

from historynerf.config import PoseEstimationConfig, NeRFConfig

def argument_name_parser(argument):
    return argument.replace("_", "-")

def flag_to_argument(flag_name, flag_value):
    return flag_name if flag_value else f"no-{flag_name}"

class NSWrapper:
    def __init__(
            self,
            preprocessing_config: PoseEstimationConfig,
            nerf_config: NeRFConfig):
        
        self.preprocessing_config = preprocessing_config
        self.nerf_config = nerf_config

        self.initialize()
    
    def initialize(self):
        '''
        Depending of the NeRFStudio output, it could be good to create output directories within the specified output directory.
        '''
        pass

    def process_data(self):
        '''
        Process images into a nerfstudio dataset.
        1. Scales images to a specified size.
        2. Calculates the camera poses for each image using `COLMAP <https://colmap.github.io/>`_.

        Run example:
        ns-process-data images --data ../data/bridge_of_sighs/output_temp/every5frames/ --output-dir ../data/bridge_of_sighs/output_temp/processed_data
        '''
        # Parse all the arguments names, replace "_" with "-"

        # TODO: change as needed
        # 83 images, started at 12.38 - finished at ??.??
        command = f"ns-process-data images --data {DATA_PATH} --output-dir {PROCESSED_DATA_DIR}"

        os.system(command)

    def train(self):
        command = f"ns-train nerfacto --data {PROCESSED_DATA_DIR}"
        os.system(command)

    def run(self):
        self._process_data()
        self._train()
